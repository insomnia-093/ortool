from __future__ import annotations

import argparse
import logging
import sys
import json 
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from ortools.sat.python import cp_model

# 全局配置路径与文件名
DEFAULT_EXCEL_PATH = Path(r"D:\ortool\虚拟医生数据.xlsx")
DEFAULT_OUTPUT_TXT_PATH = Path(r"D:\ortool\output.txt")
DEFAULT_SCHEDULE_EXCEL = "医生排班结果.xlsx"
LOG_FILE = "排班日志.log"

# 配置全局日志：控制台+文件双输出，记录运行信息与错误
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("physician_scheduler")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

# 读取排班配置文件（人员名单+约束+班次）
def read_schedule_config(config_path: str = "schedule_config.json") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path.absolute()}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"成功读取配置文件：{config_path.absolute()}")
    return config

# 医生数据类
@dataclass(frozen=True)
class Physician:
    name: str
    skills: Sequence[str]
    level: int
    max_shifts: int
    min_days_off: int
    preferred_shifts: Dict[str, int]
    has_night_qual: bool                # 新增：是否有夜班/深夜班资质

# 班次需求类
@dataclass(frozen=True)
class ShiftDemand:
    required: int
    required_skills: Sequence[str]
    min_level: int

# 排班问题核心数据类
@dataclass(frozen=True)
class ProblemData:
    physicians: Sequence[Physician]
    days: Sequence[str]
    shifts: Sequence[str]
    demand: Dict[str, Dict[str, ShiftDemand]]
    forbidden_sequences: Sequence[Tuple[str, str]]
    max_consecutive_days: int
    alpha: int = 10
    beta: int = 5
    gamma: int = 1
    max_time_seconds: int = 30
    num_workers: int = 8

# 核心排班调度器类
class PhysicianScheduler:
    # 初始化调度器：加载排班数据，初始化模型与核心变量容器  
    def __init__(self, data: ProblemData) -> None:
        self.data = data
        self.model = cp_model.CpModel()
        self.x: Dict[Tuple[int, int, int], cp_model.IntVar] = {}  # 医生×日期×班次 0/1分配变量
        self.is_working: Dict[Tuple[int, int], cp_model.IntVar] = {}  # 医生×日期 是否工作标识
        self.total_shifts: Dict[int, cp_model.IntVar] = {}  # 医生总班次统计
        self.overtime: Dict[int, cp_model.IntVar] = {}  # 医生加班数统计
        self.deviation: Dict[int, cp_model.IntVar] = {}  # 医生班次公平性偏差统计

    # 构建CP-SAT模型：初始化所有变量，添加所有硬性约束，构建MOO
    def build(self) -> None:
        data = self.data
        physicians = data.physicians
        days = data.days
        shifts = data.shifts
        num_phys = len(physicians)
        num_days = len(days)
        num_shifts = len(shifts)

        # 新增代码
        # 标记夜班/深夜班的班次索引（适配配置文件的班次类型）
        night_shift_names = ["夜班", "深夜班"]
        self.night_shift_idxs = [s_idx for s_idx, s in enumerate(shifts) if s in night_shift_names]
        logger.info(f"识别夜班/深夜班索引：{[(shifts[s], s) for s in self.night_shift_idxs]}")

        logger.info(f"初始化变量：{num_phys}医生 × {num_days}日期 × {num_shifts}班次 = {num_phys*num_days*num_shifts}个")
        for p in range(num_phys):
            for d in range(num_days):
                for s in range(num_shifts):
                    var_name = f"assign_p{p}_d{d}_s{s}"
                    self.x[(p, d, s)] = self.model.NewBoolVar(var_name)

        # 是否工作变量，与班次分配变量的关联约束
        for p in range(num_phys):
            for d in range(num_days):
                var_name = f"working_p{p}_d{d}"
                self.is_working[(p, d)] = self.model.NewBoolVar(var_name)
                self.model.Add(sum(self.x[(p, d, s)] for s in range(num_shifts)) == self.is_working[(p, d)])

        # 硬性约束：每人每日最多1个班次
        logger.info("添加约束：每人每日最多1个班次")
        for p in range(num_phys):
            for d in range(num_days):
                self.model.Add(sum(self.x[(p, d, s)] for s in range(num_shifts)) <= 1)

        # 硬性约束：各班次满足技能、人数、最低级别要求，   新增夜班资质约束
        logger.info("添加约束：各班次技能/人数/级别需求 + 夜班资质约束")
        for d_idx, day in enumerate(days):
            for s_idx, shift in enumerate(shifts):
                demand = data.demand[day][shift]
                eligible_phys = []
                senior_phys = []
                for p_idx, phys in enumerate(physicians):
                    if not all(req in phys.skills for req in demand.required_skills):
                        self.model.Add(self.x[(p_idx, d_idx, s_idx)] == 0)
                        continue
                    # 新增：夜班/深夜班专属约束 - 无资质直接禁止排班
                    if s_idx in self.night_shift_idxs and not phys.has_night_qual:
                        self.model.Add(self.x[(p_idx, d_idx, s_idx)] == 0)
                        # 优化：将debug改为info，确保日志能输出（原debug级别被屏蔽）
                        logger.info(f"医生{phys.name}无夜班资质，禁止排{day}-{shift}")     # 原代码：logger.debug(f"医生{phys.name}无夜班资质，禁止排{day}-{shift}")
                        continue
                    # 技能/资质都满足，加入候选列表
                    eligible_phys.append(self.x[(p_idx, d_idx, s_idx)])
                    if phys.level >= demand.min_level:
                        senior_phys.append(self.x[(p_idx, d_idx, s_idx)])
                    '''
                    原代码:
                    if all(req in phys.skills for req in demand.required_skills):
                        eligible_phys.append(self.x[(p_idx, d_idx, s_idx)])
                        if phys.level >= demand.min_level:
                            senior_phys.append(self.x[(p_idx, d_idx, s_idx)])
                    else:
                        self.model.Add(self.x[(p_idx, d_idx, s_idx)] == 0)
                    '''
                if not eligible_phys:
                    raise ValueError(f"【{day}-{shift}】无满足技能/夜班资质要求的医生")
                if not senior_phys:
                    raise ValueError(f"【{day}-{shift}】无满足最低级别{demand.min_level}的医生")
                self.model.Add(sum(eligible_phys) >= demand.required)
                self.model.Add(sum(senior_phys) >= 1)

        # 硬性约束：禁止指定的班次连续序列
        logger.info(f"添加约束：禁止班次序列 {data.forbidden_sequences}")
        forbidden_map = {(shifts.index(a), shifts.index(b)) for a, b in data.forbidden_sequences}
        for p in range(num_phys):
            for d in range(num_days - 1):
                for s_today, s_next in forbidden_map:
                    self.model.Add(self.x[(p, d, s_today)] + self.x[(p, d+1, s_next)] <= 1)

        # 硬性约束：限制最大连续工作天数
        logger.info(f"添加约束：最大连续工作{data.max_consecutive_days}天")
        max_consec = data.max_consecutive_days
        for p in range(num_phys):
            for start in range(num_days - max_consec):
                window = [self.is_working[(p, start + offset)] for offset in range(max_consec + 1)]
                self.model.Add(sum(window) <= max_consec)

        # 硬性约束：满足每人最小休息天数要求
        logger.info("添加约束：每人满足最小休息天数")
        for p_idx, phys in enumerate(physicians):
            total_working_days = sum(self.is_working[(p_idx, d)] for d in range(num_days))
            self.model.Add(total_working_days <= num_days - phys.min_days_off)

        # 初始化统计变量：总班次、加班数
        logger.info("初始化统计变量：总班次/加班/公平性偏差")
        for p in range(num_phys):
            total_var = self.model.NewIntVar(0, num_days * num_shifts, f"total_shifts_p{p}")
            self.total_shifts[p] = total_var
            self.model.Add(total_var == sum(self.x[(p, d, s)] for d in range(num_days) for s in range(num_shifts)))

            overtime_var = self.model.NewIntVar(0, num_days * num_shifts, f"overtime_p{p}")
            self.overtime[p] = overtime_var
            self.model.Add(total_var - physicians[p].max_shifts <= overtime_var)
            self.model.Add(total_var <= physicians[p].max_shifts + overtime_var)

        # 初始化公平性偏差变量：与理想班次值的绝对差
        total_demand = sum(data.demand[day][shift].required for day in days for shift in shifts)
        ideal_shifts = total_demand // num_phys if num_phys > 0 else 0
        logger.info(f"班次公平性理想值：总需求{total_demand} ÷ {num_phys}医生 = {ideal_shifts}班次/人")
        for p in range(num_phys):
            dev_var = self.model.NewIntVar(0, num_days * num_shifts, f"deviation_p{p}")
            self.deviation[p] = dev_var
            self.model.Add(self.total_shifts[p] - ideal_shifts <= dev_var)
            self.model.Add(ideal_shifts - self.total_shifts[p] <= dev_var)

        # 初始化班次偏好项：关联医生偏好与分配变量
        logger.info("初始化班次偏好项")
        preference_terms = []
        for p_idx, phys in enumerate(physicians):
            for s_idx, shift in enumerate(shifts):
                weight = phys.preferred_shifts.get(shift, 0)
                if weight != 0:
                    for d_idx in range(num_days):
                        preference_terms.append((weight, self.x[(p_idx, d_idx, s_idx)]))

        # 构建多目标优化函数：最小化（加班+公平偏差-偏好奖励）
        logger.info(f"构建目标函数：α={data.alpha}(加班) β={data.beta}(公平) γ={data.gamma}(偏好)")
        objective_terms = []
        objective_terms.extend(data.alpha * self.overtime[p] for p in range(num_phys))
        objective_terms.extend(data.beta * self.deviation[p] for p in range(num_phys))
        for coeff, var in preference_terms:
            objective_terms.append(-data.gamma * coeff * var)

        self.model.Minimize(sum(objective_terms))

    def solve(self) -> cp_model.CpSolver:
        data = self.data
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = data.max_time_seconds
        solver.parameters.num_search_workers = data.num_workers
        solver.parameters.log_search_progress = False

        logger.info(f"启动求解器：最大时间{data.max_time_seconds}s，工作线程{data.num_workers}个")
        result = solver.Solve(self.model)

        logger.info(f"求解完成 | 状态：{solver.StatusName(result)} | 耗时：{solver.WallTime():.2f}秒")
        logger.info(f"目标函数值：{solver.ObjectiveValue()}")

        if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"未找到可行解！求解状态：{solver.StatusName(result)}，请检查约束是否过严或数据是否正确")
        return solver

    def extract_schedule(self, solver: cp_model.CpSolver) -> List[List[List[int]]]:
        data = self.data
        num_phys = len(data.physicians)
        num_days = len(data.days)
        num_shifts = len(data.shifts)
        schedule = []
        for p in range(num_phys):
            phys_row = []
            for d in range(num_days):
                day_row = [int(solver.Value(self.x[(p, d, s)])) for s in range(num_shifts)]
                phys_row.append(day_row)
            schedule.append(phys_row)
        logger.info("成功提取排班结果矩阵")
        return schedule

    @staticmethod
    def pretty_print(schedule: List[List[List[int]]], data: ProblemData) -> None:
        physicians = data.physicians
        days = data.days
        shifts = data.shifts

        print("\n" + "="*80)
        print("📋 医生排班表（直观版）| 休息标为「-」，多班次用「/」分隔")
        print("="*80)
        header = ["医生姓名"] + days
        print("\t".join(header))
        for p_idx, phys in enumerate(physicians):
            row = [phys.name]
            for d_idx in range(len(days)):
                assigned = [shifts[s] for s in range(len(shifts)) if schedule[p_idx][d_idx][s] == 1]
                row.append("/".join(assigned) if assigned else "-")
            print("\t".join(row))

        print("\n" + "="*120)
        print("📊 医生排班明细表（0=未排，1=已排）| 列：日期-班次")
        print("="*120)
        detail_header = ["医生姓名"] + [f"{d}-{s}" for d in days for s in shifts]
        print("\t".join(detail_header))
        for p_idx, phys in enumerate(physicians):
            flat_vals = [str(schedule[p_idx][d][s]) for d in range(len(days)) for s in range(len(shifts))]
            print("\t".join([phys.name] + flat_vals))
        print("="*120)

    def export_schedule_to_excel(self, schedule: List[List[List[int]]], solver: cp_model.CpSolver, output_path: str) -> None:
        data = self.data
        physicians = data.physicians
        days = data.days
        shifts = data.shifts
        num_phys = len(physicians)
        num_days = len(days)
        output_path = Path(output_path)

        # 新增:标记夜班/深夜班索引
        night_shift_idxs = [s_idx for s_idx, s in enumerate(shifts) if s in ["夜班", "深夜班"]]

        logger.info(f"开始导出排班结果到Excel：{output_path.absolute()}")
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # 构建直观排班表DataFrame
            df_schedule = pd.DataFrame(columns=["医生姓名"] + days)
            for p_idx, phys in enumerate(physicians):
                row_data = [phys.name]
                for d_idx in range(num_days):
                    assigned = [shifts[s] for s in range(len(shifts)) if schedule[p_idx][d_idx][s] == 1]
                    row_data.append("/".join(assigned) if assigned else "-")
                df_schedule.loc[p_idx] = row_data
            df_schedule.to_excel(writer, sheet_name="排班表", index=False)

            # 构建医生统计信息DataFrame         -新增夜班数以及是否有夜班资质
            df_stats = pd.DataFrame(columns=[
                "医生姓名", "是否有夜班资质", "总班次", "夜班/深夜班数", "最大可排班次", "加班数", "公平性偏差",
                "工作天数", "休息天数", "要求最小休息天数", "是否满足休息要求", "是否加班"
            ])
            for p_idx, phys in enumerate(physicians):
                total = solver.Value(self.total_shifts[p_idx])
                overtime = solver.Value(self.overtime[p_idx])
                dev = solver.Value(self.deviation[p_idx])
                work_days = sum(solver.Value(self.is_working[(p_idx, d)]) for d in range(num_days))
                rest_days = num_days - work_days
                is_rest_ok = "是" if rest_days >= phys.min_days_off else "否"
                is_overtime = "是" if overtime > 0 else "否"
                # 新增：统计夜班/深夜班数
                night_shift_num = sum(
                    schedule[p_idx][d_idx][s_idx] for d_idx in range(num_days) for s_idx in night_shift_idxs
                )
                # 新增：夜班资质标识
                night_qual = "是" if phys.has_night_qual else "否"

                # 新增night_qual以及night_shift_num
                df_stats.loc[p_idx] = [
                    phys.name, night_qual, total, night_shift_num, phys.max_shifts, overtime, dev,
                    work_days, rest_days, phys.min_days_off, is_rest_ok, is_overtime
                ]
            df_stats.to_excel(writer, sheet_name="医生统计信息", index=False)

        logger.info(f"Excel导出完成：共{num_phys}名医生，保存至{output_path.absolute()}")

# 辅助函数：将医院医生源数据Excel文件导出为TXT格式，便于数据查看
def export_excel_to_text(excel_path: str, output_path: str, encoding: str = "utf-8") -> None:
    excel_path = Path(excel_path)
    output_path = Path(output_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"源Excel文件不存在：{excel_path.absolute()}")

    df = pd.read_excel(excel_path)
    # 修改
    required_cols = ["姓名", "职位", "科室", "亚专科", "上班次数（天数）", "夜班资质", "班次偏好"]
    # 原代码：required_cols = ["姓名", "职位", "科室", "分类", "上班时间", "主治病状"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"源Excel缺少必要列：{', '.join(missing_cols)}")

    lines = []
    for _, row in df.iterrows():
        fields = [str(row[col]).strip() if pd.notna(row[col]) else "" for col in required_cols]
        lines.append(" | ".join(fields))
    output_path.write_text("\n".join(lines), encoding=encoding)

    logger.info(f"源数据导出完成：{len(lines)}条记录 → {output_path.absolute()}")

def read_physician_from_excel(excel_path: str, filter_names: list | None = None) -> List[Physician]:  
    if filter_names is None:
        filter_names = []
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"目标医生数据Excel不存在：{excel_path.absolute()}")
    
    # 读取Excel，处理空值
    df = pd.read_excel(excel_path).fillna("")
    COLUMN_MAP = {      # 新增：夜班资质列，班次偏好列
        "name": "姓名",               
        "dept": "科室",                
        "sub_dept": "亚专科",          
        "position": "职位",            
        "max_shifts": "上班次数（天数）",
        "night_qual": "夜班资质",
        "pref_shifts": "班次偏好"
    }
    
    required_cols = [COLUMN_MAP[k] for k in COLUMN_MAP.keys()]    #修改了原有循环["name", "dept", "sub_dept", "position", "max_shifts"]]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"目标Excel缺少必选列：{', '.join(missing_cols)}，请补充后重试")
    
    # 【需要你确认】职位→数字级别的映射（根据你的实际职位名称修改）
    # 适配你的Excel职位的映射（直接复制替换原字典）
    POSITION_TO_LEVEL = {
        "主任医师": 3,       # 高级（最高级别）
        "副主任医师": 3,     
        "科室主任、副主任医师": 2,     # 中级
        "科室主任、主任医师": 3,
        "副教授、副主任医师": 2,
        "junior主任医师": 2,
        "讲师、主任医师": 3,
        "讲师、主治医师": 2,
        "院长、主任医师": 3,
        "副院长、主任医师": 3,
        "副院长、副主任医师": 3,
        "教授、主任医师": 3,
        "主治医师": 2,       # 中级
        "住院医师": 1,       # 初级
        "规培医师": 1,       # 初级
        "实习医师": 1        # 初级
    }
    
    physicians = [] 
    for idx, row in df.iterrows():
        # 1. 筛选人员名
        phys_name = row[COLUMN_MAP["name"]].strip()
        if filter_names and phys_name not in filter_names:
            logger.info(f"跳过非目标人员：{phys_name}")
            continue
        
        # 2. 合并科室+亚专科为技能列表
        dept = row[COLUMN_MAP["dept"]].strip()
        sub_dept = row[COLUMN_MAP["sub_dept"]].strip()
        skills = []
        if dept:
            skills.append(dept)
        if sub_dept:
            skills.append(sub_dept)
        skills = list(set(skills))  # 去重
        
        # 3. 职位映射为级别
        position = row[COLUMN_MAP["position"]].strip()
        if position not in POSITION_TO_LEVEL:
            raise ValueError(f"未知职位：{position}，请在POSITION_TO_LEVEL字典中添加映射")
        level = POSITION_TO_LEVEL[position]

        # 4. 最大可排班次，容错非法值（强制1-7）
        try:
            max_shifts = int(row[COLUMN_MAP["max_shifts"]])
            max_shifts = max(1, min(7, max_shifts))  # 强制限制在1-7天
        except (ValueError, TypeError):
            logger.warning(f"医生{phys_name}上班次数非法，默认设为5天")
            max_shifts = 5
        '''
        原代码：
        # 4. 最大可排班次（直接读取）
        max_shifts = int(row[COLUMN_MAP["max_shifts"]])
        '''
        
        # 5. 最小休息天数（若没有单独列，用“一周7天 - 上班次数”计算；若有单独列，替换这里）
        min_days_off = 7 - max_shifts  # 假设一周7天，休息天数=7-上班天数
        # 【可选】若有单独的“最小休息天数”列，替换为：min_days_off = int(row["你的列名"])
        
        # 6. 班次偏好（表头没有，设为空字典）
        preferred_shifts = {}

        # 7. 新增：读取夜班资质，统一格式（是/否）
        night_qual_str = row[COLUMN_MAP["night_qual"]].strip()
        has_night_qual = True if night_qual_str in ["是", "有", "1", "Y"] else False

        # 8. 新增：解析班次偏好（格式：上午:3,下午:1 → 字典），容错格式错误
        pref_str = row[COLUMN_MAP["pref_shifts"]].strip()
        preferred_shifts = {}
        if pref_str and pref_str != "无":
            try:
                for item in pref_str.split(","):
                    shift, weight = item.split(":")
                    shift = shift.strip()
                    weight = int(weight.strip())
                    if weight > 0:
                        preferred_shifts[shift] = weight
            except:
                logger.warning(f"医生{phys_name}班次偏好格式错误（{pref_str}），忽略偏好")
                preferred_shifts = {}
        
        # 构建Physician实例
        phys = Physician(
            name=phys_name,
            skills=skills,
            level=level,
            max_shifts=max_shifts,
            min_days_off=min_days_off,
            preferred_shifts=preferred_shifts,
            has_night_qual = has_night_qual  # 新增：传入夜班资质
        )
        physicians.append(phys)

    # 修改
    logger.info(f"从目标Excel成功读取 {len(physicians)} 名医生数据（含{sum(1 for p in physicians if p.has_night_qual)}名有夜班资质）")
    # 原代码：logger.info(f"从目标Excel成功读取 {len(physicians)} 名医生数据")
    return physicians

# 程序主入口：解析命令行参数，分支执行源数据导出或排班求解核心逻辑
def main() -> None: 
    # ====================== 第一步：定义参数解析器 ======================
    parser = argparse.ArgumentParser(description="📌 OR-Tools CP-SAT 医生排班系统 | 支持配置文件+人员筛选+Excel导出")
    # 原有参数保留
    parser.add_argument("--excel", default=None, help="医院医生源数据Excel路径（仅导出TXT用）")
    parser.add_argument("--export-txt", default=None, help="源数据导出TXT路径（需与--excel同时使用）")
    parser.add_argument("--skip-solver", action="store_true", help="仅导出源数据，不运行排班求解器")
    parser.add_argument("--use-default", action="store_true", help="使用默认路径导出源数据，跳过所有其他逻辑")
    parser.add_argument("--output-excel", default=DEFAULT_SCHEDULE_EXCEL, help=f"排班结果Excel导出路径（默认：{DEFAULT_SCHEDULE_EXCEL}）")
    parser.add_argument("--alpha", type=int, default=None, help="加班权重（覆盖配置文件，越大越避免加班）")
    parser.add_argument("--beta", type=int, default=None, help="公平性偏差权重（覆盖配置文件，越大越平均）")
    parser.add_argument("--gamma", type=int, default=None, help="班次偏好权重（覆盖配置文件，越大越满足偏好）")
    parser.add_argument("--max-time", type=int, default=None, help="求解器最大运行时间（秒，覆盖配置文件）")
    parser.add_argument("--workers", type=int, default=None, help="求解器工作线程数（覆盖配置文件）")
    parser.add_argument("--src-excel", required=True, help="目标医生真实数据Excel路径（必填，如D:\\ortool\\医院医生数据.xlsx）")
    
    parser.add_argument("--config", default="schedule_config.json", help="排班配置文件路径（默认：schedule_config.json）")
    parser.add_argument("--filter-names", nargs="+", help="临时筛选排班人员名（空格分隔，覆盖配置文件，如--filter-names 张三 李四）")

    # ====================== 第二步：解析参数（必须在所有分支/变量使用前） ======================
    args = parser.parse_args() 

    # ====================== 分支1：使用默认路径仅导出源数据 ======================
    if args.use_default:
        if not DEFAULT_EXCEL_PATH.exists():
            raise FileNotFoundError(f"默认源Excel文件不存在：{DEFAULT_EXCEL_PATH.absolute()}")
        export_excel_to_text(str(DEFAULT_EXCEL_PATH), str(DEFAULT_OUTPUT_TXT_PATH))
        logger.info("默认路径源数据导出完成，程序退出")
        return

    # ====================== 分支2：手动指定路径导出源数据 ======================
    if args.excel or args.export_txt:
        if not (args.excel and args.export_txt):
            raise ValueError("参数错误：--excel 和 --export-txt 必须同时提供")
        export_excel_to_text(args.excel, args.export_txt)
        if args.skip_solver:
            logger.info("源数据导出完成，跳过排班求解，程序退出")
            return

    # ====================== 分支3：核心逻辑：配置文件+人员筛选+排班求解 ======================
    logger.info("="*50 + " 开始执行医生排班求解 " + "="*50)
    
    # 1. 读取配置文件
    config = read_schedule_config(args.config)
    
    # 2. 人员筛选：命令行参数优先级 > 配置文件
    filter_names = args.filter_names if args.filter_names else config["筛选人员名单"]
    if filter_names:
        logger.info(f"🎯 目标排班人员：{filter_names}")
    else:
        logger.info("🎯 无人员筛选，将对Excel中所有医生排班")
    
    # 3. 从Excel读取医生数据（带筛选）      新增：夜班资质和班次偏好
    real_physicians = read_physician_from_excel(args.src_excel, filter_names)
    if not real_physicians:
        raise ValueError("❌ 筛选后无可用医生，请检查人员名单或Excel数据！")
    
    # 4. 解析配置文件中的参数（命令行参数覆盖配置文件）
    # 4.1 约束参数
    constraints = config["约束条件"]
    alpha = args.alpha if args.alpha else constraints["加班权重(alpha)"]
    beta = args.beta if args.beta else constraints["公平性权重(beta)"]
    gamma = args.gamma if args.gamma else constraints["班次偏好权重(gamma)"]
    max_consecutive_days = constraints["最大连续工作天数"]
    forbidden_sequences = [tuple(seq) for seq in constraints["禁止班次序列"]]  # 转元组
    
    # 4.2 班次配置
    shift_config = config["班次配置"]
    days = shift_config["排班日期"]
    shifts = shift_config["班次类型"]
    
    # 4.3 班次需求：转换为ShiftDemand对象（核心）
    demand = {}
    for day, day_demand in shift_config["各班次需求"].items():
        demand[day] = {}
        for shift, info in day_demand.items():
            demand[day][shift] = ShiftDemand(
                required=info["需要人数"],
                required_skills=info["必需技能"],
                min_level=info["最低级别"]
            )
    
    # 4.4 求解器参数
    solver_params = config["求解器参数"]
    max_time = args.max_time if args.max_time else solver_params["最大求解时间(秒)"]
    workers = args.workers if args.workers else solver_params["工作线程数"]
    
    # 5. 构建排班问题数据
    problem_data = ProblemData(
        physicians=real_physicians,
        days=days,
        shifts=shifts,
        demand=demand,
        forbidden_sequences=forbidden_sequences,
        max_consecutive_days=max_consecutive_days,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_time_seconds=max_time,
        num_workers=workers
    )
    
    # 6. 执行排班求解
    scheduler = PhysicianScheduler(problem_data)
    scheduler.build()
    solver = scheduler.solve()
    schedule_matrix = scheduler.extract_schedule(solver)
    scheduler.pretty_print(schedule_matrix, problem_data)
    scheduler.export_schedule_to_excel(schedule_matrix, solver, args.output_excel)

   
    logger.info("\n👨⚕️  医生排班统计信息：")
    night_shift_names = ["夜班", "深夜班"]           # 新增
    night_shift_idxs = [s_idx for s_idx, s in enumerate(shifts) if s in night_shift_names]  # 新增
    for p_idx, phys in enumerate(problem_data.physicians):
        total = solver.Value(scheduler.total_shifts[p_idx])
        overtime = solver.Value(scheduler.overtime[p_idx])
        rest_days = len(problem_data.days) - sum(solver.Value(scheduler.is_working[(p_idx, d)]) for d in range(len(problem_data.days)))
        night_num = sum(schedule_matrix[p_idx][d][s] for d in range(len(days)) for s in night_shift_idxs)
        logger.info(
            f"{phys.name} | 夜班资质：{'有' if phys.has_night_qual else '无'} | 总班次：{total} | 夜班数：{night_num} | "
            f"加班：{overtime} | 休息天数：{rest_days}（要求≥{phys.min_days_off}）"
        )
        # 原代码：(f"{phys.name} | 总班次：{total} | 加班：{overtime} | 休息天数：{rest_days}（要求≥{phys.min_days_off}）"))

    logger.info("="*50 + " 医生排班求解全部完成 " + "="*50)

# 程序启动入口
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序运行异常：{str(e)}", exc_info=True)
        sys.exit(1)
