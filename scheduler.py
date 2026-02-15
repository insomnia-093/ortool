"""
使用 OR-Tools CP-SAT 求解器实现护士排班。

本脚本会根据员工属性和约束设置，生成一个排班矩阵（护士 × 日期 × 班次）。
通用模板，根据数据源（如 CSV、数据库等）进行调整适配。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Nurse:
    name: str
    skills: Sequence[str]
    level: int  #医生级别
    max_shifts: int
    min_days_off: int
    preferred_shifts: Dict[str, int]  #正奖励，负惩罚


@dataclass(frozen=True)
class ShiftDemand:
    required: int
    required_skills: Sequence[str]
    min_level: int  # 最低医生级别，至少有一个护士满足
  

@dataclass(frozen=True)
class ProblemData:
    nurses: Sequence[Nurse]
    days: Sequence[str]
    shifts: Sequence[str]
    demand: Dict[str, Dict[str, ShiftDemand]]  #日期→班次→班次需求
    forbidden_sequences: Sequence[Tuple[str, str]]  #当日班次不能与次日班次相邻
    max_consecutive_days: int


class NurseScheduler:
    def __init__(self, data: ProblemData) -> None:
        self.data = data
        self.model = cp_model.CpModel()
        self.x: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
        self.is_working: Dict[Tuple[int, int], cp_model.IntVar] = {}
        self.total_shifts: Dict[int, cp_model.IntVar] = {}
        self.deviation: Dict[int, cp_model.IntVar] = {}
        self.overtime: Dict[int, cp_model.IntVar] = {}

    def build(self) -> None:
        nurses = self.data.nurses
        days = self.data.days
        shifts = self.data.shifts
        num_nurses = len(nurses)
        num_days = len(days)
        num_shifts = len(shifts)

        for n in range(num_nurses):
            for d in range(num_days):
                for s in range(num_shifts):
                    var_name = f"x_n{n}_d{d}_s{s}"
                    self.x[(n, d, s)] = self.model.NewBoolVar(var_name)

        for n in range(num_nurses):
            for d in range(num_days):
                var_name = f"work_n{n}_d{d}"
                self.is_working[(n, d)] = self.model.NewBoolVar(var_name)
                self.model.Add(sum(self.x[(n, d, s)] for s in range(num_shifts)) == self.is_working[(n, d)])

        for n in range(num_nurses):
            for d in range(num_days):
                self.model.Add(sum(self.x[(n, d, s)] for s in range(num_shifts)) <= 1)

        for d, day in enumerate(days):
            for s, shift in enumerate(shifts):
                demand = self.data.demand[day][shift]
                eligible_vars = []
                senior_vars = []
                for n, nurse in enumerate(nurses):
                    if all(req in nurse.skills for req in demand.required_skills):
                        eligible_vars.append(self.x[(n, d, s)])
                        if nurse.level >= demand.min_level:
                            senior_vars.append(self.x[(n, d, s)])
                    else:
                        self.model.Add(self.x[(n, d, s)] == 0)
                if eligible_vars:
                    self.model.Add(sum(eligible_vars) >= demand.required)
                else:
                    raise ValueError(f"无满足需求的护士 {day} {shift}")
                if senior_vars:
                    self.model.Add(sum(senior_vars) >= 1)
                else:
                    raise ValueError(f"无满足需求的高级护士 {day} {shift}")
        forbidden_map = {(self.data.shifts.index(a), self.data.shifts.index(b)) for a, b in self.data.forbidden_sequences}
        for n in range(num_nurses):
            for d in range(num_days - 1):
                for (s_today, s_next) in forbidden_map:
                    self.model.Add(self.x[(n, d, s_today)] + self.x[(n, d + 1, s_next)] <= 1)

        max_consec = self.data.max_consecutive_days
        for n in range(num_nurses):
            for start in range(0, num_days - max_consec):
                window = [self.is_working[(n, start + offset)] for offset in range(max_consec + 1)]
                self.model.Add(sum(window) <= max_consec)

        for n, nurse in enumerate(nurses):
            total_var = self.model.NewIntVar(0, len(days) * len(shifts), f"total_n{n}")
            self.total_shifts[n] = total_var
            self.model.Add(total_var == sum(self.x[(n, d, s)] for d in range(num_days) for s in range(num_shifts)))

            overtime = self.model.NewIntVar(0, len(days) * len(shifts), f"overtime_n{n}")
            self.overtime[n] = overtime
            self.model.Add(total_var - nurse.max_shifts <= overtime)
            self.model.Add(total_var <= nurse.max_shifts + overtime)

        total_demand = sum(self.data.demand[day][shift].required for day in days for shift in shifts)
        ideal = total_demand // num_nurses
        for n in range(num_nurses):
            deviation = self.model.NewIntVar(0, len(days) * len(shifts), f"deviation_n{n}")
            self.deviation[n] = deviation
            self.model.Add(self.total_shifts[n] - ideal <= deviation)
            self.model.Add(ideal - self.total_shifts[n] <= deviation)

        preference_terms = []
        for n, nurse in enumerate(nurses):
            for s, shift in enumerate(shifts):
                weight = nurse.preferred_shifts.get(shift, 0)
                if weight:
                    for d in range(num_days):
                        coeff = weight
                        preference_terms.append((coeff, self.x[(n, d, s)]))

        alpha = 10  # 加班权重
        beta = 5   # 公平权重
        gamma = 1  # 偏好奖励权重

        objective_terms = []
        objective_terms.extend(alpha * self.overtime[n] for n in range(num_nurses))
        objective_terms.extend(beta * self.deviation[n] for n in range(num_nurses))
        for coeff, var in preference_terms:
            objective_terms.append(-gamma * coeff * var)

        self.model.Minimize(sum(objective_terms))

    def solve(self, max_time_seconds: int = 30) -> cp_model.OptSolution:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time_seconds
        solver.parameters.num_search_workers = 8
        result = solver.Solve(self.model)
        if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("未找到可行解")
        return solver

    def extract_schedule(self, solver: cp_model.CpSolver) -> List[List[List[int]]]:
        nurses = self.data.nurses
        days = self.data.days
        shifts = self.data.shifts
        schedule = []
        for n in range(len(nurses)):
            nurse_row = []
            for d in range(len(days)):
                day_row = []
                for s in range(len(shifts)):
                    day_row.append(int(solver.Value(self.x[(n, d, s)])))
                nurse_row.append(day_row)
            schedule.append(nurse_row)
        return schedule

    @staticmethod
    def pretty_print(schedule: List[List[List[int]]], data: ProblemData) -> None:
        print("\n排班矩阵（护士 × 日期 × 班次）：")
        header = ["护士"] + [f"{day}-{shift}" for day in data.days for shift in data.shifts]
        print("\t".join(header))
        for n, nurse in enumerate(data.nurses):
            flat_assignments = []
            for d in range(len(data.days)):
                for s in range(len(data.shifts)):
                    flat_assignments.append(str(schedule[n][d][s]))
            print("\t".join([nurse.name] + flat_assignments))


def build_demo_problem() -> ProblemData:
    nurses = [
        Nurse(name="Alice", skills=["ICU", "ER"], level=3, max_shifts=5, min_days_off=2, preferred_shifts={"Morning": 2, "Night": -2}),
        Nurse(name="Bob", skills=["ER"], level=2, max_shifts=5, min_days_off=2, preferred_shifts={"Evening": 1}),
        Nurse(name="Carmen", skills=["ICU"], level=1, max_shifts=4, min_days_off=2, preferred_shifts={"Morning": 1}),
        Nurse(name="Diego", skills=["ER", "ICU"], level=2, max_shifts=5, min_days_off=2, preferred_shifts={"Night": -1}),
    ]

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    shifts = ["Morning", "Evening", "Night"]

    demand = {
        day: {
            "Morning": ShiftDemand(required=2, required_skills=["ICU"], min_level=2),
            "Evening": ShiftDemand(required=1, required_skills=["ER"], min_level=1),
            "Night": ShiftDemand(required=1, required_skills=["ICU"], min_level=2),
        }
        for day in days
    }

    forbidden_sequences = [("Night", "Morning")]

    return ProblemData(
        nurses=nurses,
        days=days,
        shifts=shifts,
        demand=demand,
        forbidden_sequences=forbidden_sequences,
        max_consecutive_days=3,
    )


def main() -> None:
    data = build_demo_problem()
    scheduler = NurseScheduler(data)
    scheduler.build()
    solver = scheduler.solve()
    schedule = scheduler.extract_schedule(solver)
    scheduler.pretty_print(schedule, data)

    print("\n每位护士的总班次：")
    for idx, nurse in enumerate(data.nurses):
        total = scheduler.total_shifts[idx]
        overtime = scheduler.overtime[idx]
        deviation = scheduler.deviation[idx]
        print(
            f"{nurse.name}: total={solver.Value(total)} overtime={solver.Value(overtime)} deviation={solver.Value(deviation)}"
        )


if __name__ == "__main__":
    main()
