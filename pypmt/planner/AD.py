# extended from QFUF.py

import time

import z3

from pypmt.planner.base import Search
from pypmt.planner.plan.smt_sequential_plan import SMTSequentialPlan
from pypmt.planner.utilities import dumpProblem
from pypmt.utilities import log

class ADSearch(Search):
    """
    Search scheme for encoders that use Achiever/Destroyer approach.
    """
    
    def search(self):
        self.horizon = 0

        log(f'Starting to solve', 1)
        total_time = 0
        for horizon in self.scheduler:
            self.horizon  = horizon
            start_time = time.time()
            formula = self.encoder.encode(self.horizon)

            if not self.solver:
                self.solver =  z3.Solver(ctx=self.encoder.ctx) if formula['objective'] == None else z3.Optimize(ctx=self.encoder.ctx)

            # add encoding specific for first step
            if self.horizon == 0:
                # execution of initial state action
                self.solver.add(formula['initial_exec'])
                # typing
                if not formula['typing'] == None:
                    self.solver.add(formula['typing'])
                # achiever constraint on first step
                if not formula['has_achiever_1'] == None:
                    self.solver.add(formula['has_achiever_1'])
                if not formula['is_achiever_1'] == None:
                    self.solver.add(formula['is_achiever_1'])
                # numeric precondition/effect on first step
                if not formula['numeric_1'] == None:
                    self.solver.add(formula['numeric_1'])
            
            # deal with the goal state
            g = z3.Bool(f"g{self.horizon}", self.encoder.ctx) # Now create a Boolean variable for assuming the goal
            reified_goal = z3.Implies(g, z3.And(formula['goal']))
            # print(reified_goal)
            self.solver.add(reified_goal) # Add the goal

            # We assert the rest of formulas to the solver
            self.solver.add(formula['initial_once'])
            if not formula['actions'] == None:
                self.solver.add(formula['actions'])
            if formula['goal_once'] is not None:
                self.solver.add(formula['goal_once'])

            if not formula['has_achiever'] == None:
                self.solver.add(formula['has_achiever'])
            if not formula['is_achiever'] == None:
                self.solver.add(formula['is_achiever'])
            if not formula['is_init_achiever'] == None:
                self.solver.add(formula['is_init_achiever'])

            if not formula['destroyers'] == None:
                self.solver.add(formula['destroyers'])

            if not formula['numeric'] == None:
                self.solver.add(formula['numeric'])

            # deal with the objective
            if not formula['objective'] == None:
                for o in formula['objective']:
                    if o[1] == self.encoder.metric_min_label:
                        self.solver.minimize(o[0] * g)
                    elif o[1] == self.encoder.metric_max_label:
                        self.solver.maximize(o[0] * g)

            # Check for satisfiability assuming the goal
            end_time = time.time()
            encoding_time = end_time - start_time
            start_time = time.time()
            res = self.solver.check(g)
            end_time = time.time()
            solving_time = end_time - start_time
            total_time = total_time + solving_time + encoding_time
            log(f'Step {horizon+1}/{(self.scheduler[-1]+1)} encoding: {encoding_time:.2f}s, solving: {solving_time:.2f}s', 2)
            log(f'Step {horizon+1}/{(self.scheduler[-1]+1)} memory: {self.solver.statistics().get_key_value("memory")}MB', 4)

            if res == z3.sat:
                log(f'Satisfiable model found. Took:{total_time:.2f}s', 3)
                log(f'Z3 statistics:\n{self.solver.statistics()}', 4)
                self.solution = self.encoder.extract_plan(self.solver.model(), self.horizon)
                break
        return self.solution

    def dump_smtlib_to_file(self, t, path):
        self.horizon = 0
        start_time = time.time()
        log(f'Encoding problem into a SMTLIB file', 1)
        for horizon in range(0, t):
            self.horizon  = horizon
            formula = self.encoder.encode(self.horizon)

            if not self.solver:
                self.solver =  z3.Solver(ctx=self.encoder.ctx) if 'objective' not in formula else z3.Optimize(ctx=self.encoder.ctx)
            
            # deal with the initial state
            if self.horizon == 0:
                self.solver.add(formula['initial_exec'])
                if not formula['typing'] == None:
                    self.solver.add(formula['typing'])
                if not formula['has_achiever_1'] == None:
                    self.solver.add(formula['has_achiever_1'])
                if not formula['is_achiever_1'] == None:
                    self.solver.add(formula['is_achiever_1'])

                if not formula['numeric_1'] == None:
                    self.solver.add(formula['numeric_1'])
            
            # deal with the goal state
            g = z3.Bool(f"g{self.horizon}", self.encoder.ctx) # Now create a Boolean variable for assuming the goal
            reified_goal = z3.Implies(g, z3.And(formula['goal']))
            # print(reified_goal)
            self.solver.add(reified_goal) # Add the goal

            # We assert the rest of formulas to the solver
            self.solver.add(formula['initial_once'])
            if not formula['actions'] == None:
                self.solver.add(formula['actions'])
            if formula['goal_once'] is not None:
                self.solver.add(formula['goal_once'])
            if not formula['has_achiever'] == None:
                self.solver.add(formula['has_achiever'])
            if not formula['is_achiever'] == None:
                self.solver.add(formula['is_achiever'])
            if not formula['is_init_achiever'] == None:
                self.solver.add(formula['is_init_achiever'])

            if not formula['destroyers'] == None:
                self.solver.add(formula['destroyers'])
            if not formula['numeric'] == None:
                self.solver.add(formula['numeric'])

            # deal with the objective
            if not formula['objective'] == None:
                for o in formula['objective']:
                    if o[1] == self.encoder.metric_min_label:
                        self.solver.minimize(o[0] * g)
                    elif o[1] == self.encoder.metric_max_label:
                        self.solver.maximize(o[0] * g)

        end_time = time.time()
        encoding_time = end_time - start_time
        self.solver.add(g) # we assert the goal happens in the last step (which would normally be an assumption)
        dumpProblem(self.solver, path, add_check_sat=True)
        log(f'Encoding the formula took: {encoding_time:.2f}s', 2)