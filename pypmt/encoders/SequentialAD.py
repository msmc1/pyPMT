import unified_planning.model
import z3

from collections import defaultdict

import unified_planning

from unified_planning.plans import SequentialPlan
from unified_planning.plans import ActionInstance

from unified_planning.shortcuts import FNode, Effect, EffectKind, Fraction, Object

from pypmt.planner.plan.smt_sequential_plan import SMTSequentialPlan
from pypmt.encoders.base import Encoder

class EncoderSequentialAchieverDestroyer(Encoder):
    
    def __init__(self, task, optimize:bool):
        self.name = "ad"
        self.task = task # The UP problem
        self.ctx = z3.Context() # The context where we will store the problem

        # Z3 EnumSort used to represent problem objects. This will be the type
        # for most of the action and fluent parameters (except the timestep)
        self.z3_objects_sort = None 
        # map from UP objects to Z3 objects and map from Z3 objects to UP objects
        self.up_objects_to_z3 = dict()
        self.z3_objects_to_up = dict()
        # Z3 EnumSort used to represent actions
        self.z3_actions_sort = None 

        # Z3 sort for index of plan slot
        self.z3_plan_ind_sort = z3.IntSort(ctx=self.ctx)
        self.z3_plan_ind_var = None

        # a function action(index) -> action object 
        # s.t. given a plan index, tells us which action is being executed
        self.z3_action_variable = None
        # this is a mapping from the UP actions (up.action) to an action object, encoding the selected action
        # and the other way: i.e., from the z3 action objects to the up action
        self.z3_actions_mapping = dict()
        self.up_actions_mapping = dict()
        # list of parameters used by the selected action. We need max(cardinality(A)) parameters.
        # Each one of them is a function param_k(index) -> object
        # The typing functions (self.z3_typing_functions) will be used to constraint the types.
        self.z3_action_parameters = []

        # maps action -> list of preconditions
        self.actions_prec = dict()
        self.max_prec_no = 0

        # Z3 sort for index of precondition
        self.z3_prec_ind_sort = z3.IntSort(ctx=self.ctx)

        # maps fluent -> set of (add action, [parameter index])
        self.fluents_adder = dict()
        # maps fluent -> set of (delete action, [parameter index])
        self.fluents_deleter = dict()
        # map numeric fluent -> set of (modify action, [parameter index])
        self.fluents_modifier = dict()

        # maps fluent added by initial state -> set of ([parameter]) so initial state is adder
        self.fluents_init_adder = dict()
        # maps fluent deleted by initial state -> set of ([parameter]) so initial state is deleter
        self.fluents_init_deleter = dict()
        
        # maps action -> set of (affected num fluent, [parameter index], effect)
        self.actions_num_effects = dict()
        # set of initial numeric values in form of numeric effects of initial state action
        self.init_num_effects = set()

        # combination of fluents_adder/fluents_deleter with actions_prec
        # maps action -> list of tuple(set of achievers/destroyers, [parameter index], [parameter constant])
        # one entry in list for each precondition, entry=None for numeric precondition
        # [parameter index] -> matches parameter of precondition fluent with parameter index of action key
        # [parameter constant] -> matches parameter of precondition fluent with constant that matches
        #                         with precondition of action key
        # achiever/destroyer: tuple(achiever/destroyer action, [parameter index], [parameter constant])
        # [parameter index/constant] in this tuple indicates how to match the precondition fluent with
        # the parameters of achiever/destroyer action
        self.actions_achievers = dict()
        self.actions_destroyers = dict()

        # similar to action_achievers, but only for initial state achiever
        # set of achievers is replaced by set of [parameters] such that precondition matches with initial stae
        self.actions_init_achievers = dict()

        # record actions that have some precondition that has no possible achiever
        self.impossible_actions = set()

        # a function achiever(i, l) -> b
        # It returns the index of the achiever for ith precondition of lth action
        self.z3_achiever_variable = None
        self.z3_modifier_ind_var = None # b

        # The encoding of the state
        self.z3_fluents = dict() # mapping from up.fluent.name to Z3_UF function

        # a function update_var(t) -> b
        # it returns the index of the action before t that lasts update the 
        # value of var
        # dictionary: var_name -> func update_var(t)
        self.z3_update_functions = dict()

        # from up.type.name to Z3_UF function that gets an object 
        # and returns a Bool saying if the object belongs to the type
        self.z3_typing_functions = dict()

        # Store the "raw" formula that we will later instantiate
        self.formula  = defaultdict(list)

        # optimization flag
        self.optimize = optimize
        # a function action_cost(t) -> c
        # it returns the action cost accummulated up to t
        self.z3_action_cost_variable = None
        # maps action -> cost
        self.action_costs = dict()
        # set of expressions to maximize/minimize
        self.maximize_objectives = set()
        self.minimize_objectives = set()
        # labels to indicate whether the expression should be maximize/minimize
        self.metric_max_label = "max"
        self.metric_min_label = "min"
        
        # Store the length of the formula
        self.formula_length = 0

        # setup the encoder
        self._setup_goal_action()
        self._setup_effects()
        self._setup_initial_action()
        self._setup_typing()
        self._setup_state()
        self._setup_preconditions()
        self._setup_achiever_destroyer()
        self._setup_actions()
        self._setup_objective()

    def __len__(self):
        return self.formula_length
        
    def _setup_initial_action(self):
        """!
        Sets up actions to represent the initial state, which has no precondition and 
        effects to set initial values.
        """

        # action representing the initial state, parameters include all objects with same name
        initial = unified_planning.model.InstantaneousAction("initial")

        # set initial values as effects
        for Fnode, value in self.task.initial_values.items():
            initial.add_effect(Fnode, value)

            fname = Fnode.fluent().name
            args = tuple(map(lambda p: p.constant_value(), Fnode.args))

            self.fluents_init_adder.setdefault(fname, set())
            self.fluents_init_deleter.setdefault(fname, set())

            # set up effect for initial action
            if(value.is_true()):
                self.fluents_init_adder.get(fname).add(args)
            elif(value.is_false()):
                self.fluents_init_deleter.get(fname).add(args)
            else:
                # self.init_num_effects.add((fname, value))
                self.init_num_effects.add(initial.effects[-1])
        
        # add initial state action to problem
        self.initial_action = initial
        # self.initial_action_args = self.task.all_objects
        self.task.add_action(initial)
        
    def _setup_goal_action(self):
        """!
        Sets up action to represent the goal state, which has the goals as its
        precondition and no effect
        """

        # action representing the goal state
        goal = unified_planning.model.InstantaneousAction("goal")

        # set goals as preconditions
        for g in self.task.goals:
            # # replace each object FNode to parameter FNode
            goal.add_precondition(g)

        # add goal state action to problem
        self.goal_action = goal
        self.task.add_action(goal)

    # copied from SequentialQFUF.py
    def _setup_typing(self):
        """!
        Map the objects and types in the UP problem to Z3 clauses.
        """
        # We create the Z3 EnumSort, with all problem objects in it
        # Then, we maintain the mapping between UP and Z3 objects
        self.z3_objects_sort, z3_objects = z3.EnumSort("object",
            list(map(lambda x: x.name, self.task.all_objects)),
            ctx=self.ctx)
        self.up_objects_to_z3 = dict(zip(self.task.all_objects, z3_objects))
        self.z3_objects_to_up = dict(zip(z3_objects, self.task.all_objects))

        if len(self.task.user_types) < 2:
            return

        # Now we create a function is_X for each type in the problem
        # This function maps from all possible objects to either True or False
        for type in self.task.user_types:
            self.z3_typing_functions[type.name] = z3.Function(f"is_{type.name}",
                                    self.z3_objects_sort, z3.BoolSort(ctx=self.ctx))

        # Finally, we create functions to enforce typing.
        # The functions ix_X will only map to true when we pass an object of that type
        # i.e., in a problem with two cars and one plane, we state:
        # is_car(car1) == True /\ is_car(car2) == True /\ is_car(plane1) == False
        for type in self.task.user_types:
            # get all z3 objects of the type "type"
            z3_objects_of_given_type = list(map(lambda x: self.up_objects_to_z3[x], self.task.objects(type)))
            self.formula['typing'].append(z3.And([
                    self.z3_typing_functions[type.name](x) == z3.BoolVal(True, ctx=self.ctx) 
                    if x in z3_objects_of_given_type else
                    self.z3_typing_functions[type.name](x) == z3.BoolVal(False, ctx=self.ctx)
                    for x in z3_objects
                ]))

    # copied from SequentialQFUF.py
    def _up_type_to_z3_type(self, type):
        """ Given a UP type, return the Z3 sort """
        if type.is_bool_type():
            return z3.BoolSort(ctx=self.ctx)
        elif type.is_user_type():
            # All user types are represented with the "object"
            # sort and filtered by the is_X() functions
            return self.z3_objects_sort
        elif type.is_real_type():
            return z3.RealSort(ctx=self.ctx)
        elif type.is_int_type():
            return z3.IntSort(ctx=self.ctx)
        else:
            raise Exception(f"UP type {type} still not supported")

    # copied from SequentialQFUF.py
    def _setup_actions(self):
        """!
        Create all the action execution infrastructure for Z3.
        We will have a UF named Exec, that gets a plan index and returns
        which action is being executed at that index.
        
        To store the parameters for the actions in each index, we will define
        too a set of param_x uninterpreted functions. We will have a number of
        those variables equal to the maximum number of parameters between all
        actions. These, similarly to the Exec function will given an index, is
        going to tell us to which object that parameter is being mapped to.

        There will be a UF named time, that gets a plan index and returns 
        the timestep the action at that index occurs.
        """
        # print(self.impossible_actions)

        # Define the actions sort for possible actions
        actions = set(self.task.actions) - self.impossible_actions
        self.z3_actions_sort, z3_actions = z3.EnumSort("action",
                list(map(lambda x: x.name, actions)), ctx=self.ctx)
        # Now map the UP actions to the corresponding Z3 object
        self.z3_actions_mapping = dict(zip(actions, z3_actions))
        self.up_actions_mapping = dict(zip(z3_actions, actions))

        # Define the function that will tell us which action is assigned to a plan slot
        self.z3_action_variable = z3.Function("Exec", self.z3_plan_ind_sort, self.z3_actions_sort)

        # find the max cardinality for all actions
        max_card = 0
        for action in actions:
            if len(action.parameters) > max_card:
                max_card = len(action.parameters)

        # for each plan index, create a function that given an index, returns us an object
        # these will be the parameters of each action assigned to a slot in plan.
        for i in range(0, max_card):
            action_parameter = z3.Function(f"param_{i}", self.z3_plan_ind_sort, self.z3_objects_sort)
            self.z3_action_parameters.append(action_parameter)

    # extended from SequentialQFUF.py
    def _setup_state(self):
        """!
        Creates a UF representation for each planning numeric fluent in UP
        """

        for fluent in self.task.fluents:
            # skip for non-numeric fluent
            if not (fluent.type.is_int_type() or fluent.type.is_real_type()):
                continue

            parameters = []
            # first add all the fluent parameters
            for p in fluent.signature:
                parameters.append(self._up_type_to_z3_type(p.type))
            # now add the timestep (int) and then return type
            parameters.append(self.z3_plan_ind_sort)
            parameters.append(self._up_type_to_z3_type(fluent.type))
            self.z3_fluents[fluent.name] = z3.Function(fluent.name, parameters)

            # create a function for each fluent that returns the last timestep up to t
            # where the fluent with the specified parameters is udpated
            # change the return type to timestep
            parameters[-1] = self.z3_plan_ind_sort
            # update_fluent(params, t) = b
            self.z3_update_functions.update({fluent.name: z3.RecFunction(f'update_{fluent.name}', parameters)})
        

    def _setup_effects(self):
        """!
        Setup dictionaries for the effects of actions with corresponding parameter indices
        """

        # each fluent maps to a default empty set
        for f in self.task.fluents:
            # identify adder/deleter for Boolean fluents
            if f.type.is_bool_type():
                self.fluents_adder.setdefault(f.name, set())
                self.fluents_deleter.setdefault(f.name, set())
            # identify modifier for numeric fluents
            elif f.type.is_int_type() or f.type.is_real_type():
                self.fluents_modifier.setdefault(f.name, set())

        # for each action
        for action in self.task.actions:
            # skip impossible action
            if action in self.impossible_actions:
                continue

            # list of action parameters' name and type
            action_params = list(map(lambda p : p.name, action.parameters))
            action_ptypes = list(map(lambda p : p.type, action.parameters))

            # each action maps to a default empty set for numeric effect
            self.actions_num_effects.setdefault(action, set())

            # for each effect
            for effect in action.effects:
                # print(effect) # debug

                fluent = effect.fluent # fluent in form of fNode: fluent(param)
                f_name = fluent.fluent().name # fluent name

                pis = [] # corresponding parameter index of the action (len = fluent arg no.)
                pts = [] # parameter type corresponding to action
                pcs = [] # constant pass as parameter of the fluent
                
                # for each parameter of the fluent in the effect
                for p in fluent.args:
                    # parameter of the action
                    if p.is_parameter_exp():
                        # get the index of the action parameter
                        pis.append(action_params.index(p.parameter().name))
                        pts.append(action_ptypes[pis[-1]])
                        pcs.append(None)
                    
                    # constant as effect fluent's parameter
                    else:
                        const = p.constant_value()
                        pis.append(None)
                        pcs.append(const)
                        if isinstance(const, Object):
                            pts.append(const.type)
                        elif isinstance(const, int):
                            pts.append(int)
                        elif isinstance(const, bool):
                            pts.append(bool)
                        elif isinstance(const, Fraction):
                            pts.append(Fraction)
                
                # add tuple (action, [param index]) to set for that fluent
                # in corresponding dictionary
                if(effect.value.is_true()): # add effect: register as fluent adder
                    self.fluents_adder.get(f_name).add((action, tuple(pis), tuple(pts), tuple(pcs)))

                elif(effect.value.is_false()): # delete effect: register as fluent deleter
                    self.fluents_deleter.get(f_name).add((action, tuple(pis), tuple(pts), tuple(pcs)))

                else: # numeric effect
                    # register as fluent modifier for update function
                    self.fluents_modifier.get(f_name).add((action, tuple(pis), tuple(pts), tuple(pcs)))

                    # register the numeric effect of the action for enforcing effect of action
                    self.actions_num_effects.get(action).add(
                        (f_name, tuple(pis), tuple(pts), tuple(pcs), effect)
                    )

    class Precondition:
        """
        Represents a precondition clause of an action, where precondition clause is extracted from the
        action's precondition in CNF form.
        - type: Boolean if the clause only contains boolean fluent such that it can be fully achieved/
                destroyed by initial state or an action, otherwise Numeric and expression must be
                evaluated during solving process to determine whether Precondition is fulfuilled
        - fNode: the FNode containing the precondition clause expression
        - fluents: a set of fluents contained in the precondition in form of Fluent object
        - in_not: whether the precondition clause is originally wrapped in a NOT
        """
        BOOL_TYPE = "bool"
        NUM_TYPE = "num"

        
        def __init__(self, prec, params, p_types, bVal):
            """
            Constructs a Precondition from given prec FNode assume it is a clause from precondition in CNF
            so no need to worry about AND, assume compiler already removes OR and IMPLIES

            @params
            - prec: the original precondition clause FNode
            - params: the parameters' name of the action that the precondition belongs to
            - p_types: the parameters' type of the action that the precondition belongs to
            - bVal: indicates whether the precondition requires the clause to be TRUE or FALSE, FALSE when
                    the original clause is wrapped in a NOT
            """
            # save the FNode itself
            self.fNode = prec
            self.type = None
            # get all of the fluents involved
            self.fluents = self.__extract_fluents(prec, params, p_types)

            # check from extracted fluents whether this is numeric or boolean
            # if contains at least one numeric fluent, must be numeric expression
            # UP does not support boolean equality -> won't have is(x) == 1
            # boolean precondition must be a single fluent
            # seems like UP can't mix boolean type with numeric type
            if len([x for x in self.fluents if x.is_number()]) > 0 or len(self.fluents) == 0:
                self.type = self.NUM_TYPE
            else:
                self.type = self.BOOL_TYPE

            self.in_not = not bVal

        def __str__(self):
            return str(self.fNode)

        def __extract_fluents(self, fnode, params, p_types):
            """
            Recursively extracts all fluents contained in the given FNode expression, and returns the fluents
            inside a set in form of Fluent objects.

            @params
            - fnode: the FNode to parse for fluents (expect FNode of precondition clause)
            - params: list of parameters' name from the relevant action
            - p_types: list of parameters' type matching params
            """
            fluents = set()

            # parse expression 
            if(fnode.is_fluent_exp()): # simple fluent
                fluent = fnode.fluent()
                # get fluent name
                f_name = fluent.name 

                # list to store the corresponding parameter index of the action (len = fluent arg no.)
                pis = [] 
                # list to store the type of the parameter
                pts = [] 
                # list to store the constant value passed as a parameter
                pcs = []
                # for each parameter of the fluent
                for p in fnode.args:
                    # action parameter as fluent parameter
                    if p.is_parameter_exp():
                        # get the index of the action parameter
                        pis.append(params.index(p.parameter().name))
                        pts.append(p_types[pis[-1]])
                        pcs.append(None)
                    # constant as fluent parameter
                    else:
                        const = p.constant_value()
                        pis.append(None)
                        pcs.append(const)
                        if isinstance(const, Object):
                            pts.append(const.type)
                        elif isinstance(const, int):
                            pts.append(int)
                        elif isinstance(const, bool):
                            pts.append(bool)
                        elif isinstance(const, Fraction):
                            pts.append(Fraction)

                # get the type of fluent
                f_type = fluent.type
                type = None
                if(f_type.is_bool_type()):
                    type = self.Fluent.BOOL_TYPE
                elif(f_type.is_int_type() or f_type.is_real_type()):
                    type = self.Fluent.NUM_TYPE

                # add a new Fluent Object for this fluent
                fluents.add(self.Fluent(f_name, type, pis, pts, pcs))
            
            # equality and arithmetic expressions
            elif(fnode.is_lt() or fnode.is_le() or fnode.is_equals() or
                 fnode.is_not() or fnode.is_plus() or fnode.is_minus() or 
                 fnode.is_times() or fnode.is_div() or fnode.is_dot()):
                # recursively calls for each component of the expression
                for fn in fnode.args:
                    # combines fluent set returned from the recursive calls
                    fluents = fluents | self.__extract_fluents(fn, params, p_types)
                
            return fluents
        
        def is_boolean(self):
            """
            Returns TRUE if the precondition clause only contains one Boolean fluent.
            """
            return self.type == self.BOOL_TYPE
            
        def is_number(self):
            """
            Returns TRUE if the precondition clause is one of the followings:
            - equality or inequality expression that involves at least one numeric fluent
            - equality or inequality expression between action parameters
            """
            return self.type == self.NUM_TYPE
        
        def get_fluent(self):
            """
            Returns the fluent(s) contained in the preconidition clause:
            - returns the Boolean fluent only for Boolean clause
            - returns the set of fluents for numeric clause
            """
            if self.is_boolean():
                for x in self.fluents:
                    return x
            else:
                return self.fluents
            
        def get_boolean_value(self):
            if self.is_boolean():
                return not self.in_not
            else:
                return None
            
        def get_expr_node(self):
            """
            Returns the FNode containing the precondition clause expression
            """
            if self.in_not:
                return self.fNode.Not()
            else:
                return self.fNode
        
        class Fluent:
            # constants for representing fluent type
            BOOL_TYPE = "bool"
            NUM_TYPE = "num"

            def __init__(self, name, type, params, p_types, p_consts):
                self.name = name
                self.type = type
                self.params = params
                self.p_types = p_types
                self.p_consts = p_consts

            def is_boolean(self):
                return self.type == self.BOOL_TYPE
            
            def is_number(self):
                return self.type == self.NUM_TYPE

    def _setup_preconditions(self):
        """!
        Setup dictionaries for the precondition clauses of actions with corresponding parameter indices.
        Each action maps to a list of precondition clauses, where each clause is in form of Precondition
        object.
        """

        # for each action
        for action in self.task.actions:

            # a list of preconditions
            precs = []
            # a list of action parameter name and type in order
            params = list(map(lambda p: p.name, action.parameters))
            p_types = list(map(lambda p: p.type, action.parameters))

            # flatten precondition so a long AND expression is broken into
            # smaller preconditions, ideally each with one fluent
            # for each Boolean precondition
            for prec in action.preconditions:
                # extract precondition clauses from each precondition of the action
                # and add to list
                precs.extend(self.extract_preconditions(params, p_types, prec, True))

            # update the dictionary: action -> list of preconditions
            self.actions_prec.update({action: precs})
            # keep track of the maximum number of precondition clauses an action has
            if len(precs) > self.max_prec_no:
                self.max_prec_no = len(precs)

    def extract_preconditions(self, params:list, p_types:list, fnode:FNode, bVal:bool):
        """!
        Assuming the given fnode is a CNF formula, return a list of clauses, 
        each represented by a Precondition object with the fluent(s) it contains
        and the corresponding parameter indices from params
        """
        preconditions = []

        # parse expression
        if(fnode.is_fluent_exp() or # simple fluent
           fnode.is_equals() or     # LHS = RHS
           fnode.is_lt() or         # LHS < RHS
           fnode.is_le()):          # LHS <= RHS
            # let Precondition constructor deal with extracting the fluents within
            preconditions.append(self.Precondition(fnode, params, p_types, bVal))
        
        if(fnode.is_and()): # AND expression
            for fn in fnode.args:
                preconditions.extend(self.extract_preconditions(params, p_types, fn, bVal))

        if(fnode.is_not()): # NOT expression
            preconditions.extend(self.extract_preconditions(params, p_types, fnode.args[0], not bVal))

        return preconditions

    def _setup_achiever_destroyer(self):
        """!
        Setup dictionaries for achievers and destroyers of actions and fluents, and 
        function to enforce achiever/destroyer structure
        """

        # each action maps to a list of Boolean indicating whether the ith precondition has
        # initial state as a possible achiever
        has_init_achs = dict()

        # matches up actions with achievers for each precondition and the parameter indices
        # for each action and their preconditions
        for a, ps in self.actions_prec.items():
            # print(f"action: {a.name}") # debug

            # ith entry: for ith precondition in precs
            achievers = []  # ith entry: tuple(set of achiever tuple, params from supported action)
            destroyers = [] # ith entry: tuple(set of destroyer tuple, params from destroyed action)
            init_achievers = []

            # for each precondition
            for p in ps:
                # print(p.get_expr_node()) # debug

                # skip for non-boolean precondition (achiever/destroyer not applicable)
                if not p.is_boolean():
                    achievers.append(None)
                    destroyers.append(None)
                    init_achievers.append(None)
                    continue

                achs = set()
                dess = set()
                init_achs = set()

                fluent = p.get_fluent() # get precondition fluent
                value = p.get_boolean_value() # check whether the fluent should be true for precondition to hold
                f_name = fluent.name # fluent name
                f_types = fluent.p_types # parameter type of precondition fluent
                f_const = fluent.p_consts # constants in parameter

                # get achievers and destroyers for this precondition
                if(value == True): # fluent being true
                    # register the fluent's adder as achiever
                    achs = self.fluents_adder.get(f_name) 
                    # register the fluent's deleter as destroyer
                    dess = self.fluents_deleter.get(f_name)
                    # tuple(p_consts)
                    init_achs = self.fluents_init_adder.get(f_name)

                elif(value == False): # fluent being false
                    achs = self.fluents_deleter.get(f_name)
                    dess = self.fluents_adder.get(f_name)
                    init_achs = self.fluents_init_deleter.get(f_name)

                # achs/dess: set of tuple(action, tuple(params), tuple(p_types), tuple(p_consts))
                # init_achs: set of tuple(p_consts) (constants that the fluent's parameter should be in order to be achieved by init)

                # filter out adder/deleter with mismatch types and constants
                # eg. drive(car,A,B) needs at(car,A), walk(man,A,B) makes at(man,B) true
                # walk is not an achiever of precondition at(car,A) of drive as type car != man and neither is subtype of the other
                # eg. Jon_move(A,B) needs at(Jon,A), Amy_move(A,B) makes at(Amy,B) true
                # Amy_move is not an achiever of precondition at(Jon,A) of Jon_move as Amy_move specifies the effect with mismatch
                # const Amy instead of Jon
                achs = set([x for x in achs if self.match_params_type(f_types, x[2]) and self.match_params_const(f_const, x[3])])
                dess = set([x for x in dess if not x[0] == self.initial_action 
                            and self.match_params_type(f_types, x[2]) and self.match_params_const(f_const, x[3])])

                init_achs = set([args for args in init_achs if self.match_params_type(f_types, [a.type for a in args])
                                 and self.match_params_const(f_const, args)])

                # ( set{tuple(achiever/destroyer, [matching params of fluent with ach/des])}, 
                #  [matching params of fluent with supported/destroyed action] )
                # all the lists in the same entry should have the same length
                achievers.append((achs, fluent.params, f_const))
                destroyers.append((dess, fluent.params, f_const))
                init_achievers.append((init_achs, fluent.params, f_const))

                # if there is no achiever for any of the precondition, add to impossible record
                if(len(achs) < 1 and len(init_achs) < 1):
                    self.impossible_actions.add(a)
            
            # update dictionaries
            self.actions_achievers.update({a: achievers})
            self.actions_destroyers.update({a: destroyers})
            self.actions_init_achievers.update({a: init_achievers})

            # record which precondition of the action has initial state as possible achiever
            has_init_achs.update({a: [x is not None and len(x[0]) > 0 for x in init_achievers]})

        # setup achiever function for pointing to the position of the achiver of specific ith precondition
        # for action at specific slot
        self.z3_achiever_variable = z3.Function("achiever", self.z3_prec_ind_sort, self.z3_plan_ind_sort, 
                                                self.z3_plan_ind_sort)

        # remove impossible achievers and destroyers from dictionaries
        change = len(self.impossible_actions) > 0
        # repeat removing impossible achievers from dictionaries until none is removed
        while change:
            change = False

            for action in self.task.actions:
                change = self.__remove_impossible_actions(action, self.actions_achievers, True, has_init_achs[action])

        # after all impossible actions are found from last step, remove impossible destroyers
        for action in self.task.actions:
            self.__remove_impossible_actions(action, self.actions_destroyers, False)
                    
    def __remove_impossible_actions(self, action, dict, is_achiever, has_init_achs=None):
        """
        Removes the given action from given dictionary if the action is impossible ie. has no
        possible achiever, and removes its possible achievers if those achiever actions are identified
        as impossible.  Returns whether the dictionary is modified.

        @params
        - action: the action to be considered
        - dict: the dictionary to remove impossible actions, should be either actions_achievers or actions_destroyers
        - is_achiever: indicates whether the given dictionary is for achievers
        - has_init_achs: a list of Boolean indicating whether the precondition clauses of the given action has
                         initial state as a possible achiever
        """
        # keep track of whether the dictionary is modified
        change = False

        # get the achievers entry for the given action
        entry = dict.get(action) if action in dict else None
                
        if(entry is not None):
            # remove entry for impossible action
            if action in self.impossible_actions:
                dict.pop(action)
                # print(f"{action.name} is one of impossible actions") # debug
                return True
            
            # check for achiever/destroyer set of each precondition
            for i in range(0, len(entry)):
                # skip for non-boolean precondition
                if entry[i] == None:
                    continue

                acts,f_params,f_const = entry[i]
                # get recorded no. of achievers for precondition i
                orig_len = len(acts)
                # add to impossible set if no achiever
                if orig_len < 1 and is_achiever and not has_init_achs[i]:
                    # print(f"{action.name} impossible for precondition {i}") # debug                    
                    self.impossible_actions.add(action)
                    dict.pop(action)
                    return True
                # get new set of achievers with all impossible ones removed
                acts = set([a for a in acts if a[0] not in self.impossible_actions])
                entry[i] = (acts, f_params, f_const)

                if len(acts) < 1 and is_achiever and not has_init_achs[i]:
                    # print(f"{action.name} impossible for precondition {i}") # debug
                    self.impossible_actions.add(action)
                    dict.pop(action)
                    return True

                if len(acts) < orig_len:
                    change = True

        return change
        
    def match_params_type(self, prec, ach):
        """!
        Compares whether the parameters' type of achiever matches with the precondition
        it supports.  It returns true when the parameter type of achiever is the same or
        a child of that of the precondition, or vice versa.  prec and ach are expected
        to be lists of same length that contains types
        """

        for i in range(len(prec)):
            child_prec = self.task.user_types_hierarchy[prec[i]]
            child_ach = self.task.user_types_hierarchy[ach[i]]
            # mismatch if 1) achiever type is not the same as precondition type
            # 2) achiever type is not a child of precondition type
            if (prec[i] != ach[i] and ach[i] not in child_prec and prec[i] not in child_ach):
                # print("mismatch") # debug
                return False

        return True
    
    def match_params_const(self, prec, ach):
        """!
        Compares whether the constants in the parameter of the precondition fluent matches
        with that of the achiever fluent.  prec and ach are expected to be lists of same
        length that contains either a constant or None.
        """
        for i in range(len(prec)):
            # skip if the parameter is not constant
            if(prec[i] == None or ach[i] == None):
                continue
            
            # mismatch if the constant does not match for both side
            # eg. prec: face(a,north), achiever: face(a,south)
            if (prec[i] != ach[i]):
                # print("mismatch") # debug
                return False

        return True

    def _setup_objective(self):
        """
        Setup objectives as expressions to be either minimized or maximized.  For objective
        of minimizing action costs, if among all possible actions there are at least two
        unique costs (ie. not all have same cost), a new function is created to keep track
        of the action costs of the plan and added as expression to minimize.  Currently ignore
        minimize sequential plan length as is doing by default.  Other plan quality metrics
        not supported at the moment.
        """
        if not self.optimize:
            return

        for metric in self.task.quality_metrics:
            if metric.is_minimize_expression_on_final_state():
                self.minimize_objectives.add(metric.expression)
            elif metric.is_maximize_expression_on_final_state():
                self.maximize_objectives.add(metric.expression)
            elif metric.is_minimize_action_costs():
                # print(metric)
                costs = set()
                # setup dict for possible action -> cost, including default cost
                for action in self.task.actions:
                    if(action == self.initial_action):
                        continue

                    if(action == self.goal_action):
                        continue

                    if(action in self.impossible_actions):
                        continue
                    
                    cost = metric.get_action_cost(action)
                    self.action_costs[action] = cost
                    costs.add(cost)

                # if possible actions do not all have same cost
                if len(costs) > 1:
                    # setup function for keeping track of action cost
                    self.z3_action_cost_variable = z3.Function("actions-cost", self.z3_plan_ind_sort, z3.IntSort)
            
            elif metric.is_minimize_sequential_plan_length():
                continue
            else:
                print("Metric type not supported: ")
                print(metric)

        # print(f"minimize: {self.minimize_objectives}") # debug
        # print(f"maximize: {self.maximize_objectives}") # debug
        # print(f"minimize action costs: {self.z3_action_cost_variable is not None}") # debug

    def encode_initial_state(self):
        """!
        Encodes formula defining initial state as the execution of an action, and 
        restrictions on its execution for any future step
        @return initial: a list of Z3 formula asserting the initial state
        """

        initial = []
        init_l = 0

        # assert it is happening before the first action in plan (ie. t = 0)
        action_matches = self.z3_action_variable(init_l) == self.z3_actions_mapping[self.initial_action]

        # asserting the initial values of numeric fluents
        num_effects = []

        for effect in self.init_num_effects:
            num_effects.append(self._expr_to_z3(effect, init_l))

        if self.z3_action_cost_variable is not None:
            num_effects.append(self.z3_action_cost_variable(init_l) == 0)

        initial.append(action_matches)
        initial.extend(num_effects)

        return initial

    def encode_initial_once(self):
        """!
        Encodes formula limiting the initialization of state to only occur once
        @return initial: a list of Z3 formula asserting the restriction on state initialization
        """
        l = self.z3_plan_ind_var
        # assert initial action does not happen anytime after t = 0
        action_once = self.z3_action_variable(l) != self.z3_actions_mapping[self.initial_action]

        return [action_once]

    def encode_goal_state(self):
        """!
        Encodes formula defining goal state as the execution of an action
        @return goal: a list of Z3 formula asserting the execution of the goal action after the last step
        """
        goal = []
        l = self.z3_plan_ind_var

        # assert it is happening after the last action in plan (ie. t = len(plan)+1)
        action_matches = self.z3_action_variable(l) == self.z3_actions_mapping[self.goal_action]

        goal.append(action_matches)
        # goal.extend(params_match)

        return goal

    def encode_goal_once(self):
        """!
        Encodes formula limiting the goal action to only occur once
        @return initial: a list of Z3 formula asserting the restriction on goal action occurence
        """
        l = self.z3_plan_ind_var
        # assert initial action does not happen anytime after t = 0
        action_once = self.z3_action_variable(l) != self.z3_actions_mapping[self.goal_action]

        return [action_once]

    # copied from SequentialQFUF.py
    def encode_actions(self):
        """!
        Encodes the Actions

        Enforce the parameter types of the actions when there are more than one object type in
        the problem
        Exec(t) = fly -> is_plane(param_1(t)) /\ is_city(param_2(t)) /\ is_city(param_3(t))

        @return actions: list of Z3 formulas asserting the actions
        """

        l = self.z3_plan_ind_var
        actions = []

        # for each action in unified planning
        for up_action in self.task.actions:
            if(up_action == self.initial_action):
                continue

            if(up_action == self.goal_action):
                continue

            if(up_action in self.impossible_actions):
                continue

            # constraint that says the action executed is up_action
            action_matches = self.z3_action_variable(l) == self.z3_actions_mapping[up_action]

            if len(self.task.user_types) < 2:
                continue

            # constraint that ensures the parameter types match
            action_typing = []
            for i in range(0, len(up_action.parameters)): # for each parameter
                type_str = up_action.parameters[i].type.name # get name of type of param_i
                typing_function = self.z3_typing_functions[type_str] # get matching type function
                action_parameter = self.z3_action_parameters[i] # param_i()
                action_typing.append(typing_function(action_parameter(l))) # is_plane(param1(t+1))

            if len(action_typing) > 0:
                actions.append(z3.Implies(action_matches, z3.And(action_typing)))

        return actions
    
    # Copied from SequentialQFUF.py
    def _expr_to_z3(self, expr, t, ctx=None, update=False):
        """
        Traverses a tree expression in-order and converts it to a Z3 expression.
        expr: The tree expression node. (Can be a value, variable name, or operator)
        t: The timestep for the Fluents to be considered 
        ctx: A context manager, as we need to take into account parameters from actions, fluents, etc ...
        Returns A Z3 expression or Z3 value.
        update: Flag to indicate whether the expression uses plain timestep or the index from update
        fluent function
        """
        if isinstance(expr, int): # A python Integer
            return z3.IntVal(expr, ctx=self.ctx)
        elif isinstance(expr, bool): # A python Boolean
            return z3.BoolVal(expr, ctx=self.ctx)
        elif isinstance(expr, Object): # a UP object
            return self.up_objects_to_z3[expr]
        elif isinstance(expr, Effect): # A UP Effect
            eff = None
            if expr.kind == EffectKind.ASSIGN:
                eff = self._expr_to_z3(expr.fluent, t + 1, ctx) == self._expr_to_z3(expr.value, t, ctx, True)
            if expr.kind == EffectKind.DECREASE:
                eff = self._expr_to_z3(expr.fluent, t + 1, ctx) == self._expr_to_z3(expr.fluent, t, ctx, True) - self._expr_to_z3(expr.value, t, ctx, True)
            if expr.kind == EffectKind.INCREASE:
                eff = self._expr_to_z3(expr.fluent, t + 1, ctx) == self._expr_to_z3(expr.fluent, t, ctx, True) + self._expr_to_z3(expr.value, t, ctx, True)
            if expr.is_conditional():
                return z3.Implies(self._expr_to_z3(expr.condition, t, ctx) , eff)
            else:
                return eff

        # TODO: Many operations are missing, but are trivial to add once needed
        elif isinstance(expr, FNode): # A UP FNode ( can be anything really )
            if expr.is_object_exp(): # A UP object
                return self.up_objects_to_z3[expr.object()]
            elif expr.is_constant(): # A UP constant
                return expr.constant_value()
            elif expr.is_or():  # A UP or
                return z3.Or([self._expr_to_z3(x, t, ctx, update) for x in expr.args])
            elif expr.is_and():  # A UP and
                return z3.And([self._expr_to_z3(x, t, ctx, update) for x in expr.args])
            elif expr.is_fluent_exp(): # A UP fluent
                f = expr.fluent() # the fluent
                p = [self._expr_to_z3(x, t, ctx) for x in expr.args] # its parameters translated
                # append update function for timestep for precondition
                # else append plain timestep
                p.append(self.z3_update_functions[f.name](*p,t) if update else t)
                return self.z3_fluents[f.name](p) # return the application
            elif expr.is_parameter_exp(): # A UP parameter
                p = expr.parameter()
                return ctx[p] # recover the param depending on the expression we are in
            elif expr.is_lt():
                return self._expr_to_z3(expr.args[0], t, ctx, update) < self._expr_to_z3(expr.args[1], t, ctx, update)
            elif expr.is_le():
                return self._expr_to_z3(expr.args[0], t, ctx, update) <= self._expr_to_z3(expr.args[1], t, ctx, update)
            elif expr.is_times():
                return self._expr_to_z3(expr.args[0], t, ctx, update) * self._expr_to_z3(expr.args[1], t, ctx, update)
            elif expr.is_div():
                return self._expr_to_z3(expr.args[0], t, ctx, update) / self._expr_to_z3(expr.args[1], t, ctx, update)
            elif expr.is_plus():
                return z3.Sum([self._expr_to_z3(x, t, ctx, update) for x in expr.args])
            elif expr.is_minus():
                return self._expr_to_z3(expr.args[0], t, ctx, update) - self._expr_to_z3(expr.args[1], t, ctx, update)
            elif expr.is_not():
                return z3.Not(self._expr_to_z3(expr.args[0], t, ctx, update))
            elif expr.is_equals():
                return self._expr_to_z3(expr.args[0], t, ctx) == self._expr_to_z3(expr.args[1], t, ctx, update)
            elif expr.is_implies():
                return z3.Implies(self._expr_to_z3(expr.args[0], t, ctx, update), self._expr_to_z3(expr.args[1], t, ctx, update))
            else:
                raise TypeError(f"Unsupported expression: {expr} of type {type(expr)}")
        elif isinstance(expr, Fraction):
            return z3.RealVal(f"{expr.numerator}/{expr.denominator}", ctx=self.ctx)
        else:
            raise TypeError(f"Unsupported expression: {expr} of type {type(expr)}")
        
    def encode_has_achiever(self):
        """!
        Encode that each action must have an achiever for each precondition before it:
        Exec(t) = a -> And{i=1..|prec(a)|} (Or{b=0..t-1} (achiever(i,t) = b))
        For the possible timestep of achiever to consider,
        - consider 0 if the precondition clause has initial state as possible achiever
        - consider any > 0 up to t-1 if the precondition clause has ordinary action as possible
          achiever, where this variable timestep is represented as b

        @return formula: list of Z3 formulas asserting each precondition has a preceding achiever
        """

        formula = []
        l = self.z3_plan_ind_var
        b = self.z3_modifier_ind_var

        for up_action in self.task.actions:
            # skip for initial action
            if(up_action == self.initial_action):
                continue

            # skip impossible action
            if(up_action in self.impossible_actions):
                continue

            # constraint that says the action executed is up_action
            action_matches = self.z3_action_variable(l) == self.z3_actions_mapping[up_action]
            
            achs = []
            # for each precondition clause
            for i in range(0, len(self.actions_prec[up_action])):
                # skip for numeric precondition
                if not self.actions_prec[up_action][i].is_boolean():
                    continue

                ach_exp = []
                # add achiever(i, t) = 0 for actions with initial achiever
                if len(self.actions_init_achievers[up_action][i][0]) > 0:
                    ach_exp.append(self.z3_achiever_variable(i, l) == 0)
                # add achiever(i, t) = b for actions with non-initial achiever
                if len(self.actions_achievers[up_action][i][0]) > 0:
                    ach_exp.append(self.z3_achiever_variable(i, l) == b)

                achs.append(z3.Or(ach_exp))

            if(len(achs) > 0):
                formula.append(z3.Implies(action_matches, z3.And(achs)))

        return formula

    def encode_is_achiever(self):
        """!
        Encode the semantics of achiever(i,t) for ordinary actions with b>0:
        Exec(t) = a /\ achiever(i,t) = b -> Or{a in achievers for ith precondition of Exec(t)}
            (Exec(b) = a /\ matching parameters)

        @return formula: list of Z3 formulas asserting the semantics of achiever function for
        ordinary actions as achievers
        """
        formula = []

        l = self.z3_plan_ind_var
        b = self.z3_modifier_ind_var

        # for each action
        for up_action in self.task.actions:
            # print(up_action.name) # debug

            # skip for initial action
            if(up_action == self.initial_action):
                continue

            # skip impossible action
            if(up_action in self.impossible_actions):
                continue

            # constraint that says the action executed is up_action
            action_matches = self.z3_action_variable(l) == self.z3_actions_mapping[up_action]

            # for achiever set entry of each precondition
            for i, ach_entry in enumerate(self.actions_achievers[up_action]):
                # skip for non-boolean precondition
                if(not self.actions_prec[up_action][i].is_boolean()):
                    continue

                # a list of conjunction (Exec(b) /\ matching params) for each possible achiever
                prec_achievers = []

                # constraint that says achiever for i precondition is at b
                achiever_at = self.z3_achiever_variable(i, l) == b
                # list of fluent parameter, corresponding parameter index from up_action
                f_action_params = ach_entry[1]
                f_action_const = ach_entry[2]

                # for each possible achiever for the precondition
                for achiever, f_achiever_params, _, f_const in ach_entry[0]:
                    # skip impossible action (should be removed in setup)
                    if(achiever in self.impossible_actions):
                        continue

                    # constraint that says Exec(b) = the achiever action
                    achiever_matches = self.z3_action_variable(b) == self.z3_actions_mapping[achiever]
                    # list of constraint that says the parameters match
                    params_match = []
                    # for each fluent parameter
                    for p in range(0,len(f_action_params)):
                        # no need to enforce equality if both are constants
                        if f_action_params[p] == None and f_achiever_params[p] == None:
                            continue

                        # LHS of equality expression: fluent parameter in terms of supported action's parameter or constant
                        left = self.z3_action_parameters[f_action_params[p]](l) if f_action_params[p] \
                               is not None else self._expr_to_z3(f_action_const[p], l)
                        # RHS of equality expression: fluent parameter in terms of achiever's parameter or constant
                        right = self.z3_action_parameters[f_achiever_params[p]](b) if f_achiever_params[p] \
                                is not None else self._expr_to_z3(f_const[p], b)
                        # LHS = RHS: match the parameters of supported action and the achiever
                        params_match.append(left == right)
                    
                    prec_achievers.append(z3.And(achiever_matches, *params_match))

                # Exec(t) /\ achiever(i,t)=b -> Exec(b)=ach_action /\ params
                if len(prec_achievers) > 0:
                    formula.append(z3.Implies(z3.And(action_matches, achiever_at),
                                          z3.Or(prec_achievers)))

        return formula
    
    def encode_is_init_achiever(self):
        """!
        Encode the semantics of achiever(i,t) with initial state as the achiever ie. b=0:
        Exec(t) = a /\ achiever(i,t) = 0 -> Or(match valid parameters combo of the supported
        action such that it matches with the initial state)

        @return formula: list of Z3 formulas asserting the semantics of achiever function for
        intial state achievers
        """
        formula = []
        l = self.z3_plan_ind_var

        for up_action in self.task.actions:
            # print(up_action.name) # debug

            # skip for initial action
            if(up_action == self.initial_action):
                continue

            # skip impossible action
            if(up_action in self.impossible_actions):
                continue

            # constraint that says the action executed is up_action
            action_matches = self.z3_action_variable(l) == self.z3_actions_mapping[up_action]

            # for achiever set entry of each precondition
            for i, init_ach_entry in enumerate(self.actions_init_achievers[up_action]):
                # skip non-boolean precondition
                if(not self.actions_prec[up_action][i].is_boolean()):
                    continue

                # fluent parameters in terms of up_action parameter indices
                f_action_params = init_ach_entry[1]

                # check the initial state is the achiever
                init_at = self.z3_achiever_variable(i,l) == 0
                init_achievers = []

                # for each fluent parameters supported by initial state
                for f_init_consts in init_ach_entry[0]:
                    params_match = []
                    # for each fluent parameter
                    for p in range(0,len(f_action_params)):
                        # skip if is also in terms of constant on up_action side
                        if f_action_params[p] is None:
                            continue

                        # match up_action's parameters with initial state
                        params_match.append(self.z3_action_parameters[f_action_params[p]](l) == 
                                            self.up_objects_to_z3[f_init_consts[p]])

                    if len(params_match) > 0:
                        init_achievers.append(z3.And(params_match))

                if len(init_achievers) > 0:
                    formula.append(z3.Implies(z3.And(action_matches, init_at),
                                                z3.Or(init_achievers)))

        return formula

    def __encode_update_functions(self):
        """!
        Define the update_fluent(params, t) functions in a recursive manner such that
        update_fluent(params, t) returns the index at which the fluent's value is last set.
        That will be one after the last action before t that updates the value of fluent(params).
        """

        # define the update functions (return one after the plan index of last action that updates fluent)
        for fluent in self.z3_update_functions:
            update_func = self.z3_update_functions[fluent]
            
            n = z3.Int('n', ctx=self.ctx)

            # create constants for each parameter of fluent
            f_params = [z3.Const(f'p_{i}', self.z3_objects_sort) 
                  for i in range(len(self.task.fluent(fluent).signature))]
            
            # update function should always return 1 for any slots before slot 1 including slot 1 itself
            base_case = n <= 1

            modifiers = []
            # get the actions that modify the fluents
            for action, a_params, _, a_const in self.fluents_modifier[fluent]:
                # skip initial action, already covered by base_case
                if(action == self.initial_action):
                    continue

                if(action in self.impossible_actions):
                    continue

                # action at the previous step
                action_matches = self.z3_action_variable(n-1) == self.z3_actions_mapping[action]

                # list of constraint that says the parameters match
                params_match = []
                for p in range(0,len(f_params)):

                    left = f_params[p]
                    right = self.z3_action_parameters[a_params[p]](n-1) if a_params[p] \
                            is not None else self._expr_to_z3(a_const[p], (n-1))
                    params_match.append(left == right)

                modifiers.append(z3.And(action_matches, *params_match))

            # print(modifiers) # debug

            # if the fluent has at least one action that modifies its value
            if(len(modifiers) > 0):
                z3.RecAddDefinition(update_func, [*f_params, n], 
                                    z3.If(base_case, 1, # base case: n<=1 -> 1
                                        z3.If(z3.Or(modifiers), n, # has modifier at n-1 -> n
                                                update_func(*f_params, n-1)) # else -> recursive with n-1 and same fluent parameters
                                        ) 
                                    ) 
            # else: initial state is the only one that sets the fluent's value, always return 1
            else:
                z3.RecAddDefinition(update_func, [*f_params, n], z3.IntVal(1, ctx=self.ctx))

    def encode_numeric(self):
        """
        Encode the numeric preconditions and effects of actions with the use of the update functions.
        
        Encode precondition using the update function to get the latest values of fluents involved:
        Exec(t) = a -> x(update_x(t)) < 5

        Encode effect using the update function to get the latest values of fluents used to set the
        new value of the affected fluent:
        Exec(t) = a -> y(t+1) = y(update_y(t)) + 4

        @return formula: list of Z3 formulas asserting numeric preconditions and effects of actions
        """
        self.__encode_update_functions()

        formula = []

        l = self.z3_plan_ind_var

        for action in self.task.actions:
            # skip for initial action
            if action == self.initial_action:
                continue

            if(action in self.impossible_actions):
                continue

            # print(f"numeric effect for {action.name}") # debug

            action_matches = self.z3_action_variable(l) == self.z3_actions_mapping[action]

            ctx = {} # context for the translation from FNode to z3
            # we add the mappings of the action parameter to the variable that selects the value
            # timestep given to action parameter variable should be update_f
            for i in range(0, len(action.parameters)):
                action_parameter = self.z3_action_parameters[i]
                ctx[action.parameters[i]] = action_parameter(l) 

            preconditions = []

            # get precondition list
            precs = self.actions_prec.get(action)

            # for each numeric precondition p
            for p in [x for x in precs if x.is_number()]:
                # eg value(c)+1 <= max_int()
                # with timestep: value(c,t)+1 <= max_int(t)
                # to value(param_0(t), update_value(c,t)+1) + 1 <= max_int(update_max_int(t))
                preconditions.append(self._expr_to_z3(p.get_expr_node(), l, ctx, True))

            # get numeric effects of action
            num_effects = self.actions_num_effects.get(action)

            effects = []

            # for each numeric effect e
            for _,_,_,_,e in num_effects:
                effects.append(self._expr_to_z3(e, l, ctx))

            # add effect on action cost if has minimize action costs as objective
            if self.z3_action_cost_variable is not None:
                effects.append(self.z3_action_cost_variable(l+1) == self.z3_action_cost_variable(l) + self.action_costs[action])

            if len(preconditions) > 0 or len(effects) > 0:
                formula.append(z3.Implies(action_matches, z3.And(*preconditions, *effects)))

        return formula

    def encode_destroyers(self):
        """!
        Encode destroyers: there must not be any destroyer between the action and achievers
        for any of its precondition, where the destroyer will undo the supporting effect of
        achievers
        
        forall 1<t<=L+1, 1<=b<t, i={1..|prec(Exec(t)|}, d in destroyers of ith precondition of Exec(t)
            Exec(t) != a \/ Exec(b) != d \/ achiever(i,t) > b \/ has non-matching parameter
        Equivalent expression
            !(Exec(t) = a /\ Exec(b) = d /\ achiever(i,t) < b /\ matching parameter)
        
        One of the following must not be true:
        - action a executed at t
        - action d, a destroyer of ith precondition of a at b (b < t)
        - achiever for ith precondition of a is before b ie. destroyer
        - d has matching parameters with a to destroy the precondition

        @return formula: list of Z3 formulas asserting there must not be any destroyer between the
        considered action and any of its achiever
        """
        formula = []

        l = self.z3_plan_ind_var
        b = self.z3_modifier_ind_var

        # for each action
        for up_action in self.task.actions:
            # skip for initial action
            if(up_action == self.initial_action):
                continue

            # skip impossible action
            if(up_action in self.impossible_actions):
                continue

            # print(up_action.name) # debug

            # constraint that says the action executed is up_action
            # not_action = self.z3_action_variable(a_l) != up_action
            action_matches = self.z3_action_variable(l) == self.z3_actions_mapping[up_action]

            # for destroyer entry of each precondition
            for i, des_entry in enumerate(self.actions_destroyers[up_action]):
                # skip for non-boolean precondition
                if(not self.actions_prec[up_action][i].is_boolean()):
                    continue

                # constraint that says the achiever is before b
                achiever_before = self.z3_achiever_variable(i,l) < b
            
                # list of fluent parameter, corresponding parameter index from up_action
                f_action_params = des_entry[1]
                f_action_const = des_entry[2]

                # for each possible destroyer for the precondition
                for destroyer, f_destroyer_params, f_types, f_const in des_entry[0]:
                    # skip destroyer if is initial action
                    if(destroyer == self.initial_action):
                        continue

                    if(destroyer in self.impossible_actions):
                        continue

                    # print(f"destroyer: {destroyer.name}") # debug

                    # constraint that says destroyer is executed at b 
                    destroyer_matches = self.z3_action_variable(b) == self.z3_actions_mapping[destroyer]
                    
                    # list of constraint that says the parameters match
                    params_match = []
                    # for each fluent parameter
                    for p in range(0,len(f_action_params)):
                        # skip if the parameter is constant in terms of both up_action and destroyer -> must match
                        if f_action_params[p] == None and f_destroyer_params[p] == None:
                            continue

                        # LHS expression of equality: fluent parameter in terms of up_action's parameter or constant
                        left = self.z3_action_parameters[f_action_params[p]](l) if f_action_params[p] \
                               is not None else self._expr_to_z3(f_action_const[p], l)
                        # RHS expression of equality: fluent parameter in terms of destroyer's parameter or constant
                        right = self.z3_action_parameters[f_destroyer_params[p]](b) if f_destroyer_params[p] \
                                is not None else self._expr_to_z3(f_const[p], b)
                        # LHS = RHS: match parameters of up_action and destroyer
                        params_match.append(left == right)
                    
                    formula.append(z3.Not(z3.And(
                        action_matches, destroyer_matches, *params_match, achiever_before
                    )))

        return formula

    def encode_objective(self):
        """
        Encode the objectives of the plan in order to optimize the plan quality.

        @return formula: list of tuple(Z3 formula, min/max label) asserting the objectives
        """
        
        formula = []

        # return empty list if optimization flag is off
        if not self.optimize: return formula

        l = self.z3_plan_ind_var

        # add maximizing expressions
        for max in self.maximize_objectives:
            formula.append((self._expr_to_z3(max, l, update=True), self.metric_max_label))

        # add minimzing expressions
        for min in self.minimize_objectives:
            formula.append((self._expr_to_z3(min, l, update=True), self.metric_min_label))

        # add action cost function with minimize label if applicable
        if self.z3_action_cost_variable is not None:
            formula.append(self.z3_action_cost_variable(l), self.metric_min_label)

        # print(formula) # debug

        return formula

    # copied from SequentialQFUF.py
    def extract_plan(self, model, horizon):
        """!
        Extracts plan from model of the formula.
        Plan returned is linearized. (length = horizon + 1)

        @param model: Z3 model of the planning formula.
        @param encoder: encoder object, contains maps variable/variable names.

        @returns: dictionary containing plan. Keys are steps, values are actions.
        """

        plan = SequentialPlan([])
        if not model: return plan
        ## linearize partial-order plan
        for step in range(1, horizon+2):
            # which action is in step "step?"
            # print(f"extract step {step}") # debug

            action_selected = model.evaluate(self.z3_action_variable(step))
            up_action = self.up_actions_mapping[action_selected]

            if(up_action == self.goal_action):
                continue

            # print(f"action: {action_selected}") # debug

            action_parameters = []
            for i in range(0, len(up_action.parameters)):
                z3_object = model.evaluate(self.z3_action_parameters[i](step))
                up_object = self.z3_objects_to_up[z3_object]
                action_parameters.append(up_object)

            # print(f"parameters: {action_parameters}") # debug

            action_inst = ActionInstance(up_action, action_parameters)
            plan.actions.append(action_inst)
            
        # maps compiled actions back to original actions
        # plan = plan.replace_action_instances(self.compile_result.map_back_action_instance)

        # print(plan) # debug
        # print(model)

        return SMTSequentialPlan(plan, self.task)

    # extended from SequentialQFUF.py
    def encode(self, t):
        """!
        Builds and returns the formulas for a single transition step (from t to t+1).
        @param t: the current timestep we want the encoding for
        @returns: A dict with the different parts of the formula encoded
        """

        if self.goal_action in self.impossible_actions:
            return None

        # for the first step of encoding, create the base encoding with the timestep variables to be replaced
        if t == 0:
            self.base_encode()

        # self.z3_timestep_last = z3.Int("t_goal", ctx=self.ctx) # the var that stores the last 
        
        act_t = t+1 # index of newly added slot for ordinary action
        goal_t = t+2 # index of goal action for the current plan length
        
        encoded_formula = dict()
        # tuple for substituting timestep variable with indices t, t+1 and t+2
        cur_index_sub = (self.z3_plan_ind_var, z3.IntVal(t, ctx=self.ctx))
        act_index_sub = (self.z3_plan_ind_var, z3.IntVal(act_t, ctx=self.ctx))
        goal_index_sub = (self.z3_plan_ind_var, z3.IntVal(goal_t, ctx=self.ctx))
        
        # initial state
        encoded_formula['initial_exec'] = self.formula['initial_exec']
        encoded_formula['initial_once'] = z3.substitute(self.formula['initial_once'], act_index_sub)

        # goal state
        encoded_formula['goal'] = z3.substitute(self.formula['goal'], goal_index_sub)
        if(t > 0):
            encoded_formula['goal_once'] = z3.substitute(self.formula['goal_once'], act_index_sub)
        else:
            encoded_formula['goal_once'] = None
        if(t == 1):
            encoded_formula['goal_once'] = z3.And(encoded_formula['goal_once'],
                                                  z3.substitute(self.formula['goal_once'], cur_index_sub))

        # actions
        encoded_formula['actions'] = z3.substitute(self.formula['actions'], act_index_sub) \
                                     if not self.formula['actions'] == None else None
        encoded_formula['typing']  = self.formula['typing']

        # achievers
        # encode for action at (t+2), supposed to be goal if successful planning
        # encoding for action at (t+1) should be done by previous step encoding (encode(t-1))
        # if planning failed, same formula can be reused for later iteration with larger t
        has_achievers = z3.substitute(self.formula['has_achiever'], goal_index_sub) \
                        if not self.formula['has_achiever'] == None else None
        is_achievers = z3.substitute(self.formula['is_achiever'], goal_index_sub) \
                       if not self.formula['is_achiever'] == None else None
        
        # replace achiever(i,l) = b as a large OR for all possible b < l, b > 0, for all preconditions i applicable
        # sub_achievers[i] = (achiever(i,l) = b, OR expression joining all possible achiever(i,l) values)
        # eg. i=0 & l=2, achiever(0,2) = b replace with (achiever(0,2) = 0 \/ achiever(0,2) = 1)
        if has_achievers is not None:
            sub_achievers = [ (self.z3_achiever_variable(i,goal_t) == self.z3_modifier_ind_var,
                            z3.Or([self.z3_achiever_variable(i,goal_t) == b for b in range(1, goal_t)])) # for all possible b
                            for i in range(0, self.max_prec_no) ] # for all preconditions
            encoded_formula['has_achiever'] = z3.substitute(has_achievers, sub_achievers)
        else:
            encoded_formula['has_achiever'] = None
        encoded_formula['is_achiever'] = z3.And([z3.substitute(is_achievers, 
                                                               (self.z3_modifier_ind_var, z3.IntVal(b, ctx=self.ctx)))
                                                for b in range(1, goal_t)]) \
                                          if not is_achievers == None else None
        # for initial state achievers, just replace the timestep of the supported action
        encoded_formula['is_init_achiever'] = z3.substitute(self.formula['is_init_achiever'], goal_index_sub) \
                                              if not self.formula['is_init_achiever'] == None else None
        encoded_formula['has_achiever_1'] = self.formula['has_achiever_1']
        encoded_formula['is_achiever_1'] = self.formula['is_achiever_1']

        # destroyers
        if self.formula['destroyers'] is not None:
            # achiever(0,3) < b
            destroyers = z3.substitute(self.formula['destroyers'], goal_index_sub)
            # achiever(0,3) < 2 ... achiever(0,3) < 1
            destroyers = z3.And([z3.substitute(destroyers, 
                                                (self.z3_modifier_ind_var, z3.IntVal(b, ctx=self.ctx)))
                                                for b in range(1, goal_t)])
            # list of tuples for substituting achiever(i,l) < b for all possible achiever positions between
            # initial state and the considered action exclusively for all preconditions
            sub_achbef = [(self.z3_achiever_variable(i, goal_t) < b, 
                           z3.Or([self.z3_achiever_variable(i, goal_t) == j for j in range(0,b)]))
                           for i in range(0, self.max_prec_no) for b in range(1, goal_t)]
            destroyers = z3.substitute(destroyers, sub_achbef)
            encoded_formula['destroyers'] = destroyers
        else:
            encoded_formula['destroyers'] = None
        
        # numeric precondition + effect
        encoded_formula['numeric'] = z3.substitute(self.formula['numeric'], goal_index_sub) \
                                       if not self.formula['numeric'] == None else None
        encoded_formula['numeric_1'] = self.formula['numeric_1']

        # objective (plan optimization)
        encoded_formula['objective'] = [(z3.substitute(o[0], goal_index_sub), o[1]) for o in self.formula['objective']] \
                                        if not self.formula['objective'] == None else None
        
        self.formula_length += 1
        return encoded_formula
    
    # extended from SequentialQFUF.py
    def base_encode(self):
        """!
        Builds the base encoding, where formula stored will then be used to generate formulae
        for future timesteps by substituting variables/constants within
        """

        # the var that stores the index of last step
        self.z3_plan_ind_var           = z3.Int('l', ctx=self.ctx)
        self.z3_modifier_ind_var       = z3.Int('b', ctx=self.ctx)

        # initial state
        self.formula['initial_exec']   = z3.And(self.encode_initial_state()) # Exec(0) = initial
        self.formula['initial_once']   = z3.And(self.encode_initial_once())  # Exec(t>0) != initial
        
        # goal state
        self.formula['goal']           = z3.And(self.encode_goal_state())    # Exec(t+2) = goal
        self.formula['goal_once']      = z3.And(self.encode_goal_once())     # Exec(t<t+2) != goal

        # actions
        acts = self.encode_actions()
        self.formula['actions']        = z3.And(acts) if len(acts) > 0 else None      # action parameter typing

        # typing
        self.formula['typing']         = z3.And(self.formula['typing']) if len(self.formula['typing']) > 0 else None

        # achievers
        ha = self.encode_has_achiever()
        self.formula['has_achiever']   = z3.And(ha) if len(ha) > 0 else None
        ia = self.encode_is_achiever()
        self.formula['is_achiever']    = z3.And(ia) if len(ia) > 0 else None
        iia = self.encode_is_init_achiever()
        self.formula['is_init_achiever']=z3.And(iia) if len(iia) > 0 else None

        # numeric preconditions & effects
        num = self.encode_numeric()
        self.formula['numeric']        = z3.And(num) if len(num) > 0 else None
        # encoding specifically for t+1 when t=0 so all slots are encoded
        self.formula['numeric_1']      = z3.substitute(self.formula['numeric'],
                                                       (self.z3_plan_ind_var, z3.IntVal(1, ctx=self.ctx))) \
                                         if not self.formula['numeric'] == None else None

        # achiever formula specifically for actions at t=1
        # achiever must be the initial state ie. Exec(0)
        # Exec(1) = a -> achiever(0,1) = 0 /\ achiever(1,1) = 0 /\ ...
        if not self.formula['has_achiever'] == None:
            ha_f = z3.substitute(self.formula['has_achiever'],
                                (self.z3_plan_ind_var, z3.IntVal(1, ctx=self.ctx)))
            ha_f = z3.substitute(ha_f,
                                 *[(self.z3_achiever_variable(i,1) == self.z3_modifier_ind_var, 
                                  z3.BoolVal(False, ctx=self.ctx)) for i in range(0, self.max_prec_no)])
            self.formula['has_achiever_1'] = ha_f
        else:
            self.formula['has_achiever_1'] = None
        
        # (Exec(1) = a /\ achiever(0,1) = 0 -> Exec(0) = achiever /\ params) /\
        # (Exec(1) = a /\ achiever(0,2) = 0 -> Exec(0) = achiever /\ params) /\ ...
        self.formula['is_achiever_1']  = z3.substitute(self.formula['is_init_achiever'],
                                                       (self.z3_plan_ind_var, z3.IntVal(1, ctx=self.ctx))) \
                                         if not self.formula['is_init_achiever'] == None else None
        
        # destroyers
        ds = self.encode_destroyers()
        self.formula['destroyers']     = z3.And(ds) if len(ds) > 0 else None

        # objective
        obj = self.encode_objective()
        self.formula['objective']      = obj if len(obj) > 0 else None

class EncoderSequentialAD(EncoderSequentialAchieverDestroyer):
    def __init__(self, task):
        super().__init__(task, False)

class EncoderOAD(EncoderSequentialAchieverDestroyer):
    def __init__(self, task):
        super().__init__(task, True)