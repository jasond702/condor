from sksundae import ida
import sundials4py as sun4py
from sundials4py import idas
from sundials4py.core import (
    SUNContext_Create,
    SUN_COMM_NULL,
    N_VNew_Serial,
    N_VGetArrayPointer,
    N_VClone,  # it is N_VCloneVectorArray in example
    SUN_SUCCESS,
    sunrealtype,
)
from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from casadi import sum2
from math import isclose


class SolverMixin:
    def store_result(self, store_t, store_y, store_yp):  # store_y=False):
        system = self.system
        results = system.result
        results.t.append(store_t)
        results.y.append(store_y)
        results.yp.append(store_yp)
        # breakpoint()
        # if system.dynamic_output and store_y:
        #     results.y.append(
        #         np.array(system.dynamic_output(results.p, store_t, store_x)).reshape(-1)
        #     )


class Sundials4PySolver(SolverMixin):
    def __init__(
        self,
        system,
        atol=1e-10,
        rtol=1e-6,
    ):
        self.system = system
        self.atol = atol
        self.rtol = rtol

    def simulate(self):
        system = self.system
        results = system.result

        last_x = system.initial_state
        last_xp = system.initial_dot

        last_s = system.sensitivity_initial_state.T
        last_sd = system.sensitivity_initial_dot.T

        SUNFALSE = 0
        SUNTRUE = 1

        time_generator = system.time_generator()
        last_t = next(time_generator)

        # Step 1: Create the SUNDIALS Context
        status, sunctx = SUNContext_Create(SUN_COMM_NULL)
        assert status == SUN_SUCCESS

        # Step 2: Set up the DAE problem
        y = N_VNew_Serial(system.NEQ, sunctx)
        yp = N_VNew_Serial(system.NEQ, sunctx)
        assert y is not None
        assert yp is not None

        system.set_init_cond(y, last_x)  # set up initial conditions
        system.set_init_cond(yp, last_xp)

        # Step 3: Create and Initialize IDAS
        IDAS = idas.IDACreate(sunctx)
        assert IDAS is not None

        # maybe temp, initialize IDA depending if logspace or linspace
        if "logspace" in system.analysis_options:
            status = idas.IDAInit(
                IDAS.get(), system.residualfunction, 10**system.start_time, y, yp
            )
            assert status == idas.IDA_SUCCESS
        else:
            status = idas.IDAInit(
                IDAS.get(), system.residualfunction, system.start_time, y, yp
            )
            assert status == idas.IDA_SUCCESS

        # Step 4: Set IDAS options (tolerances, linear solver, etc.)
        reltol = self.rtol
        abstol = N_VClone(y)  # self.atol
        system.set_init_cond(abstol, [1e-8, 1e-14, 1e-6])  # only set up for robertson
        status = idas.IDASVtolerances(IDAS.get(), reltol, abstol)
        assert status == idas.IDA_SUCCESS

        """
        Root stuff is only set up for Robertson
        """
        # set up root functions (when do events occur)
        status = idas.IDARootInit(IDAS.get(), 3, system.RobertsonEventsTEMP)
        assert status == idas.IDA_SUCCESS

        # this is from pysundae
        status = idas.IDASetRootDirection(IDAS.get(), [0, 0, -1])
        assert status == idas.IDA_SUCCESS

        status = idas.IDASetNoInactiveRootWarn(IDAS.get())
        assert status == idas.IDA_SUCCESS

        state_id = N_VClone(y)
        system.set_init_cond(state_id, [1, 1, 0])
        status = idas.IDASetId(IDAS.get(), state_id)
        assert status == idas.IDA_SUCCESS

        # set up linear and nonlinear solvers
        A = sun4py.core.SUNDenseMatrix(system.NEQ, system.NEQ, sunctx)
        assert A is not None

        LS = sun4py.core.SUNLinSol_Dense(y, A, sunctx)
        assert LS is not None

        status = idas.IDASetLinearSolver(IDAS.get(), LS, A)
        assert status == idas.IDA_SUCCESS

        NLS = sun4py.core.SUNNonlinSol_Newton(y, sunctx)
        assert NLS is not None

        status = idas.IDASetNonlinearSolver(IDAS.get(), NLS)
        assert status == idas.IDA_SUCCESS

        # Optional to set Jacobian

        # Set up Sensitivity Stuff, only set up for robertson
        # create sensitivity arrays for the states

        yS = [N_VClone(y) for _ in range(3)]
        ypS = [N_VClone(yp) for _ in range(3)]

        assert yS is not None
        assert ypS is not None

        for index, _elem in enumerate(yS):
            system.set_init_cond(yS[index], last_s[index, :])
            system.set_init_cond(ypS[index], last_sd[index, :])

        status = idas.IDASensInit(
            IDAS.get(),
            3,
            idas.IDA_STAGGERED,
            system.residualfunction_sensitivity,
            yS,
            ypS,
        )
        assert status == idas.IDA_SUCCESS

        status = idas.IDASensEEtolerances(IDAS.get())
        assert status == idas.IDA_SUCCESS

        err_con = 1  # err_con = SUNFALSE = 0 in Robertson example, SUNTRUE = 1
        status = idas.IDASetSensErrCon(
            IDAS.get(), err_con
        )  # 2nd parameter is error control (err_con)
        assert status == idas.IDA_SUCCESS

        # NLSSens = sun4py.core.SUNNonlinSol_NewtonSens(3, y, sunctx)
        # assert NLSSens is not None

        # status = idas.IDASetNonlinearSolverSensStg(IDAS.get(), NLSSens)
        # assert status == idas.IDA_SUCCESS

        # Set up Quadratures (for Adjoint in future, but can help now)
        # yQ = N_VNew_Serial(2, sunctx)
        # ICs of it are 0

        # status = idas.IDAQuadInit(IDAS.get(), system.rhsQ, yQ)

        status = idas.IDAGetSensConsistentIC(IDAS.get(), yS, ypS)
        assert status == idas.IDA_SUCCESS

        # Step 5: Advance the DAE in time
        if "logspace" in system.analysis_options:
            tret = 10**system.start_time
        else:
            tret = system.start_time

        # while True:
        #     next_t = next(time_generator)
        #     if np.isinf(next_t):
        #         break
        #     if next_t < 0:
        #         # breakpoint()
        #         pass
        yarr = N_VGetArrayPointer(y)
        yparr = N_VGetArrayPointer(yp)

        # copy initial conditions into array
        self.store_result(
            np.copy(tret),
            np.copy(yarr),
            np.copy(yparr),
        )

        rootsfound = np.zeros(3, dtype=sunrealtype)

        while tret < system.final_time:
            if "logspace" in system.analysis_options:
                status, tret = idas.IDASolve(
                    IDAS.get(),
                    10 ** (np.log10(tret) + system.time_step),
                    y,
                    yp,
                    idas.IDA_NORMAL,
                )
            else:
                status, tret = idas.IDASolve(
                    IDAS.get(),
                    tret + system.time_step,
                    y,
                    yp,
                    idas.IDA_NORMAL,
                )

            if status == idas.IDA_SUCCESS:
                assert status == idas.IDA_SUCCESS

            elif status == idas.IDA_ROOT_RETURN:
                status = idas.IDAGetRootInfo(IDAS.get(), rootsfound)
                assert status == idas.IDA_SUCCESS
                results.e.append(Root(iout, rootsfound))

            status = idas.IDAGetSens(IDAS.get(), yS)
            assert status[0] == idas.IDA_SUCCESS

            # if isclose(tret, 4, rel_tol=1e-2):
            #     print(f"when t = {tret}")
            #     print(N_VGetArrayPointer(yS[0]))
            #     print(N_VGetArrayPointer(yS[1]))
            #     print(N_VGetArrayPointer(yS[2]))
            #     print("=======================")
            #     breakpoint()

            self.store_result(
                np.copy(tret),
                np.copy(yarr),
                np.copy(yparr),
            )
            iout += 1
        # Step 6: Get IDAS Statistics (optional)


class Root(NamedTuple):
    index: int
    rootsfound: list[int]


class NextTimeFromSlice:
    def __init__(self, at_time_func):
        self.at_time_func = at_time_func

    def set_p(self, p):
        start, stop, step = np.array(self.at_time_func(p)).squeeze()
        self.start = start
        self.step = step
        self.stop = stop
        self.direction = np.sign(step)

    def before_start(self, t):
        if self.direction < 0:
            return t > self.start
        return t < self.start

    def after_stop(self, t):
        # if t is exactly stop time, this has already occured
        if self.direction < 0:
            return t <= self.stop
        return t >= self.stop

    def __call__(self, t):
        if self.before_start(t):
            return self.start
        if self.after_stop(t):
            return self.direction * np.inf
        # TODO handle negative step -- may need to adjust a few of these
        return (1 + (t - self.start) // self.step) * self.step + self.start


class TimeGeneratorFromSlices:
    def __init__(self, time_slices, direction=1):
        self.time_slices = time_slices
        self.direction = direction
        # breakpoint()

    def __call__(self, p):
        # TODO handle negative step?
        for time_slice in self.time_slices:
            time_slice.set_p(p)

        t = -self.direction * np.inf
        get_time = min if self.direction > 0 else max
        while True:
            next_times = [time_slice(t) for time_slice in self.time_slices]
            t = get_time(next_times)
            yield t
            if np.isinf(t):
                breakpoint()
        # breakpoint()


class DAESystemAnalysis:
    def __init__(
        self,
        initial_conditions,
        p,
        final_time,
        start_time,
        state_count,
        count_diff,
        residual_func,
        sens_func,
        sens_state_initial_condition_func,
        sens_dot_initial_condition_func,
        time_generator,
        **analysis_options,
    ):
        self.state_count = state_count
        self.NEQ = state_count
        self.analysis_options = analysis_options
        self.residual_func = residual_func

        self.sens_func = sens_func
        self.sens_state_initial_condition_func = sens_state_initial_condition_func
        self.sens_dot_initial_condition_func = sens_dot_initial_condition_func

        self.initial_conditions = initial_conditions
        self.p = np.array(p, dtype=sunrealtype)

        self.start_time = start_time
        self.final_time = final_time
        self.result = None
        self._time_generator = time_generator
        # self.time_step = (final_time - start_time) / (
        #     analysis_options.pop("num_steps") - 1
        # )
        # TODO: i want to fix handling log vs lin space

        if "logspace" in analysis_options:
            if start_time == 0:
                self.start_time = -6
            self.time_step = (np.log10(final_time) - self.start_time) / (
                analysis_options.pop("num_steps") - 1
            )
        else:
            self.time_step = (final_time - self.start_time) / (
                analysis_options.pop("num_steps") - 1
            )

        self.initial_state = initial_conditions.variable.flatten()[:state_count]
        self.initial_dot = initial_conditions.variable.flatten()[state_count:]

        self.sens_initial_state = sens_state_initial_condition_func(self.p)
        self.sens_initial_dot = sens_dot_initial_condition_func(self.initial_state)

        self.start_solver()

    def start_solver(self):
        self.system_solver = Sundials4PySolver(system=self)

    def set_init_cond(self, yvec, y0):
        y = N_VGetArrayPointer(yvec)
        y[:] = y0

        return 0

    def residualfunction(self, t, yvec, ypvec, resvec, _):
        y = N_VGetArrayPointer(yvec)
        yp = N_VGetArrayPointer(ypvec)
        res = N_VGetArrayPointer(resvec)
        res[:, None] = self.residual_func(
            y,
            yp,
            self.p,
        )
        return 0

    def RobertsonEventsTEMP(self, t, yvec, ypvec, evvec, _):
        y = N_VGetArrayPointer(yvec)
        yp = N_VGetArrayPointer(ypvec)
        evvec[0] = y[0] - 0.1
        evvec[1] = y[1] * 1e4 - 0.2
        evvec[2] = y[2] - 0.7
        return 0

    def residualfunction_sensitivity(
        self, NS, t, yvec, ypvec, resvec, ySvec, ypSvec, resSvec, _, tmp1, tmp2, tmp3
    ):
        y = N_VGetArrayPointer(yvec)
        yp = N_VGetArrayPointer(ypvec)
        res = N_VGetArrayPointer(resvec)

        yS = np.array([list(N_VGetArrayPointer(ySvec[i])) for i in range(3)]).T
        ypS = np.array([list(N_VGetArrayPointer(ypSvec[i])) for i in range(3)]).T

        resS_value = self.sens_func(yS, ypS, y, yp, self.p)

        for index in range(NS):
            resS = N_VGetArrayPointer(resSvec[index])
            resS[:] = resS_value[:, index].full().flatten()

        return 0

    def time_generator(self):
        for t in self._time_generator(self.result.p):
            yield np.array(t).reshape(-1)[0]

    def __call__(self):
        self.result = DAEResult(p=self.p, system=self)  # initialize the results?
        self.system_solver.simulate()
        result = self.result
        result.t = np.array(result.t)
        result.y = np.array(result.y)
        result.yp = np.array(result.yp)
        return result


@dataclass
class ResultMixin:
    p: list[float]


@dataclass
class ResultBase:
    system: DAESystemAnalysis
    t: list[float] = field(default_factory=list)
    y: list[list] = field(default_factory=list)
    yp: list[list] = field(default_factory=list)
    e: list[Root] = field(default_factory=list)

    def __getitem__(self, key):
        return self.__class__(
            *tuple(getattr(self, field.name) for field in fields(self)[:-3]),
            t=self.t[key],
            y=self.y[key],
            yp=self.yp[key],
            e=self.e[key],
        )

    def save(self, filename):
        e_idxs = [e.index for e in self.e]
        e_roots = [e.rootsfound for e in self.e]
        np.savez_compressed(
            filename,
            e_idxs=e_idxs,
            e_roots=e_roots,
            t=self.t,
            y=self.y,
            yp=self.yp,
            p=self.p,
        )

    @classmethod
    def load(cls, filename):
        data = dict(np.load(filename))
        data["e"] = [
            Root(index=int(ei), rootsfound=er)
            for ei, er in zip(data.pop("e_idxs"), data.pop("e_roots"))
        ]
        return cls(system=None, **data)


@dataclass
class DAEResult(ResultBase, ResultMixin):
    pass
