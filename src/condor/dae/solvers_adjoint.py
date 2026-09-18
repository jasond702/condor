from sksundae import ida
import sundials4py as sun4py
from sundials4py import idas
from sundials4py.core import (
    SUNContext_Create,
    SUN_COMM_NULL,
    N_VNew_Serial,
    N_VGetArrayPointer,
    N_VGetNumpyArray,
    N_VClone,  # it is N_VCloneVectorArray in example
    SUN_SUCCESS,
    sunrealtype,
)
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from scipy.interpolate import make_interp_spline
import condor.solvers.sweeping_gradient_method as sgm
import numpy as np
from casadi import sum2
from math import isclose
from time import time


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
        atol=1e-12,
        rtol=1e-10,
    ):
        self.system = system
        self.atol = atol
        self.rtol = rtol

        # Step 1: Create the SUNDIALS Context
        status, sunctx = SUNContext_Create(SUN_COMM_NULL)
        assert status == SUN_SUCCESS

        # Initialize states and dots
        self.y = N_VNew_Serial(system.NEQ, sunctx)
        self.yp = N_VNew_Serial(system.NEQ, sunctx)
        assert self.y is not None
        assert self.yp is not None

        self.sunctx = sunctx

        # move solver (LS and NLS) here
        # set up linear solver
        self.A = sun4py.core.SUNDenseMatrix(system.NEQ, system.NEQ, self.sunctx)
        assert self.A is not None

        self.LS = sun4py.core.SUNLinSol_Dense(self.y, self.A, self.sunctx)
        assert self.LS is not None

        # set up nonlinear solver
        self.NLS = sun4py.core.SUNNonlinSol_Newton(self.y, self.sunctx)
        assert self.NLS is not None

    def assign_nvec_value(self, yvec, y0):
        y = N_VGetNumpyArray(yvec)
        y[:] = y0

        return 0

    def simulate(self, one_step, max_step, test_out_vectors=None):
        system = self.system
        results = system.result

        # last_s = system.sens_initial_state.T
        # last_sd = system.sens_initial_dot.T
        last_x, last_xp = system.dae_initial_conditions(system.p_model)

        SUNFALSE = 0
        SUNTRUE = 1

        time_generator = system.time_generator()
        last_t = next(time_generator)

        gs = system.events(last_t, last_x, last_xp)
        self.store_result(last_t, last_x, last_xp)

        rootsfound = (gs == 0.0).astype(int)

        if np.any(rootsfound):
            last_x, last_xp = system.update(
                rootsfound, last_t, np.copy(last_x), np.copy(last_xp)
            )
        results.e.append(Root(1, rootsfound))
        self.store_result(last_t, last_x, last_xp)

        terminate = np.any(rootsfound[system.terminating] != 0)
        if terminate:
            return

        y = self.y
        yp = self.yp
        A = self.A
        LS = self.LS
        NLS = self.NLS

        self.assign_nvec_value(y, last_x)
        self.assign_nvec_value(yp, last_xp)

        breakpoint()
        while True:
            next_t = next(time_generator)
            if np.isinf(next_t):
                break
            if next_t < 0:
                pass

            # create and initialize IDAS
            IDAS = idas.IDACreate(self.sunctx)
            assert IDAS is not None

            status = idas.IDAInit(
                IDAS.get(), system.residualfunction, system.start_time, y, yp
            )

            # Set tolerances
            status = idas.IDASStolerances(IDAS.get(), self.rtol, self.atol)
            assert status == idas.IDA_SUCCESS

            # Set max steps
            status = idas.IDASetMaxNumSteps(IDAS.get(), 100000)
            assert status == idas.IDA_SUCCESS

            # attach linear and nonlinear matrices to solvers
            status = idas.IDASetLinearSolver(IDAS.get(), LS, A)
            assert status == idas.IDA_SUCCESS

            status = idas.IDASetNonlinearSolver(IDAS.get(), NLS)
            assert status == idas.IDA_SUCCESS

            # TODO: copy over quad stuff

            # set up quad and sens init here

            # yS = [N_VClone(y) for _ in range(3)]
            # ypS = [N_VClone(yp) for _ in range(3)]

            # assert yS is not None
            # assert ypS is not None

            # for index, _elem in enumerate(yS):
            #     system.set_init_cond(yS[index], last_s[index, :])
            #     system.set_init_cond(ypS[index], last_sd[index, :])

            # status = idas.IDASensInit(
            #     IDAS.get(),
            #     3,
            #     idas.IDA_STAGGERED,
            #     system.residualfunction_sensitivity,
            #     yS,
            #     ypS,
            # )
            # assert status == idas.IDA_SUCCESS

            # status = idas.IDASensEEtolerances(IDAS.get())
            # assert status == idas.IDA_SUCCESS

            # err_con = 1  # err_con = SUNFALSE = 0 in Robertson example, SUNTRUE = 1
            # status = idas.IDASetSensErrCon(
            #     IDAS.get(), err_con
            # )  # 2nd parameter is error control (err_con)
            # assert status == idas.IDA_SUCCESS

            # status = idas.IDAGetSensConsistentIC(IDAS.get(), yS, ypS)
            # assert status == idas.IDA_SUCCESS

            status = idas.IDASetStopTime(IDAS.get(), next_t)
            assert status == idas.IDA_SUCCESS

            # dt = (next_t - last_t) / system.analysis_options.get("num_steps")

            status = idas.IDASetMaxStep(IDAS.get(), max_step)  # smooth was set to 0.01
            assert status == idas.IDA_SUCCESS

            status = idas.IDARootInit(
                IDAS.get(), system._events.size_out(0)[0], system.eventsfunction
            )
            assert status == idas.IDA_SUCCESS

            # status = idas.IDASetRootDirection(
            #     IDAS.get(), [0] * system._events.size_out(0)[0]
            # )
            # assert status == idas.IDA_SUCCESS

            yarr = N_VGetNumpyArray(y)
            yparr = N_VGetNumpyArray(yp)

            t_start = time()
            while last_t < next_t:
                if one_step and 1:
                    while True:
                        status, tret = idas.IDASolve(
                            IDAS.get(), next_t, y, yp, idas.IDA_ONE_STEP
                        )
                        self.store_result(np.copy(tret), np.copy(yarr), np.copy(yparr))
                        if status in (idas.IDA_TSTOP_RETURN, idas.IDA_ROOT_RETURN):
                            break

                else:
                    self.adj_system = None  # IT IS NOT ADDED YET
                    if self.adj_system is not None:
                        pass
                    else:
                        status, tret = idas.IDASolve(
                            IDAS.get(), next_t, y, yp, idas.IDA_NORMAL
                        )

                    if status not in (
                        idas.IDA_TSTOP_RETURN,
                        idas.IDA_SUCCESS,
                        idas.IDA_ROOT_RETURN,
                    ):
                        breakpoint()

                    self.store_result(np.copy(tret), np.copy(yarr), np.copy(yparr))
                # INSERT quad stuff

                if status == idas.IDA_ROOT_RETURN:
                    rootsfound = np.zeros(system._events.size_out(0)[0], dtype=np.int32)
                    idas.IDAGetRootInfo(IDAS.get(), rootsfound)

                if status == idas.IDA_TSTOP_RETURN:
                    print("hi")
                    gs = system.events(results.t[-1], results.y[-1], results.yp[-1])
                    min_e = np.abs(gs).min()
                    rootsfound = (gs == min_e).astype(int)
                    assert next_t == tret

                if status in (
                    idas.IDA_TSTOP_RETURN,
                    idas.IDA_ROOT_RETURN,
                    idas.IDA_SUCCESS,
                ):
                    idx = len(results.t)
                    results.e.append(Root(idx, rootsfound))
                    if (
                        rootsfound[system.terminating] != 0
                    ):  # update regardless, remove this if statement
                        next_x, next_xp = system.pass_through_update(
                            rootsfound, results.t[-1], results.y[-1], results.yp[-1]
                        )
                    else:
                        next_x, next_xp = system.update(
                            rootsfound, results.t[-1], results.y[-1], results.yp[-1]
                        )
                    try:
                        terminate = np.any(rootsfound[system.terminating] != 0)

                    except Exception as e:
                        print("Hit exemption:")
                        print(e)
                        print("You may try to continue through or exit")
                        breakpoint()
                    self.store_result(np.copy(tret), np.copy(next_x), np.copy(next_xp))

                yarr[:] = next_x
                yparr[:] = next_xp
                if terminate:
                    break
                status = idas.IDAReInit(IDAS.get(), results.t[-1], y, yp)
                assert status == idas.IDA_SUCCESS

                last_t = tret
                # breakpoint()
            print("Solve took:", time() - t_start)
        # adjoint solve


class Root(NamedTuple):
    index: int
    rootsfound: list[int]


# import stuff from sgm


class DAESystem:
    def __init__(
        self,
        dae_initial_conditions_solver,
        p,
        parameter_names,
        final_time,
        start_time,
        state_count,
        count_diff,
        residual_func,
        events,
        event_solvers,
        terminating,
        # sens_func,
        # sens_state_initial_condition_func,
        # sens_dot_initial_condition_func,
        time_generator,
        one_step=False,  # edit <- same for max_step, atol, rtol
        **analysis_options,
    ):
        self.state_count = state_count
        self.NEQ = state_count
        self.analysis_options = analysis_options
        self.residual_func = residual_func

        # self.sens_func = sens_func
        # self.sens_state_initial_condition_func = sens_state_initial_condition_func
        # self.sens_dot_initial_condition_func = sens_dot_initial_condition_func

        self.dae_initial_condition_solver = dae_initial_conditions_solver
        self.p_model = p
        self.p = p.flatten().astype(sunrealtype)
        self._parameter_names = parameter_names

        self.start_time = start_time
        self.final_time = final_time
        self.result = None
        self._time_generator = time_generator

        self._events = events
        self._event_solvers = event_solvers
        self.terminating = terminating

        if self.analysis_options.get("one_step"):
            self.one_step = self.analysis_options.get("one_step")
        else:
            self.one_step = 0
        if self.analysis_options.get("max_step"):
            self.max_step = self.analysis_options.get("max_step")
        else:
            self.max_step = 0
        # breakpoint()
        # self.time_step = (final_time - start_time) / (
        #     analysis_options.pop("num_steps") - 1
        # )

        # self.sens_initial_state = sens_state_initial_condition_func(self.p)
        # self.sens_initial_dot = sens_dot_initial_condition_func(self.initial_state)

        self.start_solver()

    def start_solver(self):
        self.system_solver = Sundials4PySolver(system=self)

    def residualfunction(self, t, yvec, ypvec, resvec, _):
        y = N_VGetNumpyArray(yvec)
        yp = N_VGetNumpyArray(ypvec)
        res = N_VGetNumpyArray(resvec)
        res[:, None] = self.residual_func(y, yp, self.p, t)
        return 0

    def eventsfunction(self, t, yvec, ypvec, evvec, _):
        y = N_VGetNumpyArray(yvec)
        yp = N_VGetNumpyArray(ypvec)
        evvec[:, None] = self._events(y, yp, self.p, t)
        return 0

    # def residualfunction_sensitivity(
    #     self, NS, t, yvec, ypvec, resvec, ySvec, ypSvec, resSvec, _, tmp1, tmp2, tmp3
    # ):
    #     y = N_VGetArrayPointer(yvec)
    #     yp = N_VGetArrayPointer(ypvec)
    #     res = N_VGetArrayPointer(resvec)

    #     yS = np.array([list(N_VGetArrayPointer(ySvec[i])) for i in range(3)]).T
    #     ypS = np.array([list(N_VGetArrayPointer(ypSvec[i])) for i in range(3)]).T

    #     resS_value = self.sens_func(yS, ypS, y, yp, self.p)

    #     for index in range(NS):
    #         resS = N_VGetArrayPointer(resSvec[index])
    #         resS[:] = resS_value[:, index].full().flatten()

    #     return 0

    def time_generator(self):
        for t in self._time_generator(self.result.p):
            yield np.array(t).reshape(-1)[0]

    def pass_through_update(self, rootsfound, t, y, yp):
        return y, yp

    def update(self, rootsfound, t, y, yp, *args):
        next_x = y
        next_xp = yp

        values = np.concatenate((self.p, next_x, next_xp))
        parameter_dict = dict(zip(self._parameter_names, values))
        solver_idx = np.where(rootsfound != 0)[0].item()

        if self._event_solvers[
            solver_idx
        ].residual._elements:  # don't look at the _elements
            update_values = self._event_solvers[np.where(rootsfound != 0)[0].item()](
                **parameter_dict
            ).variable
            return update_values.flatten()[: self.state_count], update_values.flatten()[
                self.state_count :
            ]
        else:
            next_x, next_xp = self.pass_through_update(rootsfound, t, y, yp)
            return next_x, next_xp

    def events(self, t, y, yp):
        return np.array(self._events(y, yp, self.result.p, t)).reshape(-1)

    def dae_initial_conditions(self, p):
        initial_condition_values = self.dae_initial_condition_solver(**p)

        return initial_condition_values.variable.flatten()[
            : self.state_count
        ], initial_condition_values.variable.flatten()[self.state_count :]

    def __call__(self, p):
        self.result = DAEResult(p=self.p, system=self)  # initialize the results?
        self.system_solver.simulate(one_step=self.one_step, max_step=self.max_step)
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
    system: DAESystem
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


class DAEResultSegmentInterpolant(NamedTuple):
    interpolant: callable
    idx0: int
    idx1: int
    t0: float
    t1: float
    y0: list[float]
    y1: list[float]
    yp0: list[float]
    yp1: list[float]

    def __call__(self, t):
        return self.interpolant(t)


@dataclass
class ResultInterpolant:
    result: DAEResult
    function: callable = lambda p, t, y, yp: (y, yp)  # add xdot: xdot?
    # don't pass interpolants to init?
    # should state_Result be saved or just be an initvar? I back-references are OK so
    # we can keep it...
    interpolants: Optional[list[callable]] = None  # field(init=False)
    time_bounds: Optional[list[float]] = None  # field(init=False)
    time_comparison: callable = field(init=False)  # method
    interval_select: int = field(init=False)
    event_idxs: Optional[list] = None
    max_deg: int = 3

    def __post_init__(self):
        # make interpolants
        result = self.result
        function = self.function
        if self.event_idxs is None:
            # currently expect each root.index to be to the right of each event so is
            # the start of each segment, so zero belongs with this set by adding the
            # element len(result.t), the slice of pairs slice(idx[i], idx[i+1])
            # captures the whole segment including the last one.

            self.event_idxs = np.array(
                [root.index for root in result.e]
            )  # + [len(result.t)])
            if np.any(np.diff(self.event_idxs) < 1):
                breakpoint()

        event_idxs = self.event_idxs
        if self.time_bounds is None:
            self.time_bounds = np.empty(self.event_idxs.shape)
            self.time_bounds[:-1] = np.array(result.t)[event_idxs[:-1]]
            self.time_bounds[-1] = result.t[-1]
        if result.t[-1] < result.t[0]:
            self.time_comparison = self.time_bounds.__le__
            self.interval_select = -1
            self.time_sort = slice(None, None, -1)
        else:
            self.time_comparison = self.time_bounds.__ge__
            self.interval_select = 0
            self.time_sort = slice(None)

        if self.interpolants is None:
            # TODO figure out how to combine (and possibly reverse direction) state and
            # parameter jacobian of state equation to reduce number of calls (and
            # distance at each call), since this is potentially most expensive call -- I
            # guess only if the ODE depends on inner-loop solver? then split
            # coefficients
            # --> then also combine e.g., adjoint forcing function and jacobian
            # interpolant

            # expect integrand-term related functions to be cheap functions of state
            # point-wise along trajectory
            try:
                all_coeff_data = [
                    [
                        np.array(function(y, yp, result.p, t)).squeeze()
                        for t, y, yp in zip(
                            result.t[idx0:idx1],
                            result.y[idx0:idx1],
                            result.yp[idx0:idx1],
                        )
                    ]
                    for idx0, idx1 in zip(event_idxs[:-1], event_idxs[1:])
                ]
            except Exception as my_e:
                print(my_e)
                breakpoint()
                pass

            self.interpolants = []
            for idx0, idx1, coeff_data in zip(
                event_idxs[:-1],
                event_idxs[1:],
                all_coeff_data,
            ):
                if np.all(np.diff(coeff_data, axis=0) == 0.0):
                    try:
                        interp = DAEResultSegmentInterpolant(
                            make_interp_spline(
                                result.t[idx0],
                                [coeff_data[0]],
                                k=0,
                            ),
                            idx0,
                            idx1,
                            result.t[idx0],
                            result.t[idx1],
                            result.y[idx0],
                            result.y[idx1],
                            result.yp[idx0],
                            result.yp[idx1],
                        )
                    except Exception as my_e:
                        print(my_e)
                        breakpoint()

                else:
                    ts = result.t[idx0:idx1][self.time_sort]
                    coefs = coeff_data[self.time_sort]

                    for count, orig_idx in enumerate(np.where(np.diff(ts) <= 0)[0]):
                        idx = count + orig_idx
                        ts = ts[:idx] + ts[idx + 1 :]
                        coefs = coefs[:idx] + coefs[idx + 1 :]
                        print(
                            f"stripping non-decreasing {ts[idx]} -- creating result "
                            "interpolant"
                        )

                    try:
                        interp = DAEResultSegmentInterpolant(
                            make_interp_spline(
                                ts,
                                coefs,
                                k=min(
                                    self.max_deg, idx1 - idx0 - 1
                                ),  # not needed with adaptive step size!
                                # bc_type=["natural", "natural"],
                            ),
                            idx0,
                            idx1,
                            result.t[idx0],
                            result.t[idx1],
                            result.y[idx0],
                            result.y[idx1],
                            result.yp[idx0],
                            result.yp[idx1],
                        )
                    except Exception as my_e:
                        print(my_e)
                        breakpoint()
                        pass
                self.interpolants.append(interp)
                # if result.t[idx1] != result.t[idx0]

    def __call__(self, t):
        interval_idx = np.where(self.time_comparison(t))[0][self.interval_select] - 1
        return self.interpolants[interval_idx](t)

    def __iter__(self):
        yield from self.interpolants


# AdjointResultMixin

# AdjointResult


@dataclass
class DAETrajectoryAnalysis:
    state_system: DAESystem

    # for constructing the output of the trajectory analysis
    integrand_terms: callable = None
    terminal_terms: callable = None

    cache_size: int = 1

    def __post_init__(self):
        self.cached_p = None
        # self.cached_output = None  # added this

    def __call__(self, p):
        # breakpoint()
        if self.cached_p is not None and np.all(self.cached_p == p):
            # breakpoint()
            return self.cached_output
        self.cached_p = p
        result = self.res = self.state_system(p)

        # evaluate the trajectory analysis of this result
        # should this return a dataclass? Or just the vector of results?
        integral = 0.0
        integrand_interpolant = ResultInterpolant(
            result=result, function=self.integrand_terms
        )
        for segment in integrand_interpolant:
            integrand_antideriv = segment.interpolant.antiderivative()
            integral += integrand_antideriv(segment.t1) - integrand_antideriv(
                segment.t0
            )
        self.cached_output = (
            self.terminal_terms(result.y[-1], result.yp[-1], result.p, result.t[-1])
            + integral
        )
        breakpoint()
        return self.cached_output


# SweepingGradientMethod

# DAETrajectoryAnalysisSGM?
