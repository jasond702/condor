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
        # self.sen_system
        # self.adjoint_system
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

        # self.Q <- for quadrature
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

        # if adjoint_system is not None:
        # self.uB
        # self.qB
        # self.lsb

    def assign_nvec_value(self, yvec, y0):
        y = N_VGetNumpyArray(yvec)
        y[:] = y0

        return 0

    def adjoint_quad(self):
        pass

    def adjoint_dots(self):
        pass

    def simulate(self, one_step, max_step, test_out_vectors=None):
        system = self.system
        results = system.result

        # last_s = system.sens_initial_state.T
        # last_sd = system.sens_initial_dot.T
        # breakpoint()
        last_x, last_xp = system.dae_initial_conditions(system.p)

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
        yarr = N_VGetNumpyArray(y)
        yparr = N_VGetNumpyArray(yp)

        # Q = self.Q
        # Qarr = N_VGetNumpyArray(Q)
        # Qarr[:] = 0.0

        # if self.adjoint_system is not None and test_out_vectors is not None:
        breakpoint()
        grab_next_t = True
        while True:
            # move if grab_next_t
            if grab_next_t:
                next_t = next(time_generator)

            if np.isinf(next_t):
                break
            if next_t < 0:
                pass

            # adaptive min steps?

            # create and initialize IDAS
            IDAS = idas.IDACreate(self.sunctx)
            assert IDAS is not None
            results.IDAS_mems.append(IDAS)
            # breakpoint()
            status = idas.IDAInit(IDAS.get(), system.residualfunction, last_t, y, yp)

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

            # if system.dim_output and self.sen_system is None and self.adjoint_system is None
            # status = idas.IDAQuadInit(IDAS.get(), integrand terms, Q)
            # assert status == idas.IDA_SUCCESS

            # if self.sen_system is not None and test_in is not None:
            # status = idas.IDASensInit()
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

            # if self.adjoint_system is not None: # and test_out_vectors is not None:
            # status = idas.IDAAdjInit(IDAS.get(), steps, HERMITE?)
            # assert status == idas.IDA_SUCCESS

            status = idas.IDARootInit(
                IDAS.get(), system._events.size_out(0)[0], system.eventsfunction
            )
            assert status == idas.IDA_SUCCESS

            # status = idas.IDASetRootDirection(
            #     IDAS.get(), [0] * system._events.size_out(0)[0]
            # )
            # assert status == idas.IDA_SUCCESS

            # t_start = time()
            # breakpoint()
            if one_step and 1:
                while True:
                    status, tret = idas.IDASolve(
                        IDAS.get(), next_t, y, yp, idas.IDA_ONE_STEP
                    )
                    self.store_result(np.copy(tret), np.copy(yarr), np.copy(yparr))
                    if status in (idas.IDA_TSTOP_RETURN, idas.IDA_ROOT_RETURN):
                        break
                    # self.store_result(np.copy(tret), np.copy(yarr), np.copy(yparr))

            else:
                self.adj_system = None  # IT IS NOT ADDED YET
                if self.adj_system is not None:
                    status, tret, ncheck = idas.IDASolveF(
                        IDAS.get(), next_t, y, idas.IDA_NORMAL
                    )
                else:
                    status, tret = idas.IDASolve(
                        IDAS.get(), next_t, y, yp, idas.IDA_NORMAL
                    )
                self.store_result(np.copy(tret), np.copy(yarr), np.copy(yparr))

            """
            if (self.system.dim_output and self.sen_system is None
            and self.adjoint_system is None):
                quad_status, tret = idas.IDAGetQuad(IDAS.get(), Q)
                assert quad_status == idas.IDA_SUCCESS
                results.o = Qarr[:] + system.terminal_terms(tret, yarr[:], yparr[:])
            """

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
                # if (
                #     rootsfound[system.terminating] != 0
                # ):  # update regardless, remove this if statement
                #     next_x, next_xp = system.pass_through_update(
                #         rootsfound, results.t[-1], results.y[-1], results.yp[-1]
                #     )
                # else:
                next_x, next_xp = system.update(
                    rootsfound, results.t[-1], results.y[-1], results.yp[-1]
                )

                # terminate = np.any(rootsfound[system.terminating] != 0)
                terminate = any(rootsfound[i] != 0 for i in system.terminating)
                grab_next_t = any(rootsfound[i] != 0 for i in system.time_triggering)

                self.store_result(np.copy(tret), np.copy(next_x), np.copy(next_xp))

            yarr[:] = next_x
            yparr[:] = next_xp
            last_t = tret

            if terminate:
                breakpoint()
                break

            # status = idas.IDAReInit(IDAS.get(), results.t[-1], y, yp)
            # assert status == idas.IDA_SUCCESS

            # breakpoint()
            # if terminate:
            #     breakpoint()
            #     break
            # print("Solve took:", time() - t_start)

        # if self.adjoint_system is None: #or test_out_vectors is None:
        # return
        # uB = self.uB
        # qB = self.qB
        # uBarr = N_VGetNumpyArray(uB)
        # last_lambda = self.adjoint_system.initial_condition(tret, yarr[:])
        # uBarr[:] = last_lambda

        # qBarr = N_VGetNumpyArray(qB)
        # qBarr[:] = self.adjoint_system.terminal_terms(tret, yarr[:])

        # a bunch of adjoint stuff


class Root(NamedTuple):
    index: int
    rootsfound: list[int]


# import stuff from sgm


class DAESystem:
    def __init__(
        self,
        dae_initial_conditions_solver,
        p,
        final_time,
        start_time,
        state_count,
        count_diff,
        residual_func,
        events,
        event_solvers,
        terminating,
        time_triggering,
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

        self.start_time = start_time
        self.final_time = final_time
        self.result = None
        self._time_generator = time_generator

        self._events = events
        self._event_solvers = event_solvers
        self.terminating = terminating
        self.time_triggering = time_triggering
        breakpoint()
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
        self.one_step = 1.0
        self.start_solver()

    def start_solver(self):
        self.system_solver = Sundials4PySolver(system=self)

    def residualfunction(self, t, yvec, ypvec, resvec, _):
        y = N_VGetNumpyArray(yvec)
        yp = N_VGetNumpyArray(ypvec)
        res = N_VGetNumpyArray(resvec)
        res[:, None] = self.residual_func(t, y, yp, self.p)
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
        # breakpoint()
        # solver_idx = np.where(rootsfound != 0)[0].item()
        for idx, root in enumerate(rootsfound):
            # breakpoint()
            if root != 0:
                update_values = self._event_solvers[idx](t, self.p, next_x, next_xp)
                next_x = update_values[0].reshape(-1)
                next_xp = update_values[1].reshape(-1)
        # update_values = self._event_solvers[solver_idx](t, self.p, y, yp)

        return next_x, next_xp
        # update_values[0].reshape(-1), update_values[1].reshape(-1)
        # if self._event_solvers[
        #     solver_idx
        # ].residual._elements:  # don't look at the _elements
        #     update_values = self._event_solvers[np.where(rootsfound != 0)[0].item()](
        #         *values  # **parameter_dict
        #     ).variable
        #     return update_values.flatten()[: self.state_count], update_values.flatten()[
        #         self.state_count :
        #     ]
        # else:
        #     next_x, next_xp = self.pass_through_update(rootsfound, t, y, yp)
        #     return next_x, next_xp

    def events(self, t, y, yp):
        return np.array(self._events(y, yp, self.result.p, t)).reshape(-1)

    def dae_initial_conditions(self, p):
        ic_values = self.dae_initial_condition_solver(p)

        return ic_values[0].reshape(-1), ic_values[1].reshape(-1)

    def terminal_terms(self, t, y, yp):
        pass

    def integrand_terms(self, t, y, yp):
        pass

    def jac(self, *args):
        pass

    def __call__(self, p):
        self.result = IDAResult(p=self.p, system=self)  # initialize the results?
        self.system_solver.simulate(one_step=self.one_step, max_step=self.max_step)
        result = self.result
        result.t = np.array(result.t)
        result.y = np.array(result.y)
        result.yp = np.array(result.yp)
        self.result = None

        return result


@dataclass
class DAEResult(sgm.ResultBase, sgm.ResultMixin):
    yp: list[list] = field(default_factory=list)

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
            o=self.o,
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
class IDAResult(DAEResult):
    IDAS_mems: list[idas.IDACreate] = field(default_factory=list)
    adjoint_mems: list[idas.IDACreateB] = field(default_factory=list)


# AdjointResultMixin

# AdjointResult

# SweepingGradientMethod

# DAETrajectoryAnalysisSGM?
