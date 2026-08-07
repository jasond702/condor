from sksundae import ida
import sundials4py as sun4py
from sundials4py import idas
from sundials4py.core import (
    SUNContext_Create,
    SUN_COMM_NULL,
    N_VNew_Serial,
    N_VGetArrayPointer,
    SUN_SUCCESS,
)
from dataclasses import dataclass, field
import numpy as np


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


# Probably Sunset the solver :(
class SundaeSolver(SolverMixin):
    def __init__(
        self,
        system,
        atol=1e-12,
        rtol=1e-6,
    ):
        self.system = system

    def simulate(self):
        system = self.system
        results = system.result

        last_x = system.initial_state
        last_xp = system.initial_dot

        time_generator = system.time_generator()
        last_t = next(time_generator)

        # each iteration of this loop simulates until next generated time
        while True:
            # breakpoint()
            next_t = next(time_generator)
            if np.isinf(next_t):
                break
            if next_t < 0:
                # breakpoint()
                pass
            """
            if self.adaptive_min_steps:
                solver.set_options(
                    max_step_size=np.abs(next_t - last_t) / self.adaptive_min_steps
                )
            """
            solver_res = self.system.solver.init_step(last_t, last_x, last_xp)
            self.store_result(
                np.copy(solver_res.t),
                np.copy(solver_res.y),
                np.copy(solver_res.yp),
            )

            # solver.set_options(tstop=next_t)
            # integration_direction = np.sign(next_t - last_t)

            # each iteration of this loop is one step until next event or time stop
            while True:
                # solver_res = solver.init_step(
                #     self.start_time, self.initial_state, self.initial_dot
                # )
                # if next_t > system.final_time:
                #     break
                # breakpoint()
                # breakpoint()
                solver_res = self.system.solver.step(last_t + system.time_step)
                # next_state = self.system.update()
                # if solver_res.flag < 0:
                #     breakpoint()

                self.store_result(
                    np.copy(solver_res.t),
                    np.copy(solver_res.y),
                    np.copy(solver_res.yp),
                )
                last_t = solver_res.t
                if last_t + system.time_step >= system.final_time:
                    solver_res = self.system.solver.step(system.final_time)
                    self.store_result(
                        np.copy(solver_res.t),
                        np.copy(solver_res.y),
                        np.copy(solver_res.yp),
                    )
                    break
                # breakpoint()
                """
                if solver_res.flag == StatusEnum.ROOT_RETURN:
                    rootsfound = solver.rootinfo()
                """

                """
                if solver_res.flag == StatusEnum.TSTOP_RETURN:
                    # assume this is associated with an event
                    # does occur on time_switch but not sp_lqr
                    gs = system.events(results.t[-1], results.x[-1])
                    min_e = np.abs(gs).min()
                    rootsfound = (gs == min_e).astype(int)
                """
                # print("hi")
                # if solver_res.flag in (StatusEnum.TSTOP_RETURN, StatusEnum.ROOT_RETURN):
                #     idx = len(results.t)
                #     # results.e.append(Root(idx, rootsfound))
                #     next_x = system.update(
                #         results.t[-1],
                #         results.x[-1],
                #         rootsfound,
                #     )
                #     try:
                #         terminate = np.any(rootsfound[system.terminating] != 0)
                #     except Exception as e:
                #         print("Hit exemption:")
                #         print(e)
                #         print("You may try to continue through or exit")
                #         breakpoint()
                #     self.store_result(np.copy(solver_res.values.t), next_x)

                # if terminate:
                #     self.store_result(np.copy(solver_res.values.t), next_x)
                #     return

                # solver.init_step(solver_res.values.t, next_x)
                # last_x = next_x

                # if (integration_direction * solver_res.values.t) >= (
                #     integration_direction * next_t
                # ):
                #     break
                # if solver_res.flag == StatusEnum.TSTOP_RETURN:
                #     # does occur on time_switch but not sp_lqr
                #     break
            # last_t = next_t


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

    class SundialsDAESystem:
        def __init__(
            self, p, state_count, dot_count, initial_conditions, residual_func
        ):
            self.p = np.array(p, dtype=sun4py.core.sunrealtype)
            self.dot_count = dot_count
            self.initial_conditions = initial_conditions
            self.residual_func = residual_func
            self.NEQ = state_count

        def set_init_cond(self, yvec, ypvec, y0, yp0):
            y = N_VGetArrayPointer(yvec)
            yp = N_VGetArrayPointer(ypvec)
            for i, elem in enumerate(y0):
                y[i] = y0[i]
                yp[i] = yp0[i]
            return 0

        def residualfunction(self, t, yvec, ypvec, resvec, _):
            y = N_VGetArrayPointer(yvec)
            yp = N_VGetArrayPointer(ypvec)
            res = N_VGetArrayPointer(resvec)
            res[:, None] = self.residual_func(
                y[: self.dot_count],
                y[self.dot_count :],
                yp[: self.dot_count],
                self.initial_conditions.parameter.flatten(),
            )
            return 0

    def simulate(self):
        system = self.system
        results = system.result
        last_x = system.initial_state
        last_xp = system.initial_dot

        time_generator = system.time_generator()
        last_t = next(time_generator)

        # Step 1: Create the SUNDIALS Context
        status, sunctx = SUNContext_Create(SUN_COMM_NULL)
        assert status == SUN_SUCCESS

        # Step 2: Set up the DAE problem
        dae_system = Sundials4PySolver.SundialsDAESystem(
            system.p,
            system.state_count,
            system.dot_count,
            system.initial_conditions,
            system.residual_func,
        )

        y = N_VNew_Serial(dae_system.NEQ, sunctx)
        yp = N_VNew_Serial(dae_system.NEQ, sunctx)
        assert y is not None
        assert yp is not None

        dae_system.set_init_cond(y, yp, last_x, last_xp)

        # Step 3: Create and Initialize IDAS
        IDAS = idas.IDACreate(sunctx)
        assert IDAS is not None

        status = idas.IDAInit(IDAS.get(), dae_system.residualfunction, 0.0, y, yp)
        assert status == idas.IDA_SUCCESS

        # algidx = N_VNew_Serial(testProblem.NEQ, sunctx)
        # idx = N_VGetArrayPointer(algidx)
        # idx_list = np.ones(system.state_count)
        # idx_list[-(system.state_count - system.dot_count) :] = 0.0
        # idx[:] = idx_list

        # status = idas.IDASetId(IDAS.get(), algidx)
        # assert status == idas.IDA_SUCCESS

        # Step 4: Set IDAS options (tolerances, linear solver, etc.)
        reltol = self.rtol
        abstol = self.atol
        status = idas.IDASStolerances(IDAS.get(), reltol, abstol)
        assert status == idas.IDA_SUCCESS

        A = sun4py.core.SUNDenseMatrix(dae_system.NEQ, dae_system.NEQ, sunctx)
        assert A is not None

        LS = sun4py.core.SUNLinSol_Dense(y, A, sunctx)
        assert LS is not None

        status = idas.IDASetLinearSolver(IDAS.get(), LS, A)
        assert status == idas.IDA_SUCCESS

        # Optional to set Jacobian

        # Step 5: Advance the DAE in time
        tret = 0.0
        yarr = N_VGetArrayPointer(y)
        yparr = N_VGetArrayPointer(yp)

        t_vals = [tret]
        y_vals = [yarr]
        yp_vals = [yparr]
        self.store_result(
            np.copy(tret),
            np.copy(yarr),
            np.copy(yparr),
        )

        # status = idas.IDACalcIC(IDAS.get(), idas.IDA_YA_YDP_INIT, 1e-2)
        # status = idas.IDACalcIC(IDAS.get(), idas.IDA_Y_INIT, 1e-2)
        # assert status == idas.IDA_SUCCESS

        while tret < system.final_time:
            status, tret = idas.IDASolve(
                IDAS.get(),
                tret + system.time_step,
                y,
                yp,
                # idas.IDA_NORMAL,
                idas.IDA_ONE_STEP,
            )
            assert status == idas.IDA_SUCCESS

            self.store_result(
                np.copy(tret),
                np.copy(yarr),
                np.copy(yparr),
            )

        # Step 6: Get IDAS Statistics (optional)


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
        dot_count,
        count_diff,
        residual_func,
        time_generator,
        **analysis_options,
    ):
        self.state_count = state_count
        self.dot_count = dot_count
        self.residual_func = residual_func
        self.initial_conditions = initial_conditions
        self.p = p

        self.start_time = start_time
        self.final_time = final_time
        self.result = None
        self._time_generator = time_generator
        self.time_step = (final_time - start_time) / (
            analysis_options.pop("num_steps") - 1
        )

        self.initial_state = initial_conditions.variable.flatten()[:state_count]
        self.initial_dot = initial_conditions.variable.flatten()[state_count:]

        # i feel like this will eventually throw an error (think it over, maybe not?)

        if len(self.initial_state) < len(self.initial_dot):
            self.initial_state = np.append(self.initial_state, [0] * count_diff)
        elif len(self.initial_state) > len(self.initial_dot):
            self.initial_dot = np.append(self.initial_dot, [0] * count_diff)

        # this was all for pysundae

        # def residualfunction(t, y, yp, res):
        #     res[:, None] = self.residual_func(
        #         y[: self.dot_count],
        #         y[self.dot_count :],
        #         yp[: self.dot_count],
        #         self.initial_conditions.parameter.flatten(),
        #     )

        # if all(x == 0 for x in self.initial_state):
        #     self.solver = ida.IDA(
        #         residualfunction,
        #         atol=1e-8,
        #         algebraic_idx=list(range(self.state_count)[self.dot_count :]),
        #         calc_initcond="y0",
        #     )
        #     breakpoint()
        # elif all(x == 0 for x in self.initial_dot):
        #     self.solver = ida.IDA(
        #         residualfunction,
        #         atol=1e-8,
        #         algebraic_idx=list(range(self.state_count)[self.dot_count :]),
        #         calc_initcond="yp0",
        #     )
        #     breakpoint()
        # else:
        #     self.solver = ida.IDA(
        #         residualfunction,
        #         atol=1e-8,
        #         algebraic_idx=list(range(self.state_count)[self.dot_count :]),
        #     )
        #     breakpoint()

        self.start_solver()

    def start_solver(self):
        # self.system_solver = SundaeSolver(system=self)
        self.system_solver = Sundials4PySolver(system=self)

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
    # e: list[Root] = field(default_factory=list)

    def __getitem__(self, key):
        return self.__class__(
            *tuple(getattr(self, field.name) for field in fields(self)[:-3]),
            t=self.t[key],
            y=self.y[key],
            yp=self.yp[key],
            # e=self.e[key],
        )

    def save(self, filename):
        # e_idxs = [e.index for e in self.e]
        # e_roots = [e.rootsfound for e in self.e]
        np.savez_compressed(
            filename,
            # e_idxs=e_idxs,
            # e_roots=e_roots,
            t=self.t,
            y=self.y,
            yp=self.yp,
            p=self.p,
        )

    @classmethod
    def load(cls, filename):
        data = dict(np.load(filename))
        # data["e"] = [
        #     Root(index=int(ei), rootsfound=er)
        #     for ei, er in zip(data.pop("e_idxs"), data.pop("e_roots"))
        # ]
        return cls(system=None, **data)


@dataclass
class DAEResult(ResultBase, ResultMixin):
    pass
