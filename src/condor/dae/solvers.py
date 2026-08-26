from sksundae import ida
import sundials4py as sun4py
from sundials4py import idas
from sundials4py.core import (
    SUNContext_Create,
    SUN_COMM_NULL,
    N_VNew_Serial,
    N_VGetArrayPointer,
    SUN_SUCCESS,
    sunrealtype,
)
from dataclasses import dataclass, field
from typing import NamedTuple
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

        system.set_init_cond(y, yp, last_x, last_xp)

        # Step 3: Create and Initialize IDAS
        IDAS = idas.IDACreate(sunctx)
        assert IDAS is not None

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
        abstol = self.atol
        status = idas.IDASStolerances(IDAS.get(), reltol, abstol)
        assert status == idas.IDA_SUCCESS

        # set up root functions (when do events occur)
        status = idas.IDARootInit(IDAS.get(), 3, system.RobertsonEventsTEMP)
        assert status == idas.IDA_SUCCESS

        # status = idas.IDASetRootDirection(IDAS.get(), [0, 0, -1])
        # assert status == idas.IDA_SUCCESS

        # status = idas.IDASetNoInactiveRootWarn(IDAS.get())
        # assert status == idas.IDA_SUCCESS

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

        # Step 5: Advance the DAE in time
        if "logspace" in system.analysis_options:
            tret = 10**system.start_time
        else:
            tret = system.start_time

        yarr = N_VGetArrayPointer(y)
        yparr = N_VGetArrayPointer(yp)

        t_vals = [tret]
        y_vals = [yarr]
        yp_vals = [yparr]
        # copy initial conditions into array
        self.store_result(
            np.copy(tret),
            np.copy(yarr),
            np.copy(yparr),
        )
        # breakpoint()
        iout = 0
        while True:
            next_t = next(time_generator)
            if np.isinf(next_t):
                break
            if next_t < 0:
                # breakpoint()
                pass
            rootsfound = np.zeros(3, dtype=sunrealtype)  # check later
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
        time_generator,
        **analysis_options,
    ):
        self.state_count = state_count
        self.NEQ = state_count
        self.analysis_options = analysis_options
        self.residual_func = residual_func
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

        self.start_solver()

    def start_solver(self):
        self.system_solver = Sundials4PySolver(system=self)

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
            y,
            yp,
            self.initial_conditions.parameter.flatten(),
        )
        # breakpoint()
        return 0

    def RobertsonEventsTEMP(self, t, yvec, ypvec, evvec, _):
        y = N_VGetArrayPointer(yvec)
        yp = N_VGetArrayPointer(ypvec)
        # evvec[0] = y[0] - 0.0001
        # evvec[1] = y[2] - 0.01
        # breakpoint()
        evvec[0] = y[0] - 0.1
        evvec[1] = y[1] * 1e4 - 0.2
        evvec[2] = y[2] - 0.7
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
        result.e = np.array()
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
