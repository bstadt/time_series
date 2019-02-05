import numpy as np


def col(x):
    return np.array(x).reshape(-1, 1)


class MovingAverage:

    def __init__(self,
                 order,
                 coefs,
                 innovator_fn=lambda t : np.random.normal(0, 1)):

        self._order = order
        self._coefs = col(coefs)
        self._innovator_fn = innovator_fn
        self._history = [0 for _ in range(order)]


    def step(self):

        relevant_history = col(self._history[-1*self._order:])
        #list is reversed since we are going to dot with _coefs
        #and _coefs are passed with the most recent entry first
        relevant_history = relevant_history[::-1]

        t = len(self._history) - self._order

        cur_val = self._coefs.T @ relevant_history + self._innovator_fn(t)
        cur_val = float(cur_val)
        self._history.append(cur_val)

        return cur_val
