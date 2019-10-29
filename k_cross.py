import random
class kcross():
    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def kcross(self, batch_x, batch_y):
        combined = list(zip(batch_x, batch_y))
        shuffled = sorted(combined, key=lambda k: random.random())
        fold = (list(self.chunks(shuffled, 10)))
        return fold
