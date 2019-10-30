import random
class kcross():
    def slice(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


    def getTestSlice(self, fold, n, j, xfold):
        testFold = fold[j]
        restFold = [x for i,x in enumerate(fold) if i!=j]

        if j < 0:
            return xfold
        else:
            xfold.append((testFold, restFold))
            return self.getTestChunk(fold, len(fold), (j - 1), xfold)


    def kcross(self, batch_x, batch_y):
        combined = list(zip(batch_x, batch_y))
        shuffled = sorted(combined, key=lambda k: random.random())
        fold = (list(self.slice(shuffled, 100)))
        xfold = []
        xfold = self.getTestSlice(fold, len(fold), len(fold) - 1, xfold)

        return xfold
