import numpy as np
import random
import time


class Sublist:
    def __init__(self, size: int, maxentry: int, sublist_sizes: np.ndarray, data,
                 ):
        """
        Sublist based on array
        """
        nlist = len(sublist_sizes)
        self.size = size
        self.array = -np.ones(size, dtype=int)
        self.index = np.zeros((nlist, 2), dtype=int)  # index[e]=[start_position, length]

        self.nlist = nlist
        self.maxentry = maxentry

        self.index_recorder = np.zeros(size, dtype=int)
        self.sublist_recorder = np.zeros(size, dtype=int)
        self.index_recorder[:] = -1
        self.sublist_recorder[:] = -1

        empty_space = int((size - maxentry) / nlist)
        inds = np.zeros(1 + nlist)
        inds[1:] = np.add.accumulate(sublist_sizes)
        inds = inds.astype(int)
        for i, (l, u) in enumerate(zip(inds[:-1], inds[1:])):
            self.index[i, 0] = l + i * empty_space
            self.index[i, 1] = sublist_sizes[i]
            self.array[l + i * empty_space:u + i * empty_space] = data[l:u]
            self.index_recorder[data[l:u]] = np.arange(l + i * empty_space, u + i * empty_space)
            self.sublist_recorder[data[l:u]] = i

        self.len = len(data)
        if self.len > self.maxentry:
            print("WARNING: Excessive entry input may cause memory leak.")

    def push(self, e, item):
        if item < 0 or item >= self.size:
            raise ValueError("item should be in range [0,size)")

        ind = self.index[e, 0] + self.index[e, 1]
        neighbor = self.array[ind]
        e_neighbor = e
        while True:
            if e_neighbor == self.nlist - 1:
                break
            if ind < self.index[e_neighbor + 1, 0]:
                break
            e_neighbor += 1
        self.e = e
        self.e_neighbor = e_neighbor
        self.array[ind] = item
        self.index[e, 1] += 1
        self.index_recorder[item] = ind
        self.sublist_recorder[item] = e
        if e == self.nlist - 1:
            self.len += 1
            if self.index[e, 0] + self.index[e, 1] == self.size:
                self.__init__(self.size, self.maxentry, self.index[:, 1],
                              np.hstack([self._get_sublist(e) for e in range(self.nlist)]))
                print("WARNING: Array bound reached. Restructuring array.")
        elif e_neighbor > e:
            # new space needed
            self.index[e + 1:e_neighbor + 1, 0] += 1
            if self.index[e_neighbor, 1] > 0:
                self.index[e_neighbor, 1] -= 1
            if neighbor != -1:

                self.push(e_neighbor, neighbor)
            else:
                self.len += 1
                if self.index[-1, 0] + self.index[-1, 1] == self.size:
                    self.__init__(self.size, self.maxentry, self.index[:, 1],
                                  np.hstack([self._get_sublist(e) for e in range(self.nlist)]))
                    print("WARNING: Array bound reached. Restructuring array.")
                    # print("check")
        else:
            self.len += 1
            if self.index[-1, 0] + self.index[-1, 1] == self.size:
                self.__init__(self.size, self.maxentry, self.index[:, 1],
                              np.hstack([self._get_sublist(e) for e in range(self.nlist)]))
                print("WARNING: Array bound reached. Restructuring array.")

    def extend(self, e, array):
        pass

    def __len__(self):
        return self.len

    def get(self, ind):
        return self.array[ind]

    def get_rand(self, e):
        rand = np.random.randint(self.index[e, 1])
        return self.array[rand + self.index[e, 0]]

    def pop(self, value):
        ind = self.index_recorder[value]
        opt = value
        e = self.sublist_recorder[value]
        if e == -1:
            raise ValueError("%d is not stored." % value)
        end_ind = self.index[e, 0] + self.index[e, 1] - 1
        end = self.array[end_ind]
        self.array[ind] = self.array[end_ind]
        self.array[end_ind] = -1
        self.index[e, 1] -= 1
        self.index_recorder[opt] = -1
        self.index_recorder[end] = ind
        self.sublist_recorder[opt] = -1
        self.len -= 1
        return opt

    def tolist(self):
        return np.hstack([self._get_sublist(e) for e in range(self.nlist)])

    def _get_sublist(self, e):
        """For testing, don't need to be called."""
        return self.array[self.index[e, 0]:self.index[e, 0] + self.index[e, 1]]


def test_sublist():
    N = 20000
    n = 15000
    nlist = 3
    list_sizes = [5000, 5000, 5000]
    data = np.arange(n)

    # currect data structure
    dic = {}
    accu_sizes = np.zeros(nlist + 1, dtype=int)
    accu_sizes[1:] = np.add.accumulate(list_sizes)
    for i, (l, u) in enumerate(zip(accu_sizes[:-1], accu_sizes[1:])):
        dic[i] = data[l:u].tolist()

    # def check(actual: Sublist, expect: dict):
    #     # actual: sublist
    #     # expect: dictionary based structure
    #     if actual.nlist != len(expect.keys()):
    #         print("Fail at nlist")
    #         return False
    #     if len(actual) != sum([len(k) for k in expect.values()]):
    #         print("Fail at len")
    #         return False
    #     for e in range(actual.nlist):
    #         if not np.all(np.sort(actual._get_sublist(e)) == np.sort(expect[e])):
    #             print("Fail at values")
    #             return False
    #         for d in expect[e]:
    #             if not d == actual.get(actual.index_recorder[d]):
    #                 print("Fail at index recorder")
    #                 return False
    #         if not np.all(np.sort(np.where(actual.sublist_recorder == e)) == np.sort(expect[e])):
    #             print("Fail at sublist recorder")
    #             return False
    #     return True
    #
    # print("Correct data: ", data)
    # print("Sublist sizes: ", list_sizes)
    # print("\n")
    #
    # # test_initialization
    sublist = Sublist(N, n, list_sizes, data)
    # print(sublist.array)
    # print(sublist.index_recorder)
    # print("Recording correct: ",
    #       np.all(np.sort(data) == sublist.array[sublist.index_recorder[sublist.index_recorder >= 0]]))
    #
    # if not check(sublist, dic):
    #     raise RuntimeError("Initialization fail.")
    #
    # # test push
    # n_trial = 1000000
    # success = True
    # for i in range(n_trial):
    #
    #     # pop one, push one
    #     pop_ind = np.random.choice(sublist.index_recorder[sublist.index_recorder >= 0])
    #     pop = sublist.get(pop_ind)
    #     e = sublist.sublist_recorder[pop]
    #     sublist.pop(pop)
    #     dic[e].remove(pop)
    #     if not check(sublist, dic):
    #         raise RuntimeError("Pop fail: %d, e=%d, pop=%d, pop_ind=%d" % (i, e, pop, pop_ind))
    #     else:
    #         if i % 1000 == 0:
    #             print("Pop successful: %d, e=%d, pop=%d, pop_ind=%d" % (i, e, pop, pop_ind))
    #
    #     e = np.random.choice(nlist)
    #     sublist.push(e, pop)
    #     dic[e].append(pop)
    #     if not check(sublist, dic):
    #         raise RuntimeError("Push fail: %d, e=%d, pop=%d, pop_ind=%d" % (i, e, pop, pop_ind))
    #     else:
    #         if i % 1000 == 0:
    #             print("Push successful: %d, e=%d, pop=%d, pop_ind=%d" % (i, e, pop, pop_ind))
    #
    # tolist = sublist.tolist()
    # print(tolist)
    # if not np.all(np.sort(tolist) == np.sort(data)):
    #     raise RuntimeError("tolist() Error.")

    # test speed
    print("Testing Sublist speed.")
    ntrial = 10000
    begin_t = time.time()
    for i in range(ntrial):
        sublist.get_rand(np.random.randint(nlist))
    end_t = time.time()
    print("get_rand() x%d time: %f" % (ntrial, end_t - begin_t))

    begin_t = time.time()
    for i in range(ntrial):
        value = np.random.randint(n)
        sublist.pop(value)
        sublist.push(np.random.randint(nlist), value)
    end_t = time.time()
    print("push() and pop() x%d time: %f" % (ntrial, end_t - begin_t))


class sublist_dic:
    def __init__(self, size: int, nlist: int, dic: dict):
        self.size = size
        self.nlist = nlist
        self.dic=dic
        self.sublist_recorder=-np.ones(size,dtype=int)
        for key, sublist in self.dic.items():
            sub = list(sublist)
            self.sublist_recorder[sub] = key

    def push(self, e, item):
        if e < 0 or e >= self.nlist:
            raise ValueError("Invalid sublist index.")
        try:
            self.dic[e].add(item)
        except KeyError:
            self.dic[e] = {item}
        self.sublist_recorder[item] = e

    def extend(self, e, array):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get(self, ind):
        raise NotImplementedError

    def get_rand(self, e):
        return random.choice(tuple(self.dic[e]))

    def pop(self, value):
        self.dic[self.sublist_recorder[value]].remove(value)
        self.sublist_recorder[value] = -1

    def tolist(self):
        return np.hstack([list(l) for l in self.dic.values()]).astype(int)

    def _get_sublist(self, e):
        try:
            return list(self.dic[e])
        except KeyError:
            return []

    def sublist_sizes(self):
        return np.array([len(sub) for sub in self.dic.values()],dtype=int)


def test_sublist_dic():
    # test speed

    print("Testing: sublist_dic")

    nlist = 3
    n = 15000
    dic = {}
    for i in range(nlist):
        dic[i] = set(range(int(i * n / nlist), int((i + 1) * n / nlist)))
    sublist = sublist_dic(n, nlist, dic)

    ntrial = 10000
    begin_t = time.time()
    for i in range(ntrial):
        sublist.get_rand(np.random.randint(nlist))

    end_t = time.time()
    print("get_rand() x%d time: %f" % (ntrial, end_t - begin_t))

    begin_t = time.time()
    for i in range(ntrial):
        value = np.random.randint(n)
        sublist.pop(value)
        sublist.push(np.random.randint(nlist), value)
    end_t = time.time()
    print("push() and pop() x%d time: %f" % (ntrial, end_t - begin_t))
    # sublist=sublist_dic(nlist,dic)
    # correct data structure
    # cor = dic.copy()
    # accu_sizes = np.zeros(nlist + 1, dtype=int)
    # accu_sizes[1:] = np.add.accumulate(list_sizes)
    # for i, (l, u) in enumerate(zip(accu_sizes[:-1], accu_sizes[1:])):
    #     dic[i] = data[l:u].tolist()

    # def check(actual: Sublist, expect: dict):
    #     # actual: sublist
    #     # expect: dictionary based structure
    #     if actual.nlist != len(expect.keys()):
    #         print("Fail at nlist")
    #         return False
    #     if len(actual) != sum([len(k) for k in expect.values()]):
    #         print("Fail at len")
    #         return False
    #     for e in range(actual.nlist):
    #         if not np.all(np.sort(actual._get_sublist(e)) == np.sort(expect[e])):
    #             print("Fail at values")
    #             return False
    #         for d in expect[e]:
    #             if not d == actual.get(actual.index_recorder[d]):
    #                 print("Fail at index recorder")
    #                 return False
    #         if not np.all(np.sort(np.where(actual.sublist_recorder == e)) == np.sort(expect[e])):
    #             print("Fail at sublist recorder")
    #             return False
    #     return True

    # print("Correct data: ", data)
    # print("Sublist sizes: ", list_sizes)
    # print("\n")
    #
    # # test_initialization
    # sublist = Sublist(N, n, list_sizes, data)
    # print(sublist.array)
    # print(sublist.index_recorder)
    # print("Recording correct: ",
    #       np.all(np.sort(data) == sublist.array[sublist.index_recorder[sublist.index_recorder >= 0]]))
    #
    # if not check(sublist, dic):
    #     raise RuntimeError("Initialization fail.")
    #
    # # test push
    # n_trial = 1000000
    # success = True
    # for i in range(n_trial):
    #
    #     # pop one, push one
    #     pop_ind = np.random.choice(sublist.index_recorder[sublist.index_recorder >= 0])
    #     pop = sublist.get(pop_ind)
    #     e = sublist.sublist_recorder[pop]
    #     sublist.pop(pop)
    #     dic[e].remove(pop)
    #     if not check(sublist, dic):
    #         raise RuntimeError("Pop fail: %d, e=%d, pop=%d, pop_ind=%d" % (i, e, pop, pop_ind))
    #     else:
    #         if i % 1000 == 0:
    #             print("Pop successful: %d, e=%d, pop=%d, pop_ind=%d" % (i, e, pop, pop_ind))
    #
    #     e = np.random.choice(nlist)
    #     sublist.push(e, pop)
    #     dic[e].append(pop)
    #     if not check(sublist, dic):
    #         raise RuntimeError("Push fail: %d, e=%d, pop=%d, pop_ind=%d" % (i, e, pop, pop_ind))
    #     else:
    #         if i % 1000 == 0:
    #             print("Push successful: %d, e=%d, pop=%d, pop_ind=%d" % (i, e, pop, pop_ind))
    #
    # tolist = sublist.tolist()
    # print(tolist)
    # if not np.all(np.sort(tolist) == np.sort(data)):
    #     raise RuntimeError("tolist() Error.")


if __name__ == '__main__':
    test_sublist()
    test_sublist_dic()
