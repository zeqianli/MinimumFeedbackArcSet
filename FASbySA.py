"""Python realization of Feedback Arc Set algorithm by Simulated Annealing.

Translated from C++ implementation by Prof. Haijun zhou.

Author: Li Zeqian

==================

July 20, 2018: I wrote this a while ago. It worked but was pretty slow. I left a
    bunch of debugging stuff. Also need optimization on vectorization and 
    these network representations. 
    
    I'll just leave it like that. Email me if you need anything. 

==================

"""

import numpy as np
import scipy.sparse as sp
from structure import Sublist, sublist_dic
import time
import logging


class FASbySA:
    """ FASbySA class.

    Attributes:
        nvertex     : int, # vertices.
        narc        : int, # total arcs.

        adjmtx      : sparse matrix, network of simple arcs (x double, x arcs)
        simple_arcs : ndarray(nsimple, 2), simple (unidirectional) arcs
        double_arcs : ndarray(ndouble, 2), double arcs        
        self_arcs   : ndarray(nself, 2), self arcs
        in_degree   : ndarray(nvertex,), in-degrees of vertices
        out_degree  : ndarray(nvertex,), out-degrees of vertices

        h           : ndarray(nvertex,), hierarchy of vertices
        min_h       : ndarray(nvertex,), best hierarchy
        nFAS        : int, FAS size (only include simple arcs)
        min_nFAS    : int, minimum FAS size

        initialized : bool, if the object is initialized
        simulated   : bool, if SA is performed

    """

    def __init__(self):
        # Fixed fields
        self.nvertex = 0
        self.narc, self.nsimple, self.ndouble, self.nself = 0, 0, 0, 0

        self.all_arcs, self.simple_arcs, self.double_arcs, self.self_arcs = None, None, None, None
        self.in_degree, self.out_degree = None, None
        self.max_in_degree, self.max_out_degree, self.max_degree = 0, 0, 0

        self.adjmtx, self.m_csr, self.m_csc = None, None, None  # adjacency matrix. (i,j) : i->j
        self.nscc, self.scc_ind = 0, None

        self.parents_of, self.children_of = {}, {}  # neighborhood representation of networks.
        self.in_arcs_of, self.out_arcs_of = {}, {}  # only include simple arcs

        # Following fields are related to FAS
        # Updated in each iteration
        self.h: np.ndarray = None
        self.min_h: np.ndarray = None  # h->v
        self.h_of_v: np.ndarray = None
        self.min_h_of_v: np.ndarray = None  # v->h
        self.nFAS, self.min_nFAS = 0, 0
        self.min_FAS: np.ndarray = None

        self.e_mveprt: sublist_dic = None  # sublists of moving parent energy
        self.e_mvecld: sublist_dic = None  # sublists of moving child energy

        self.initialized = False
        self.simulated = False

        # self._energy_change_batch=np.vectorize(self._energy_change)

    def initialize(self, nvertex, narc, edges, inspect=True):
        """Initialization by parameters

        nvertex: # vertex
        narc: # arc
        edges: ndarray(n, 2), directional edges
        """

        self.nvertex, self.narc = nvertex, narc

        # find simple/double/self arcs
        
        # remove repetitive edges
        edge_hash = edges[:, 0] * (nvertex + 1) + edges[:, 1]
        edge_hash_unique, edge_unique_ind = np.unique(edge_hash, return_index=True)
        edges = edges[edge_unique_ind]
        self.all_arcs = edges.copy()

        # pick self arcs
        is_selfarcs = edges[:, 0] == edges[:, 1] 

        # pick double arcs
        temp1 = edges[:, 0] * (self.nvertex + 1) + edges[:, 1]
        temp2 = edges[:, 1] * (self.nvertex + 1) + edges[:, 0]
        is_doublearcs = np.in1d(temp1, temp2)  

        self.self_arcs = edges[is_selfarcs, :]
        self.double_arcs = edges[np.logical_and(is_doublearcs, np.logical_not(is_selfarcs)), :]
        self.simple_arcs = edges[np.logical_not(np.logical_or(is_selfarcs, is_doublearcs)), :]
        self.nsimple = len(self.simple_arcs)
        self.ndouble = len(self.double_arcs)
        self.nself = len(self.self_arcs)

        # TODO: what more information is needed here?
        self.adjmtx = sp.coo_matrix((np.ones(self.nsimple), (self.simple_arcs[:, 0], self.simple_arcs[:, 1])),
                                    shape=(self.nvertex, self.nvertex))
        self.m_csr = self.adjmtx.tocsr()
        self.m_csc = self.adjmtx.tocsc()

        self.out_degree = (self.m_csr @ np.ones(self.nvertex)).astype(int)
        self.in_degree = (np.ones(self.nvertex) @ self.m_csr).astype(int)
        self.max_in_degree = max(self.in_degree)
        self.max_out_degree = max(self.out_degree)
        self.max_degree = max(self.max_in_degree, self.max_out_degree)

        for i in range(self.nvertex):
            self.parents_of[i], self.children_of[i] = [], []
            self.in_arcs_of[i], self.out_arcs_of[i] = [], []
        for i, (p, c) in enumerate(self.simple_arcs):
            self.parents_of[c].append(p)
            self.children_of[p].append(c)
            self.in_arcs_of[c].append(i)
            self.out_arcs_of[p].append(i)
        
        # for i in range(self.nvertex):
        #      self.parents_of[i] = np.array(self.parents_of[i])
        #      self.children_of[i] = np.array(self.children_of[i])
        #      self.in_arcs_of[i] = np.array(self.in_arcs_of[i])
        #      self.out_arcs_of[i] = np.array(self.out_arcs_of[i])
        

        # Initialize a hierarchy by 1. aligning by SCC order 2. randomizing within each SCC.
        # SCC: strongly connected components
        # TODO: check SCC
        nscc, scc_ind = sp.csgraph.connected_components(self.adjmtx, directed=True, connection='strong',
                                                        return_labels=True)  # scc is indexed automatically by its feed-foward order

        self.nscc, self.scc_ind = nscc, scc_ind
        hierarchy = np.argsort(scc_ind)
        scc_ind_sorted = scc_ind[hierarchy]
        for i in range(nscc):
            hierarchy[scc_ind_sorted == i] = np.random.permutation(hierarchy[scc_ind_sorted == i])

        self.h = hierarchy
        self.h_of_v = np.argsort(self.h)
        self.min_h = hierarchy.copy()
        self.min_h_of_v = self.h_of_v.copy()

        # Initialize FAS
        self.nFAS = self._count_FAS()
        self.min_nFAS = self.nFAS
        self.min_FAS = self._FAS()
        if self.nFAS == 0:
            print("This graph is feed-forward (excluding double/self arcs).")

        # Output some info
        if inspect:
            print("Graph constructed. Info: ")
            print("Vertex: %d, Arc: %d, Simple-arc: %d, Self-arc: %d, Double-arc: %d"
                  % (self.nvertex, self.narc, self.nsimple, self.nself, self.ndouble))
            print("Max in-degree: %d, Max out-degree: %d"
                  % (self.max_in_degree, self.max_out_degree))
            print("SCC #: %d" % nscc)
            print("Initial FAS size by separating SCCs: %d" % self.nFAS)

        self.initialized = True
        self.simulated = False

    def initialize_by_file(self, fname: str, delimiter=' ', inspect=True):
        """Initialization from text files. 

        File format:
            - 1st line: nvertex, narc
            - Following each line: v1 v2
                (v1, v2) represents an arc v1->v2, 
            - Delimiter ' ' by default.
        """
        # format check
        try:
            edges = np.loadtxt(fname, delimiter=delimiter, dtype=int).reshape(-1, 2)
        except ValueError:
            raise ValueError("File %s format error." % fname)

        nvertex, narc = edges[0]
        edges = edges[1:, :]
        self.initialize(nvertex, narc, edges, inspect=inspect)

    def simulated_annealing(self, T=1, alpha=0.99, max_move=None, max_fail=50, debug=False,
                            file_record=False,
                            fname_nFAS=None,
                            fname_min_FAS=None):
        """Simulated annealing.

        Return: final temperature"""

        T_0 = T
        # record SA time
        time_total, time_energy, time_sublist, time_other = 0, 0, 0, 0
        begin_total = time.time()

        if max_move is None:
            max_move = 5 * self.nvertex

        if not self.initialized:
            raise RuntimeError("FASbySA state error: graph not initialized.")

        if T <= 0:
            raise ValueError("Temperature should not be negative. ")
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Cooling rate should be in (0,1).")

        if file_record and (fname_nFAS is None or fname_min_FAS is None):
            raise ValueError("Recording file name not given. ")
        if file_record:
            f_nFAS = open(fname_nFAS, 'w+')
            f_min_FAS = open(fname_min_FAS, 'w+')

        self._initialize_SA()

        def check():
            # Variables updated in each iteration:
            # h, h_of_v, min_h, min_h_of_v
            # nFAS, min_nFAS
            # e_mveprt, e_mvecld

            # h/h_of_v consistent
            if not np.all(self.h_of_v[self.h] == np.arange(self.nvertex)):
                raise ValueError("h/h_of_v not consistent.")
            if not np.all(self.min_h_of_v[self.min_h] == np.arange(self.nvertex)):
                raise ValueError("min_h/min_h_of_v not consistent.")

            # recount nFAS
            actual_nFAS = self._count_FAS()
            if not self.nFAS == actual_nFAS:
                raise ValueError("nFAS recording wrong.")
            if self.nFAS < self.min_nFAS:
                raise ValueError("Smaller FAS not recorded.")

            # e_mveprt, e_mvecld
            # 1. FAS stored correct
            expect_FAS = np.sort(self._FAS())
            actual_FAS_mveprt = np.sort(self.e_mveprt.tolist())
            actual_FAS_mvecld = np.sort(self.e_mvecld.tolist())
            if not np.all(expect_FAS == actual_FAS_mveprt):
                raise ValueError("e_mveprt FAS wrong. \n Expect FAS: %s \n Actual FAS: %s"
                                 % (expect_FAS, actual_FAS_mveprt))
            if not np.all(expect_FAS == actual_FAS_mvecld):
                raise ValueError("e_mvecld FAS wrong. \n Expect FAS %s \n Actual FAS: %s"
                                 % (expect_FAS, actual_FAS_mveprt))

            # 2. FAS energy correct; also test energy calculation function
            expect_e_mveprt, expect_e_mvecld = np.zeros(expect_FAS.shape), np.zeros(expect_FAS.shape)
            actual_e_mveprt, actual_e_mvecld = np.zeros(expect_FAS.shape), np.zeros(expect_FAS.shape)
            for i, arc_ind in enumerate(expect_FAS):
                # Error raised here if _energy_change has error.
                expect_e_mveprt[i], expect_e_mvecld[i] = self._energy_change(self.simple_arcs[arc_ind], floor0=True)
                actual_e_mveprt[i] = self.e_mveprt.sublist_recorder[arc_ind]
                actual_e_mvecld[i] = self.e_mvecld.sublist_recorder[arc_ind]

            prt_diff = np.where(expect_e_mveprt != actual_e_mveprt)[0]
            cld_diff = np.where(expect_e_mvecld != actual_e_mvecld)[0]
            if len(prt_diff) != 0:
                raise ValueError("parent energy wrong:\n ind=%s,\n expect=%s,\n actual=%s"
                                 % (expect_FAS[prt_diff], expect_e_mveprt[prt_diff], actual_e_mveprt[prt_diff]))
            if len(cld_diff) != 0:
                raise ValueError("child energy wrong:\n ind=%s,\n expect=%s,\n actual=%s"
                                 % (expect_FAS[cld_diff], expect_e_mvecld[cld_diff], actual_e_mvecld[cld_diff]))

        def log_info(arg_names, args):
            for name, arg in zip(arg_names, args):
                logging.debug("%s: %s" % (name, arg))

        if debug:
            check()
        if debug:
            logging.debug("Initialization")
            FAS = self._FAS().tolist()
            log_info(("h", "h_of_v", "FAS-energy"),
                     (self.h, self.h_of_v,
                      list(zip(FAS, [self._energy_change(self.simple_arcs[arc], floor0=True) for arc in FAS]))))
        print("SA initialization finished.")

        if debug:
            logging.debug("performing SA")
        nfail = 0
        while nfail < max_fail:
            fail = True
            # exp_T = np.exp(-np.arange(self.max_degree)/T)
            for nmove in range(max_move):
                # Move parent
                # 1. randomly pick an arc
                prob = np.exp(-np.arange(self.e_mveprt.nlist) / T) * self.e_mveprt.sublist_sizes()
                prob = prob / sum(prob)
                erand = np.random.choice(self.e_mveprt.nlist, p=prob)
                arc_ind = self.e_mveprt.get_rand(erand)
                picked_arc = arc_ind.copy()
                p, c = self.simple_arcs[arc_ind]

                # 2. update all energy related fields:
                # energy
                time_mark = time.time()
                self.nFAS += self._energy_change(self.simple_arcs[arc_ind])[0]
                time_energy += time.time() - time_mark
                # pop from energy list
                time_mark = time.time()
                waitlist = self._update_waitlist(arc_ind)

                for arc_ind in waitlist:
                    try:
                        self.e_mveprt.pop(arc_ind)
                        self.e_mvecld.pop(arc_ind)
                    except KeyError:
                        pass
                    except ValueError:
                        pass
                time_sublist += time.time() - time_mark

                # h,h_of_v
                hc, hp = self.h_of_v[c], self.h_of_v[p]
                self.h_of_v[self.h[hc:hp]] += 1
                self.h_of_v[p] = hc
                self.h[hc + 1:hp + 1] = self.h[hc:hp]
                self.h[hc] = p
                # energy lists;

                for arc_ind in waitlist:
                    if self._is_FAS(self.simple_arcs[arc_ind]):
                        time_mark = time.time()
                        ep, ec = self._energy_change(self.simple_arcs[arc_ind], floor0=True)
                        time_energy += time.time() - time_mark
                        time_mark = time.time()
                        self.e_mveprt.push(ep, arc_ind)
                        self.e_mvecld.push(ec, arc_ind)
                        time_sublist += time.time() - time_mark

                if self.nFAS < self.min_nFAS:
                    fail = False
                    nfail = 0
                    self.min_h = self.h.copy()
                    self.min_h_of_v = self.h_of_v.copy()
                    self.min_nFAS = self.nFAS
                    self.min_FAS = self._FAS()
                    if file_record:
                        f_min_FAS.write("%f %d %s\n" % (T, self.min_nFAS, self.min_FAS))
                if file_record:
                    f_nFAS.write("%f %d %d\n" % (T, self.nFAS, self.min_nFAS))

                if debug:
                    FAS = self._FAS().tolist()
                    log_info(("picked_arc", "erand", "h", "h_of_v", "FAS-energies", "waitlist"),
                             (picked_arc, erand, self.h, self.h_of_v,
                              list(zip(FAS, [self._energy_change(self.simple_arcs[arc], floor0=True) for arc in FAS])),
                              waitlist))

                if debug:
                    check()
                    # print("check passed %d" % nmove)

                # Move child
                # 1. randomly pick an arc
                prob = np.exp(-np.arange(self.e_mvecld.nlist) / T) * self.e_mvecld.sublist_sizes()
                prob = prob / sum(prob)
                erand = np.random.choice(self.e_mvecld.nlist, p=prob)
                arc_ind = self.e_mvecld.get_rand(erand)
                p, c = self.simple_arcs[arc_ind]
                # 2. update all energy related fields:
                # energy
                time_mark = time.time()
                self.nFAS += self._energy_change(self.simple_arcs[arc_ind])[1]
                time_energy += time.time() - time_mark

                time_mark = time.time()
                waitlist = self._update_waitlist(arc_ind)
                for arc_ind in waitlist:
                    try:
                        self.e_mveprt.pop(arc_ind)
                        self.e_mvecld.pop(arc_ind)
                    except KeyError:
                        pass
                    except ValueError:
                        pass
                time_sublist += time.time() - time_mark

                # h,h_of_v
                hc, hp = self.h_of_v[c], self.h_of_v[p]
                self.h_of_v[self.h[hc + 1:hp + 1]] -= 1
                self.h_of_v[c] = hp
                self.h[hc:hp] = self.h[hc + 1:hp + 1]
                self.h[hp] = c

                # energy lists;

                for arc_ind in waitlist:
                    if self._is_FAS(self.simple_arcs[arc_ind]):
                        time_mark = time.time()
                        ep, ec = self._energy_change(self.simple_arcs[arc_ind], floor0=True)
                        time_energy += time.time() - time_mark
                        # if arc_ind == 0:
                        #     print("Pushing simple_arcs[0]. New: ep=%d, ec=%d")
                        time_mark = time.time()
                        self.e_mveprt.push(ep, arc_ind)
                        self.e_mvecld.push(ec, arc_ind)
                        time_sublist += time.time() - time_mark

                # check if FAS size reduced
                if self.nFAS < self.min_nFAS:
                    fail = False
                    nfail = 0
                    self.min_h = self.h.copy()
                    self.min_h_of_v = self.h_of_v.copy()
                    self.min_nFAS = self.nFAS
                    self.min_FAS = self._FAS()
                    if file_record:
                        f_min_FAS.write("%f %d %s\n" % (T, self.min_nFAS, self.min_FAS))
                if file_record:
                    f_nFAS.write("%f %d %d\n" % (T, self.nFAS, self.min_nFAS))

                if debug:
                    FAS = self._FAS().tolist()
                    log_info(("picked_arc", "erand", "h", "h_of_v", "FAS-energies", "waitlist"),
                             (picked_arc, erand, self.h, self.h_of_v,
                              list(zip(FAS, [self._energy_change(self.simple_arcs[arc], floor0=True) for arc in FAS])),
                              waitlist))
                if debug:
                    check()
                    # print("check passed %d" % nmove)
                    # print("Calculating, T=%.2f, i=%d" % (T, nmove))

            if fail:
                nfail += 1
            print("T=%.2f, nFAS=%d, min_nFAS=%d" % (T, self.nFAS, self.min_nFAS))
            T *= alpha

        self.simulated = True
        time_total = time.time() - begin_total
        time_other = time_total - time_sublist - time_energy

        # output info
        print("SA finished. T_0=%.3f, alpha=%.3f, max_move=%d, max_fail=%d." % (T_0, alpha, max_move, max_fail))
        print("T_f=%.3f" % T)
        print("min_nFAS=%d" % self.min_nFAS)
        print("time_total=%.4f, time_energy=%.4f, time_sublist=%.4f,time_other=%.4f" % (
            time_total, time_energy, time_sublist, time_other))

        if file_record:
            f_nFAS.close()
            f_min_FAS.close()
        return T

    def load_SA_result(self, dic, inspect=True):
        self.min_nFAS = int(dic['min_nFAS'])
        self.min_FAS = np.array(dic['min_FAS'], dtype=int)
        self.h = np.array(dic['min_h'], dtype=int)
        self.min_h = np.array(dic['min_h'], dtype=int)
        self.h_of_v = np.array(dic['min_h_of_v'], dtype=int)
        self.min_h_of_v = np.array(dic['min_h_of_v'], dtype=int)
        if inspect:
            print("SA result loaded.")
        self.simulated = True

    def visualize_coordinates(self, T=1, alpha=0.995, max_move=None, max_fail=50, inspect=True, debug=False):
        if max_move is None:
            max_move = 3 * self.nsimple
        if inspect:
            print("Obtaining visualization coordinates by SA...")

        dok = self.adjmtx.todok()

        nSCC, SCC_labels = sp.csgraph.connected_components(self.adjmtx, directed=True, connection='strong',
                                                           return_labels=True)
        x = np.argsort(SCC_labels)
        SCC_labels = SCC_labels[x]
        # randomize within SCCs
        for i in range(nSCC):
            x[SCC_labels == i] = np.random.permutation(x[SCC_labels == i])

        x_of_v = np.argsort(x)

        # only show simple arcs for now
        def energy():
            return np.sum((x_of_v[self.simple_arcs[:, 0]] - x_of_v[self.simple_arcs[:, 1]]) ** 2)

        def energy_change(v1, v2):
            v1_neighbors = self._all_neighbors_of(v1)
            v2_neighbors = self._all_neighbors_of(v2)

            old_e = np.sum((x_of_v[v1] - x_of_v[v1_neighbors]) ** 2) + np.sum((x_of_v[v2] - x_of_v[v2_neighbors]) ** 2)
            new_e = np.sum((x_of_v[v2] - x_of_v[v1_neighbors]) ** 2) + np.sum((x_of_v[v1] - x_of_v[v2_neighbors]) ** 2)
            if dok[v1, v2] != 0 or dok[v2, v1] != 0:
                # print("check")
                new_e += 2 * (x_of_v[v1] - x_of_v[v2]) ** 2

            return new_e - old_e

        nfail = 0
        e = energy()
        min_e = e
        min_x = np.copy(x)
        min_x_of_v = np.copy(x_of_v)

        def check():
            # check e
            if e != energy():
                raise ValueError("e recording wrong.")
            if not np.all(x[x_of_v] == np.arange(self.nvertex)) or not np.all(x_of_v[x] == np.arange(self.nvertex)):
                raise ValueError("x/x_of_v recording wrong.")
            if not np.all(min_x[min_x_of_v] == np.arange(self.nvertex)) or not np.all(
                            min_x_of_v[min_x] == np.arange(self.nvertex)):
                raise ValueError("x/x_of_v recording wrong.")

        while nfail < max_fail:
            fail = True
            for nmove in range(max_move):
                # calculate energies
                v1, v2 = np.random.choice(self.nvertex, 2, replace=False)
                e_change = energy_change(v1, v2)
                if e_change <= 0 or np.random.rand() < np.exp(-e_change / T):
                    # change x1,x2
                    # old_e = e
                    e += e_change
                    x1, x2 = x_of_v[v1], x_of_v[v2]
                    x[x1] = v2
                    x[x2] = v1
                    x_of_v[v1] = x2
                    x_of_v[v2] = x1
                    if e < min_e:
                        min_e = e
                        min_x = np.copy(x)
                        min_x_of_v = np.copy(x_of_v)
                        fail = False
                    if debug:
                        check()

            if inspect:
                print("T=%.2f, e=%d, min_e=%d" % (T, e, min_e))

            if fail:
                nfail += 1

            T *= alpha

        if inspect:
            print("x-coordinate SA finished.")
        simple_lines = np.zeros((self.nsimple, 2, 2), dtype=int)
        simple_lines[:, 0, 0] = min_x_of_v[self.simple_arcs[:, 0]]
        simple_lines[:, 0, 1] = self.min_h_of_v[self.simple_arcs[:, 0]]
        simple_lines[:, 1, 0] = min_x_of_v[self.simple_arcs[:, 1]]
        simple_lines[:, 1, 1] = self.min_h_of_v[self.simple_arcs[:, 1]]

        double_lines = np.zeros((self.ndouble, 2, 2), dtype=int)
        double_lines[:, 0, 0] = min_x_of_v[self.double_arcs[:, 0]]
        double_lines[:, 0, 1] = self.min_h_of_v[self.double_arcs[:, 0]]
        double_lines[:, 1, 0] = min_x_of_v[self.double_arcs[:, 1]]
        double_lines[:, 1, 1] = self.min_h_of_v[self.double_arcs[:, 1]]

        return min_x_of_v, self.min_h_of_v, simple_lines, double_lines

    def visualize(self, xs, ys, **kwargs):
        # given xs, ys, plot the whole thing
        # using visualization function in the other file
        pass

    def get_data(self, arcs=False):
        # return info:
        # nvertex, narc, nsimple, ndouble, nself;
        # (if arcs) simple_arcs, double_arcs, self_arcs
        # min_nFAS, min_FAS, min_h, min_h_of_v
        opt = {'nvertex': int(self.nvertex), 'narc': int(self.narc), 'nsimple': int(self.nsimple),
               'ndouble': int(self.ndouble), 'nself': int(self.nself),
               'min_nFAS': int(self.min_nFAS), "min_FAS": self.min_FAS.tolist(),
               'min_h': self.min_h.tolist(), 'min_h_of_v': self.min_h_of_v.tolist()}
        if arcs:
            opt['simple_arcs'] = self.simple_arcs.tolist()
            opt['double_arcs'] = self.double_arcs.tolist()
            opt['self_arcs'] = self.self_arcs.tolist()
        return opt

    def _energy_change(self, arc, floor0=False):
        """Energy change by moving arc (p, c). O(max_degree)
        Batch calculation should be further vectorized.

        Consider a vertex k connected to p or c,
            only when h(c)<h(k)<h(p), energy changes.
            Let v = h(c) < h < h(p)
        Moving parent: ............c..p.... -> ...........p c......
                when k->p, h(c)<h(k)<h(p), e++
                when p->k, h(c)<=h(k)<h(p), e--
                In matrix form:
                    e_mveprt = -A[p,:] @ v + v^T @ A[:,p] - 1
        Moving child: ...........c..p.... -> .............p c....
                when k->c, h(c)<h(k)<=h(p), e--
                when c->k, h(c)<h(k)<h(p), e++
                In matrix form:
                    e_mvecld = A[c,:] @ v - v^T @ A[:,c] - 1

        Return:
            None                ,if h(p) < h(c)
            e_mveprt, e_mvecld   ,else
        """
        p, c = arc
        hp, hc = self.h_of_v[p], self.h_of_v[c]
        if hp < hc:
            return None

        pp, pc = self._parents_of(p), self._children_of(p)
        hpp, hpc = self.h_of_v[pp], self.h_of_v[pc]
        cp, cc = self._parents_of(c), self._children_of(c)
        hcp, hcc = self.h_of_v[cp], self.h_of_v[cc]

        e_mveprt = np.count_nonzero(np.logical_and(hc < hpp, hpp < hp)) \
                   - np.count_nonzero(np.logical_and(hc <= hpc, hpc < hp))
        e_mvecld = -np.count_nonzero(np.logical_and(hc < hcp, hcp <= hp)) \
                   + np.count_nonzero(np.logical_and(hc < hcc, hcc < hp))

        if floor0:
            return max(0, e_mveprt), max(0, e_mvecld)
        else:
            return e_mveprt, e_mvecld

    def _energy_change_matrix(self, arc, floor0=False):
        """LOW EFFICIENCY. DON'T USE.
            Energy change by moving arc (p, c).
                Batch calculation should be further vectorized.

                Consider a vertex k connected to p or c,
                    only when h(c)<h(k)<h(p), energy changes.
                    Let v = h(c) < h < h(p)
                Moving parent: ............c..p.... -> ...........p c......
                        when k->p, h(c)<h(k)<h(p), e++
                        when p->k, h(c)<=h(k)<h(p), e--
                        In matrix form:
                            e_mveprt = -A[p,:] @ v + v^T @ A[:,p] - 1
                Moving child: ...........c..p.... -> .............p c....
                        when k->c, h(c)<h(k)<=h(p), e--
                        when c->k, h(c)<h(k)<h(p), e++
                        In matrix form:
                            e_mvecld = A[c,:] @ v - v^T @ A[:,c] - 1

                Return:
                    None                ,if h(p) < h(c)
                    e_mveprt, e_mvecld   ,else
                """

        p, c = arc
        if self.h_of_v[p] < self.h_of_v[c]:
            return None

        v = np.logical_and(self.h_of_v > self.h_of_v[c],
                           self.h_of_v < self.h_of_v[p])
        e_mveprt = int(-self.m_csr[p, :] @ v + v @ self.m_csc[:, p] - 1)
        e_mvecld = int(self.m_csr[c, :] @ v - v @ self.m_csc[:, c] - 1)
        if floor0:
            return max(0, e_mveprt), max(0, e_mvecld)
        else:
            return e_mveprt, e_mvecld

    def _energy_change_batch(self, arcs, floor0=False):
        pass

    def _test_energy_change_speed(self, arc, ntrial, floor0=False):
        t1 = 0
        t2 = 0
        t3 = 0
        for i in range(ntrial):
            begin = time.time()
            p, c = arc
            hp, hc = self.h_of_v[p], self.h_of_v[c]
            if hp < hc:
                return None
            t1 += time.time() - begin

            begin = time.time()
            pp, pc = self._parents_of(p), self._children_of(p)
            hpp, hpc = self.h_of_v[pp], self.h_of_v[pc]
            cp, cc = self._parents_of(c), self._children_of(c)
            hcp, hcc = self.h_of_v[cp], self.h_of_v[cc]
            t2 += time.time() - begin

            begin = time.time()
            e_mveprt = np.count_nonzero(np.logical_and(hc < hpp, hpp < hp)) \
                       - np.count_nonzero(np.logical_and(hc <= hpc, hpc < hp))
            e_mvecld = -np.count_nonzero(np.logical_and(hc < hcp, hcp <= hp)) \
                       + np.count_nonzero(np.logical_and(hc < hcc, hcc < hp))
            t3 += time.time() - begin
            # if floor0:
            #     return max(0, e_mveprt), max(0, e_mvecld)
            # else:
            #     return e_mveprt, e_mvecld
        print("Energy change speed: ntrial=%d, t1=%d, t2=%d, t3=%d"
              % (ntrial, t1, t2, t3))

    def _initialize_SA(self):
        """Pre-initialization of SA.
        - Iterate through simple_arcs, calculate initial energy.
        - Update energy lists.
        - Time: O(E)

        """
        e_mveprt, e_mvecld = {}, {}

        for i in range(self.max_in_degree + 1):
            e_mveprt[i] = set()
        for i in range(self.max_out_degree + 1):
            e_mvecld[i] = set()

        for i, arc_ind in enumerate(self._FAS()):
            e_prt, e_cld = self._energy_change(self.simple_arcs[arc_ind, :], floor0=True)
            e_mveprt[e_prt].add(arc_ind)
            e_mvecld[e_cld].add(arc_ind)

        self.e_mveprt = sublist_dic(self.nsimple, self.max_in_degree + 1, e_mveprt)
        self.e_mvecld = sublist_dic(self.nsimple, self.max_out_degree + 1, e_mvecld)

    def _out_arcs_of(self, p):
        return self.out_arcs_of[p]

    def _in_arcs_of(self, c):
        return self.in_arcs_of[c]

    def _all_arcs_of(self, v):
        return np.hstack((self._in_arcs_of(v), self._out_arcs_of(v))).astype(int)

    def _children_of(self, p):
        return self.children_of[p]

    def _parents_of(self, c):
        return self.parents_of[c]

    def _all_neighbors_of(self, v):
        return np.hstack((self._children_of(v), self._parents_of(v))).astype(int)

    def _count_FAS(self):
        """Expensive, O(E)."""
        return np.count_nonzero(self.h_of_v[self.simple_arcs[:, 0]]
                                > self.h_of_v[self.simple_arcs[:, 1]])

    def _FAS(self):
        """Expensive, O(E)."""
        return np.where(self.h_of_v[self.simple_arcs[:, 0]]
                        > self.h_of_v[self.simple_arcs[:, 1]])[0]

    def _is_FAS(self, arc):
        return self.h_of_v[arc[0]] > self.h_of_v[arc[1]]

    def _update_waitlist(self, arc_ind):
        """Find arcs to be updated after moving arc.

        Current version:
        1. 2nd order 
        2. go across p or c"""

        arc = self.simple_arcs[arc_ind]
        p, c = arc
        p1 = self._all_arcs_of(p)
        c1 = self._all_arcs_of(c)
        p2 = np.hstack([self._all_arcs_of(n) for n in self._all_neighbors_of(p)])
        c2 = np.hstack([self._all_arcs_of(n) for n in self._all_neighbors_of(c)])

        opt = np.unique(np.hstack((p2, c2)))

        arcs = self.simple_arcs[opt]
        conditions = np.zeros(opt.shape, dtype=bool)
        conditions[np.logical_and(self.h_of_v[arcs[:, 0]] >= self.h_of_v[c],
                                  self.h_of_v[c] >= self.h_of_v[arcs[:, 1]])] = True

        conditions[np.logical_and(self.h_of_v[arcs[:, 0]] >= self.h_of_v[p],
                                  self.h_of_v[p] >= self.h_of_v[arcs[:, 1]])] = True
        opt = np.unique(np.hstack((opt[conditions], p1, c1)))
        # opt=np.unique(np.hstack((opt,p1,c1,arc_ind)))
        # conditions[opt == arc_ind] = True

        return opt


class FASbySAIO(FASbySA):
    def __init__(self):
        super().__init__()
        self.inputs, self.outputs = set(), set()
        self.subgraph: FASbySA = None
        self._ind_in_whole_graph = None
        self.initialized_io = False

    def set_io(self, inputs=None, outputs=None):
        if not self.initialized:
            raise ValueError("Graph not initialized.")
        inputs = set() if inputs is None else set(inputs)
        outputs = set() if outputs is None else set(outputs)

        self.inputs = inputs.difference(outputs)
        self.outputs = outputs.difference(inputs)

        # set subgraph
        sub_arcs = np.array([edge for edge in self.all_arcs if
                             (edge[0] not in self.inputs and edge[0] not in self.outputs
                              and edge[1] not in self.inputs and edge[1] not in self.outputs)])

        sub_vertices = np.unique(sub_arcs.flatten())
        sub_nvertex = len(sub_vertices)
        temp = -np.ones(self.nvertex, dtype=int)
        temp[sub_vertices] = np.arange(sub_nvertex)
        sub_arcs_subind = temp[sub_arcs]
        self._ind_in_whole_graph = sub_vertices
        self.subgraph = FASbySA()
        self.subgraph.initialize(sub_nvertex, len(sub_arcs), sub_arcs_subind, inspect=False)

        self.initialized_io = True
        self.simulated = False

    def simulated_annealing_withio(self, **kwargs):
        if not self.initialized_io:
            raise ValueError("IO not set before SA.")

        T = self.subgraph.simulated_annealing(**kwargs)
        self.h = np.concatenate((list(self.inputs), self._ind_in_whole_graph[self.subgraph.h], list(self.outputs)))
        self.h_of_v = np.argsort(self.h)
        self.min_h = self.h.copy()
        self.min_h_of_v = self.h_of_v.copy()
        self.min_FAS = self._FAS()
        self.min_nFAS = len(self.min_FAS)

        return T


def _check_FAS(simple_arcs, h_of_v, FAS):
    for arc in simple_arcs:
        p, c = arc
        if h_of_v[p] >= h_of_v[c] and arc not in FAS:
            return False
    return True


def _tst_SA(debug=False):
    # logging.basicConfig(filename="test_FASbySA_dic.log", filemode='w', level=logging.DEBUG)
    # logging.debug("logging test")
    # test initializer
    fas = FASbySA()
    fas.initialize_by_file("celegans.g0", inspect=True)
    f_simple, f_double = open("celegans_170713_simple_arcs.txt", 'w+'), open("celegans_170713_double_arcs.txt", 'w+')
    for i, (p, c) in enumerate(fas.simple_arcs):
        f_simple.write("%d %d %d\n" % (i, p, c))
    for i, (p, c) in enumerate(fas.double_arcs):
        f_double.write("%d %d %d\n" % (i, p, c))
    f_simple.close()
    f_double.close()
    # test SA
    T, alpha, max_move, max_fail = 1, 0.99, 5 * fas.nvertex, 50
    fas.simulated_annealing(T, alpha, max_move, max_fail, debug=debug,
                            file_record=True,
                            fname_nFAS="celegans_170713_nFAS.txt",
                            fname_min_FAS="celegans_170713_min_FAS.txt")


if __name__ == '__main__':
    pass
