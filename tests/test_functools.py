import abc
import builtins
import collections
import collections.abc
import copy
from itertools import permutations
import pickle
from random import choice
import re
import sys
from test import support
import threading
import time
import typing
import unittest
import unittest.mock
import weakref
import gc
from weakref import proxy
import contextlib
from inspect import Signature

from test.support import import_helper
from test.support import threading_helper

import functools

py_functools = import_helper.import_fresh_module("functools",
                                                 blocked=["_functools"])
c_functools = import_helper.import_fresh_module("functools",
                                                fresh=["_functools"])

decimal = import_helper.import_fresh_module("decimal", fresh=["_decimal"])


class TestSingleDispatch(unittest.TestCase):

    def test_pep585_basic(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        def g_list_int(li):
            return "list of ints"
        # previously this failed with: 'not a class'
        g.register(list[int], g_list_int)
        self.assertEqual(g([1]), "list of ints")

    def test_pep585_annotation(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        # previously this failed with: 'not a class'
        @g.register
        def g_list_int(li: list[int]):
            return "list of ints"
        self.assertEqual(g([1,2,3]), "list of ints")

    def test_pep585_all_must_match(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        def g_list_int(li):
            return "list of ints"
        def g_list_not_ints(l):
            # should only trigger if list doesnt match `list[int]`
            # ie. at least one element is not an int
            return "!all(int)"

        g.register(list[int], g_list_int)
        g.register(list, g_list_not_ints)

        self.assertEqual(g([1,2,3]), "list of ints")
        self.assertEqual(g([1,2,3, "hello"]), "!all(int)")
        self.assertEqual(g([3.14]), "!all(int)")

    def test_pep585_specificity(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        @g.register
        def g_list(l: list):
            return "basic list"
        @g.register
        def g_list_int(li: list[int]):
            return "int"
        @g.register
        def g_list_str(ls: list[str]):
            return "str"
        @g.register
        def g_list_mixed_int_str(lmis:list[int|str]):
            return "int|str"
        @g.register
        def g_list_mixed_int_float(lmif: list[int|float]):
            return "int|float"
        @g.register
        def g_list_mixed_int_float_str(lmifs: list[int|float|str]):
            return "int|float|str"

        # this matches list, list[int], list[int|str], list[int|float|str], list[int|...|...|...|...]
        # but list[int] is the most specific, so that is correct
        self.assertEqual(g([1,2,3]), "int")

        # this cannot match list[int] because of the string
        # it does match list[int|float|str] but this is incorrect because,
        # the most specific is list[int|str]
        self.assertEqual(g([1,2,3, "hello"]), "int|str")

        # list[float] is not mapped so,
        # list[int|float] is the most specific
        self.assertEqual(g([3.14]), "int|float")

    def test_pep585_ambiguous(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        @g.register
        def g_list_int_float(l: list[int|float]):
            return "int|float"
        @g.register
        def g_list_int_str(l: list[int|str]):
            return "int|str"

        self.assertEqual(g([3.1]), "int|float") # floats only
        self.assertEqual(g(["hello"]), "int|str") # strings only
        self.assertEqual(g([3.14, 1]), "int|float") # ints and floats
        self.assertEqual(g(["hello", 1]), "int|str") # ints and strings

        # these should fail because it's unclear which target is "correct"
        with self.assertRaisesRegex(RuntimeError, "Ambiguous dispatch:"):
            g([1])

    def test_pep585_method_basic(self):
        class A:
            @functools.singledispatchmethod
            def g(self, obj):
                return "base"
            def g_list_int(self, li):
                return "list of ints"

        a = A()
        a.g.register(list[int], A.g_list_int)
        self.assertEqual(a.g([1]), "list of ints")

    def test_pep585_method_annotation(self):
        class A:
            @functools.singledispatchmethod
            def g(self, obj):
                return "base"
            # previously this failed with: 'not a class'
            @g.register
            def g_list_int(self, li: list[int]):
                return "list of ints"
        a = A()
        #self.assertEqual(a.g([1,2,3]), "list of ints")
        self.assertEqual(a.g([1,2,3]), "list of ints")

    def test_pep585_method_all_must_match(self):
        class A:
            @functools.singledispatchmethod
            def g(self, obj):
                return "base"
            def g_list_int(self, li):
                return "list of ints"
            def g_list_not_ints(self, l):
                # should only trigger if list doesnt match `list[int]`
                # ie. at least one element is not an int
                return "!all(int)"

        a = A()
        a.g.register(list[int], A.g_list_int)
        a.g.register(list, A.g_list_not_ints)

        self.assertEqual(a.g([1,2,3]), "list of ints")
        self.assertEqual(a.g([1,2,3, "hello"]), "!all(int)")
        self.assertEqual(a.g([3.14]), "!all(int)")

    def test_pep585_method_specificity(self):
        class A:
            @functools.singledispatchmethod
            def g(self, obj):
                return "base"
            @g.register
            def g_list(self, l: list):
                return "basic list"
            @g.register
            def g_list_int(self, li: list[int]):
                return "int"
            @g.register
            def g_list_str(self, ls: list[str]):
                return "str"
            @g.register
            def g_list_mixed_int_str(self, lmis:list[int|str]):
                return "int|str"
            @g.register
            def g_list_mixed_int_float(self, lmif: list[int|float]):
                return "int|float"
            @g.register
            def g_list_mixed_int_float_str(self, lmifs: list[int|float|str]):
                return "int|float|str"

        a = A()

        # this matches list, list[int], list[int|str], list[int|float|str], list[int|...|...|...|...]
        # but list[int] is the most specific, so that is correct
        self.assertEqual(a.g([1,2,3]), "int")

        # this cannot match list[int] because of the string
        # it does match list[int|float|str] but this is incorrect because,
        # the most specific is list[int|str]
        self.assertEqual(a.g([1,2,3, "hello"]), "int|str")

        # list[float] is not mapped so,
        # list[int|float] is the most specific
        self.assertEqual(a.g([3.14]), "int|float")


    def test_pep585_method_ambiguous(self):
        class A:
            @functools.singledispatchmethod
            def g(self, obj):
                return "base"
            @g.register
            def g_list_int_float(self, l: list[int|float]):
                return "int|float"
            @g.register
            def g_list_int_str(self, l: list[int|str]):
                return "int|str"
            @g.register
            def g_list_int(self, l: list[int]):
                return "int only"

        a = A()

        self.assertEqual(a.g([3.1]), "int|float") # floats only
        self.assertEqual(a.g(["hello"]), "int|str") # strings only
        self.assertEqual(a.g([3.14, 1]), "int|float") # ints and floats
        self.assertEqual(a.g(["hello", 1]), "int|str") # ints and strings

        # these should fail because it's unclear which target is "correct"
        #self.assertRaises(RuntimeError, a.g([1]))

        #self.assertRaises(RuntimeError, a.g.dispatch(list[int]))

    def test_simple_overloads(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        def g_int(i):
            return "integer"
        g.register(int, g_int)
        self.assertEqual(g("str"), "base")
        self.assertEqual(g(1), "integer")
        self.assertEqual(g([1,2,3]), "base")

    def test_mro(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        class A:
            pass
        class C(A):
            pass
        class B(A):
            pass
        class D(C, B):
            pass
        def g_A(a):
            return "A"
        def g_B(b):
            return "B"
        g.register(A, g_A)
        g.register(B, g_B)
        self.assertEqual(g(A()), "A")
        self.assertEqual(g(B()), "B")
        self.assertEqual(g(C()), "A")
        self.assertEqual(g(D()), "B")

    def test_register_decorator(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        @g.register(int)
        def g_int(i):
            return "int %s" % (i,)
        self.assertEqual(g(""), "base")
        self.assertEqual(g(12), "int 12")
        self.assertIs(g.dispatch(int), g_int)
        #self.assertIs(g.dispatch(object), g.dispatch(str))
        # Note: in the assert above this is not g.
        # @singledispatch returns the wrapper.

    def test_wrapping_attributes(self):
        @functools.singledispatch
        def g(obj):
            "Simple test"
            return "Test"
        self.assertEqual(g.__name__, "g")
        if sys.flags.optimize < 2:
            self.assertEqual(g.__doc__, "Simple test")

    @unittest.skipUnless(decimal, "requires _decimal")
    @support.cpython_only
    def test_c_classes(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        @g.register(decimal.DecimalException)
        def _(obj):
            return obj.args
        subn = decimal.Subnormal("Exponent < Emin")
        rnd = decimal.Rounded("Number got rounded")
        self.assertEqual(g(subn), ("Exponent < Emin",))
        self.assertEqual(g(rnd), ("Number got rounded",))
        @g.register(decimal.Subnormal)
        def _(obj):
            return "Too small to care."
        self.assertEqual(g(subn), "Too small to care.")
        self.assertEqual(g(rnd), ("Number got rounded",))

    def test_compose_mro(self):
        # None of the examples in this test depend on haystack ordering.
        c = collections.abc
        mro = functools._compose_mro
        bases = [c.Sequence, c.MutableMapping, c.Mapping, c.Set]
        for haystack in permutations(bases):
            m = mro(dict, haystack)
            self.assertEqual(m, [dict, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])
        bases = [c.Container, c.Mapping, c.MutableMapping, collections.OrderedDict]
        for haystack in permutations(bases):
            m = mro(collections.ChainMap, haystack)
            self.assertEqual(m, [collections.ChainMap, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])

        # If there's a generic function with implementations registered for
        # both Sized and Container, passing a defaultdict to it results in an
        # ambiguous dispatch which will cause a RuntimeError (see
        # test_mro_conflicts).
        bases = [c.Container, c.Sized, str]
        for haystack in permutations(bases):
            m = mro(collections.defaultdict, [c.Sized, c.Container, str])
            self.assertEqual(m, [collections.defaultdict, dict, c.Sized,
                                 c.Container, object])

        # MutableSequence below is registered directly on D. In other words, it
        # precedes MutableMapping which means single dispatch will always
        # choose MutableSequence here.
        class D(collections.defaultdict):
            pass
        c.MutableSequence.register(D)
        bases = [c.MutableSequence, c.MutableMapping]
        for haystack in permutations(bases):
            m = mro(D, haystack)
            self.assertEqual(m, [D, c.MutableSequence, c.Sequence, c.Reversible,
                                 collections.defaultdict, dict, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable, c.Container,
                                 object])

        # Container and Callable are registered on different base classes and
        # a generic function supporting both should always pick the Callable
        # implementation if a C instance is passed.
        class C(collections.defaultdict):
            def __call__(self):
                pass
        bases = [c.Sized, c.Callable, c.Container, c.Mapping]
        for haystack in permutations(bases):
            m = mro(C, haystack)
            self.assertEqual(m, [C, c.Callable, collections.defaultdict, dict, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])

    def test_register_abc(self):
        c = collections.abc
        d = {"a": "b"}
        l = [1, 2, 3]
        s = {object(), None}
        f = frozenset(s)
        t = (1, 2, 3)
        @functools.singledispatch
        def g(obj):
            return "base"
        self.assertEqual(g(d), "base")
        self.assertEqual(g(l), "base")
        self.assertEqual(g(s), "base")
        self.assertEqual(g(f), "base")
        self.assertEqual(g(t), "base")
        g.register(c.Sized, lambda obj: "sized")
        self.assertEqual(g(d), "sized")
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableMapping, lambda obj: "mutablemapping")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(collections.ChainMap, lambda obj: "chainmap")
        self.assertEqual(g(d), "mutablemapping")  # irrelevant ABCs registered
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableSequence, lambda obj: "mutablesequence")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableSet, lambda obj: "mutableset")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.Mapping, lambda obj: "mapping")
        self.assertEqual(g(d), "mutablemapping")  # not specific enough
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.Sequence, lambda obj: "sequence")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sequence")
        g.register(c.Set, lambda obj: "set")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(dict, lambda obj: "dict")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(list, lambda obj: "list")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(set, lambda obj: "concrete-set")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(frozenset, lambda obj: "frozen-set")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "frozen-set")
        self.assertEqual(g(t), "sequence")
        g.register(tuple, lambda obj: "tuple")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "frozen-set")
        self.assertEqual(g(t), "tuple")

    def test_c3_abc(self):
        c = collections.abc
        mro = functools._c3_mro
        class A(object):
            pass
        class B(A):
            def __len__(self):
                return 0   # implies Sized
        @c.Container.register
        class C(object):
            pass
        class D(object):
            pass   # unrelated
        class X(D, C, B):
            def __call__(self):
                pass   # implies Callable
        expected = [X, c.Callable, D, C, c.Container, B, c.Sized, A, object]
        for abcs in permutations([c.Sized, c.Callable, c.Container]):
            self.assertEqual(mro(X, abcs=abcs), expected)
        # unrelated ABCs don't appear in the resulting MRO
        many_abcs = [c.Mapping, c.Sized, c.Callable, c.Container, c.Iterable]
        self.assertEqual(mro(X, abcs=many_abcs), expected)

    def test_false_meta(self):
        # see issue23572
        class MetaA(type):
            def __len__(self):
                return 0
        class A(metaclass=MetaA):
            pass
        class AA(A):
            pass
        @functools.singledispatch
        def fun(a):
            return "base A"
        @fun.register(A)
        def _(a):
            return "fun A"
        aa = AA()
        self.assertEqual(fun(aa), "fun A")

    def test_mro_conflicts(self):
        c = collections.abc
        @functools.singledispatch
        def g(arg):
            return "base"
        class O(c.Sized):
            def __len__(self):
                return 0
        o = O()
        self.assertEqual(g(o), "base")
        g.register(c.Iterable, lambda arg: "iterable")
        g.register(c.Container, lambda arg: "container")
        g.register(c.Sized, lambda arg: "sized")
        g.register(c.Set, lambda arg: "set")
        self.assertEqual(g(o), "sized")
        c.Iterable.register(O)
        self.assertEqual(g(o), "sized")   # because it's explicitly in __mro__
        c.Container.register(O)
        self.assertEqual(g(o), "sized")   # see above: Sized is in __mro__
        c.Set.register(O)
        self.assertEqual(g(o), "set")     # because c.Set is a subclass of
                                          # c.Sized and c.Container
        class P:
            pass
        p = P()
        self.assertEqual(g(p), "base")
        c.Iterable.register(P)
        self.assertEqual(g(p), "iterable")
        c.Container.register(P)
        with self.assertRaises(RuntimeError) as re_one:
            g(p)
        self.assertIn(
            str(re_one.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Iterable'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Iterable'> "
              "or <class 'collections.abc.Container'>")),
        )
        class Q(c.Sized):
            def __len__(self):
                return 0
        q = Q()
        self.assertEqual(g(q), "sized")
        c.Iterable.register(Q)
        self.assertEqual(g(q), "sized")   # because it's explicitly in __mro__
        c.Set.register(Q)
        self.assertEqual(g(q), "set")     # because c.Set is a subclass of
                                          # c.Sized and c.Iterable
        @functools.singledispatch
        def h(arg):
            return "base"
        @h.register(c.Sized)
        def _(arg):
            return "sized"
        @h.register(c.Container)
        def _(arg):
            return "container"
        # Even though Sized and Container are explicit bases of MutableMapping,
        # this ABC is implicitly registered on defaultdict which makes all of
        # MutableMapping's bases implicit as well from defaultdict's
        # perspective.
        with self.assertRaises(RuntimeError) as re_two:
            h(collections.defaultdict(lambda: 0))
        self.assertIn(
            str(re_two.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Sized'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Sized'> "
              "or <class 'collections.abc.Container'>")),
        )
        class R(collections.defaultdict):
            pass
        c.MutableSequence.register(R)
        @functools.singledispatch
        def i(arg):
            return "base"
        @i.register(c.MutableMapping)
        def _(arg):
            return "mapping"
        @i.register(c.MutableSequence)
        def _(arg):
            return "sequence"
        r = R()
        self.assertEqual(i(r), "sequence")
        class S:
            pass
        class T(S, c.Sized):
            def __len__(self):
                return 0
        t = T()
        self.assertEqual(h(t), "sized")
        c.Container.register(T)
        self.assertEqual(h(t), "sized")   # because it's explicitly in the MRO
        class U:
            def __len__(self):
                return 0
        u = U()
        self.assertEqual(h(u), "sized")   # implicit Sized subclass inferred
                                          # from the existence of __len__()
        c.Container.register(U)
        # There is no preference for registered versus inferred ABCs.
        with self.assertRaises(RuntimeError) as re_three:
            h(u)
        self.assertIn(
            str(re_three.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Sized'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Sized'> "
              "or <class 'collections.abc.Container'>")),
        )
        class V(c.Sized, S):
            def __len__(self):
                return 0
        @functools.singledispatch
        def j(arg):
            return "base"
        @j.register(S)
        def _(arg):
            return "s"
        @j.register(c.Container)
        def _(arg):
            return "container"
        v = V()
        self.assertEqual(j(v), "s")
        c.Container.register(V)
        self.assertEqual(j(v), "container")   # because it ends up right after
                                              # Sized in the MRO

    def test_cache_invalidation(self):
        from collections import UserDict
        import weakref

        class TracingDict(UserDict):
            def __init__(self, *args, **kwargs):
                super(TracingDict, self).__init__(*args, **kwargs)
                self.set_ops = []
                self.get_ops = []
            def __getitem__(self, key):
                result = self.data[key]
                self.get_ops.append(key)
                return result
            def __setitem__(self, key, value):
                self.set_ops.append(key)
                self.data[key] = value
            def clear(self):
                self.data.clear()

        td = TracingDict()
        with support.swap_attr(weakref, "WeakKeyDictionary", lambda: td):
            c = collections.abc
            @functools.singledispatch
            def g(arg):
                return "base"
            d = {}
            l = []
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "base")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [])
            self.assertEqual(td.set_ops, [dict])
            self.assertEqual(td.data[dict], g.registry[object])
            self.assertEqual(g(l), "base")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [])
            self.assertEqual(td.set_ops, [dict, list])
            self.assertEqual(td.data[dict], g.registry[object])
            self.assertEqual(td.data[list], g.registry[object])
            self.assertEqual(td.data[dict], td.data[list])
            self.assertEqual(g(l), "base")
            self.assertEqual(g(d), "base")
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list])
            g.register(list, lambda arg: "list")
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "base")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict])
            self.assertEqual(td.data[dict],
                             functools._find_impl(dict, g.registry))
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list])
            self.assertEqual(td.data[list],
                             functools._find_impl(list, g.registry))
            class X:
                pass
            c.MutableMapping.register(X)   # Will not invalidate the cache,
                                           # not using ABCs yet.
            self.assertEqual(g(d), "base")
            self.assertEqual(g(l), "list")
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list])
            g.register(c.Sized, lambda arg: "sized")
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "sized")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict])
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            self.assertEqual(g(l), "list")
            self.assertEqual(g(d), "sized")
            self.assertEqual(td.get_ops, [list, dict, dict, list, list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            g.dispatch(list)
            g.dispatch(dict)
            self.assertEqual(td.get_ops, [list, dict, dict, list, list, dict,
                                          list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            c.MutableSet.register(X)       # Will invalidate the cache.
            self.assertEqual(len(td), 2)   # Stale cache.
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 1)
            g.register(c.MutableMapping, lambda arg: "mutablemapping")
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "mutablemapping")
            self.assertEqual(len(td), 1)
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            g.register(dict, lambda arg: "dict")
            self.assertEqual(g(d), "dict")
            self.assertEqual(g(l), "list")
            g._clear_cache()
            self.assertEqual(len(td), 0)

    def test_annotations(self):
        @functools.singledispatch
        def i(arg):
            return "base"
        @i.register
        def _(arg: collections.abc.Mapping):
            return "mapping"
        @i.register
        def _(arg: "collections.abc.Sequence"):
            return "sequence"
        self.assertEqual(i(None), "base")
        self.assertEqual(i({"a": 1}), "mapping")
        self.assertEqual(i([1, 2, 3]), "sequence")
        self.assertEqual(i((1, 2, 3)), "sequence")
        self.assertEqual(i("str"), "sequence")

        # Registering classes as callables doesn't work with annotations,
        # you need to pass the type explicitly.
        @i.register(str)
        class _:
            def __init__(self, arg):
                self.arg = arg

            def __eq__(self, other):
                return self.arg == other
        self.assertEqual(i("str"), "str")

    def test_method_register(self):
        class A:
            @functools.singledispatchmethod
            def t(self, arg):
                self.arg = "base"
            @t.register(int)
            def _(self, arg):
                self.arg = "int"
            @t.register(str)
            def _(self, arg):
                self.arg = "str"
        a = A()

        a.t(0)
        self.assertEqual(a.arg, "int")
        aa = A()
        self.assertNotHasAttr(aa, "arg")
        a.t("")
        self.assertEqual(a.arg, "str")
        aa = A()
        self.assertNotHasAttr(aa, "arg")
        a.t(0.0)
        self.assertEqual(a.arg, "base")
        aa = A()
        self.assertNotHasAttr(aa, "arg")

    def test_staticmethod_register(self):
        class A:
            @functools.singledispatchmethod
            @staticmethod
            def t(arg):
                return arg
            @t.register(int)
            @staticmethod
            def _(arg):
                return isinstance(arg, int)
            @t.register(str)
            @staticmethod
            def _(arg):
                return isinstance(arg, str)
        a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(""))
        self.assertEqual(A.t(0.0), 0.0)

    def test_slotted_class(self):
        class Slot:
            __slots__ = ("a", "b")
            @functools.singledispatchmethod
            def go(self, item, arg):
                pass

            @go.register
            def _(self, item: int, arg):
                return item + arg

        s = Slot()
        self.assertEqual(s.go(1, 1), 2)

    def test_classmethod_slotted_class(self):
        class Slot:
            __slots__ = ("a", "b")
            @functools.singledispatchmethod
            @classmethod
            def go(cls, item, arg):
                pass

            @go.register
            @classmethod
            def _(cls, item: int, arg):
                return item + arg

        s = Slot()
        self.assertEqual(s.go(1, 1), 2)
        self.assertEqual(Slot.go(1, 1), 2)

    def test_staticmethod_slotted_class(self):
        class A:
            __slots__ = ["a"]
            @functools.singledispatchmethod
            @staticmethod
            def t(arg):
                return arg
            @t.register(int)
            @staticmethod
            def _(arg):
                return isinstance(arg, int)
            @t.register(str)
            @staticmethod
            def _(arg):
                return isinstance(arg, str)
        a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(""))
        self.assertEqual(A.t(0.0), 0.0)
        self.assertTrue(a.t(0))
        self.assertTrue(a.t(""))
        self.assertEqual(a.t(0.0), 0.0)

    def test_assignment_behavior(self):
        # see gh-106448
        class A:
            @functools.singledispatchmethod
            def t(arg):
                return arg

        a = A()
        a.t.foo = "bar"
        a2 = A()
        with self.assertRaises(AttributeError):
            a2.t.foo

    def test_classmethod_register(self):
        class A:
            def __init__(self, arg):
                self.arg = arg

            @functools.singledispatchmethod
            @classmethod
            def t(cls, arg):
                return cls("base")
            @t.register(int)
            @classmethod
            def _(cls, arg):
                return cls("int")
            @t.register(str)
            @classmethod
            def _(cls, arg):
                return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t("").arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_callable_register(self):
        class A:
            def __init__(self, arg):
                self.arg = arg

            @functools.singledispatchmethod
            @classmethod
            def t(cls, arg):
                return cls("base")

        @A.t.register(int)
        @classmethod
        def _(cls, arg):
            return cls("int")
        @A.t.register(str)
        @classmethod
        def _(cls, arg):
            return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t("").arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_abstractmethod_register(self):
        class Abstract(metaclass=abc.ABCMeta):

            @functools.singledispatchmethod
            @abc.abstractmethod
            def add(self, x, y):
                pass

        self.assertTrue(Abstract.add.__isabstractmethod__)
        self.assertTrue(Abstract.__dict__["add"].__isabstractmethod__)

        with self.assertRaises(TypeError):
            Abstract()

    def test_type_ann_register(self):
        class A:
            @functools.singledispatchmethod
            def t(self, arg):
                return "base"
            @t.register
            def _(self, arg: int):
                return "int"
            @t.register
            def _(self, arg: str):
                return "str"
        a = A()

        self.assertEqual(a.t(0), "int")
        self.assertEqual(a.t(""), "str")
        self.assertEqual(a.t(0.0), "base")

    def test_staticmethod_type_ann_register(self):
        class A:
            @functools.singledispatchmethod
            @staticmethod
            def t(arg):
                return arg
            @t.register
            @staticmethod
            def _(arg: int):
                return isinstance(arg, int)
            @t.register
            @staticmethod
            def _(arg: str):
                return isinstance(arg, str)
        a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(""))
        self.assertEqual(A.t(0.0), 0.0)

    def test_classmethod_type_ann_register(self):
        class A:
            def __init__(self, arg):
                self.arg = arg

            @functools.singledispatchmethod
            @classmethod
            def t(cls, arg):
                return cls("base")
            @t.register
            @classmethod
            def _(cls, arg: int):
                return cls("int")
            @t.register
            @classmethod
            def _(cls, arg: str):
                return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t("").arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_method_wrapping_attributes(self):
        class A:
            @functools.singledispatchmethod
            def func(self, arg: int) -> str:
                """My function docstring"""
                return str(arg)
            @functools.singledispatchmethod
            @classmethod
            def cls_func(cls, arg: int) -> str:
                """My function docstring"""
                return str(arg)
            @functools.singledispatchmethod
            @staticmethod
            def static_func(arg: int) -> str:
                """My function docstring"""
                return str(arg)

        prefix = A.__qualname__ + "."
        for meth in (
            A.func,
            A().func,
            A.cls_func,
            A().cls_func,
            A.static_func,
            A().static_func
        ):
            with self.subTest(meth=meth):
                self.assertEqual(meth.__module__, __name__)
                self.assertEqual(type(meth).__module__, "functools")
                self.assertEqual(meth.__qualname__, prefix + meth.__name__)
                self.assertEqual(meth.__doc__,
                                 ("My function docstring"
                                  if support.HAVE_DOCSTRINGS
                                  else None))
                self.assertEqual(meth.__annotations__["arg"], int)

        self.assertEqual(A.func.__name__, "func")
        self.assertEqual(A().func.__name__, "func")
        self.assertEqual(A.cls_func.__name__, "cls_func")
        self.assertEqual(A().cls_func.__name__, "cls_func")
        self.assertEqual(A.static_func.__name__, "static_func")
        self.assertEqual(A().static_func.__name__, "static_func")

    def test_method_repr(self):
        class Callable:
            def __call__(self, *args):
                pass

        class CallableWithName:
            __name__ = "NOQUALNAME"
            def __call__(self, *args):
                pass

        class A:
            @functools.singledispatchmethod
            def func(self, arg):
                pass
            @functools.singledispatchmethod
            @classmethod
            def cls_func(cls, arg):
                pass
            @functools.singledispatchmethod
            @staticmethod
            def static_func(arg):
                pass
            # No __qualname__, only __name__
            no_qualname = functools.singledispatchmethod(CallableWithName())
            # No __qualname__, no __name__
            no_name = functools.singledispatchmethod(Callable())

        self.assertEqual(repr(A.__dict__["func"]),
            f"<single dispatch method descriptor {A.__qualname__}.func>")
        self.assertEqual(repr(A.__dict__["cls_func"]),
            f"<single dispatch method descriptor {A.__qualname__}.cls_func>")
        self.assertEqual(repr(A.__dict__["static_func"]),
            f"<single dispatch method descriptor {A.__qualname__}.static_func>")
        self.assertEqual(repr(A.__dict__["no_qualname"]),
            "<single dispatch method descriptor NOQUALNAME>")
        self.assertEqual(repr(A.__dict__["no_name"]),
            "<single dispatch method descriptor ?>")

        self.assertEqual(repr(A.func),
            f"<single dispatch method {A.__qualname__}.func>")
        self.assertEqual(repr(A.cls_func),
            f"<single dispatch method {A.__qualname__}.cls_func>")
        self.assertEqual(repr(A.static_func),
            f"<single dispatch method {A.__qualname__}.static_func>")
        self.assertEqual(repr(A.no_qualname),
            "<single dispatch method NOQUALNAME>")
        self.assertEqual(repr(A.no_name),
            "<single dispatch method ?>")

        a = A()
        self.assertEqual(repr(a.func),
            f"<bound single dispatch method {A.__qualname__}.func of {a!r}>")
        self.assertEqual(repr(a.cls_func),
            f"<bound single dispatch method {A.__qualname__}.cls_func of {a!r}>")
        self.assertEqual(repr(a.static_func),
            f"<bound single dispatch method {A.__qualname__}.static_func of {a!r}>")
        self.assertEqual(repr(a.no_qualname),
            f"<bound single dispatch method NOQUALNAME of {a!r}>")
        self.assertEqual(repr(a.no_name),
            f"<bound single dispatch method ? of {a!r}>")

    def test_double_wrapped_methods(self):
        def classmethod_friendly_decorator(func):
            wrapped = func.__func__
            @classmethod
            @functools.wraps(wrapped)
            def wrapper(*args, **kwargs):
                return wrapped(*args, **kwargs)
            return wrapper

        class WithoutSingleDispatch:
            @classmethod
            @contextlib.contextmanager
            def cls_context_manager(cls, arg: int) -> str:
                try:
                    yield str(arg)
                finally:
                    pass
                return "Done"

            @classmethod_friendly_decorator
            @classmethod
            def decorated_classmethod(cls, arg: int) -> str:
                return str(arg)

        class WithSingleDispatch:
            @functools.singledispatchmethod
            @classmethod
            @contextlib.contextmanager
            def cls_context_manager(cls, arg: int) -> str:
                """My function docstring"""
                try:
                    yield str(arg)
                finally:
                    pass
                return "Done"

            @functools.singledispatchmethod
            @classmethod_friendly_decorator
            @classmethod
            def decorated_classmethod(cls, arg: int) -> str:
                """My function docstring"""
                return str(arg)

        # These are sanity checks
        # to test the test itself is working as expected
        with WithoutSingleDispatch.cls_context_manager(5) as foo:
            without_single_dispatch_foo = foo

        with WithSingleDispatch.cls_context_manager(5) as foo:
            single_dispatch_foo = foo

        self.assertEqual(without_single_dispatch_foo, single_dispatch_foo)
        self.assertEqual(single_dispatch_foo, "5")

        self.assertEqual(
            WithoutSingleDispatch.decorated_classmethod(5),
            WithSingleDispatch.decorated_classmethod(5)
        )

        self.assertEqual(WithSingleDispatch.decorated_classmethod(5), "5")

        # Behavioural checks now follow
        for method_name in ("cls_context_manager", "decorated_classmethod"):
            with self.subTest(method=method_name):
                self.assertEqual(
                    getattr(WithSingleDispatch, method_name).__name__,
                    getattr(WithoutSingleDispatch, method_name).__name__
                )

                self.assertEqual(
                    getattr(WithSingleDispatch(), method_name).__name__,
                    getattr(WithoutSingleDispatch(), method_name).__name__
                )

        for meth in (
            WithSingleDispatch.cls_context_manager,
            WithSingleDispatch().cls_context_manager,
            WithSingleDispatch.decorated_classmethod,
            WithSingleDispatch().decorated_classmethod
        ):
            with self.subTest(meth=meth):
                self.assertEqual(meth.__doc__,
                                 ("My function docstring"
                                  if support.HAVE_DOCSTRINGS
                                  else None))
                self.assertEqual(meth.__annotations__["arg"], int)

        self.assertEqual(
            WithSingleDispatch.cls_context_manager.__name__,
            "cls_context_manager"
        )
        self.assertEqual(
            WithSingleDispatch().cls_context_manager.__name__,
            "cls_context_manager"
        )
        self.assertEqual(
            WithSingleDispatch.decorated_classmethod.__name__,
            "decorated_classmethod"
        )
        self.assertEqual(
            WithSingleDispatch().decorated_classmethod.__name__,
            "decorated_classmethod"
        )

    def test_invalid_registrations(self):
        msg_prefix = "Invalid first argument to `register()`: "
        msg_suffix = (
            ". Use either `@register(some_class)` or plain `@register` on an "
            "annotated function."
        )
        @functools.singledispatch
        def i(arg):
            return "base"
        with self.assertRaises(TypeError) as exc:
            @i.register(42)
            def _(arg):
                return "I annotated with a non-type"
        self.assertStartsWith(str(exc.exception), msg_prefix + "42")
        self.assertEndsWith(str(exc.exception), msg_suffix)
        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg):
                return "I forgot to annotate"
        self.assertStartsWith(str(exc.exception), msg_prefix +
            "<function TestSingleDispatch.test_invalid_registrations.<locals>._"
        )
        self.assertEndsWith(str(exc.exception), msg_suffix)

        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg: typing.Iterable[str]):
                # At runtime, dispatching on generics is impossible.
                # When registering implementations with singledispatch, avoid
                # types from `typing`. Instead, annotate with regular types
                # or ABCs.
                return "I annotated with a generic collection"
        self.assertStartsWith(str(exc.exception),
            "Invalid annotation for 'arg'."
        )
        self.assertEndsWith(str(exc.exception),
            "typing.Iterable[str] is not a class."
        )

        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg: typing.Union[int, typing.Iterable[str]]):
                return "Invalid Union"
        self.assertStartsWith(str(exc.exception),
            "Invalid annotation for 'arg'."
        )
        self.assertEndsWith(str(exc.exception),
            "int | typing.Iterable[str] not all arguments are classes."
        )

    def test_invalid_positional_argument(self):
        @functools.singledispatch
        def f(*args, **kwargs):
            pass
        msg = "f requires at least 1 positional argument"
        with self.assertRaisesRegex(TypeError, msg):
            f()
        msg = "f requires at least 1 positional argument"
        with self.assertRaisesRegex(TypeError, msg):
            f(a=1)

    def test_invalid_positional_argument_singledispatchmethod(self):
        class A:
            @functools.singledispatchmethod
            def t(self, *args, **kwargs):
                pass
        msg = "t requires at least 1 positional argument"
        with self.assertRaisesRegex(TypeError, msg):
            A().t()
        msg = "t requires at least 1 positional argument"
        with self.assertRaisesRegex(TypeError, msg):
            A().t(a=1)

    def test_union(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        @f.register
        def _(arg: typing.Union[str, bytes]):
            return "typing.Union"

        @f.register
        def _(arg: int | float):
            return "types.UnionType"

        self.assertEqual(f([]), "default")
        self.assertEqual(f(""), "typing.Union")
        self.assertEqual(f(b""), "typing.Union")
        self.assertEqual(f(1), "types.UnionType")
        self.assertEqual(f(1.0), "types.UnionType")

    def test_union_conflict(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        @f.register
        def _(arg: typing.Union[str, bytes]):
            return "typing.Union"

        @f.register
        def _(arg: int | str):
            return "types.UnionType"

        self.assertEqual(f([]), "default")
        self.assertEqual(f(""), "types.UnionType")  # last one wins
        self.assertEqual(f(b""), "typing.Union")
        self.assertEqual(f(1), "types.UnionType")

    def test_union_None(self):
        @functools.singledispatch
        def typing_union(arg):
            return "default"

        @typing_union.register
        def _(arg: typing.Union[str, None]):
            return "typing.Union"

        self.assertEqual(typing_union(1), "default")
        self.assertEqual(typing_union(""), "typing.Union")
        self.assertEqual(typing_union(None), "typing.Union")

        @functools.singledispatch
        def types_union(arg):
            return "default"

        @types_union.register
        def _(arg: int | None):
            return "types.UnionType"

        self.assertEqual(types_union(""), "default")
        self.assertEqual(types_union(1), "types.UnionType")
        self.assertEqual(types_union(None), "types.UnionType")

    def test_register_genericalias(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        f.register(list[int], lambda arg: "types.GenericAlias")
        f.register(list[float] | str, lambda arg: "types.UnionTypes(types.GenericAlias)")

        self.assertEqual(f([1]), "types.GenericAlias")
        self.assertEqual(f([1.0]), "types.UnionTypes(types.GenericAlias)")
        self.assertEqual(f(""), "types.UnionTypes(types.GenericAlias)")
        self.assertEqual(f(b""), "default")

    def test_register_genericalias_decorator(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        f.register(list[int])
        #f.register(typing.List[int])
        f.register(list[int] | str)
        #f.register(typing.List[int] | str)

    def test_register_genericalias_annotation(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        @f.register
        def _(arg: list[int]):
            return "types.GenericAlias"

        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: typing.List[float]):
                return "typing.GenericAlias"

        @f.register
        def _(arg: list[bytes] | str):
            return "types.UnionType(types.GenericAlias)"

        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: typing.List[float] | bytes):
                return "typing.Union[typing.GenericAlias]"

        self.assertEqual(f([1]), "types.GenericAlias")
        self.assertEqual(f([1.0]), "default")
        self.assertEqual(f(""), "types.UnionType(types.GenericAlias)")
        self.assertEqual(f(b""), "default")
        self.assertEqual(f([b""]), "types.UnionType(types.GenericAlias)")

    def test_forward_reference(self):
        @functools.singledispatch
        def f(arg, arg2=None):
            return "default"

        @f.register
        def _(arg: str, arg2: undefined = None):
            return "forward reference"

        self.assertEqual(f(1), "default")
        self.assertEqual(f(""), "forward reference")

    def test_unresolved_forward_reference(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        with self.assertRaisesRegex(TypeError, "is an unresolved forward reference"):
            @f.register
            def _(arg: undefined):
                return "forward reference"

    def test_method_equal_instances(self):
        # gh-127750: Reference to self was cached
        class A:
            def __eq__(self, other):
                return True
            def __hash__(self):
                return 1
            @functools.singledispatchmethod
            def t(self, arg):
                return self

        a = A()
        b = A()
        self.assertIs(a.t(1), a)
        self.assertIs(b.t(2), b)

    def test_method_bad_hash(self):
        class A:
            def __eq__(self, other):
                raise AssertionError
            def __hash__(self):
                raise AssertionError
            @functools.singledispatchmethod
            def t(self, arg):
                pass

        # Should not raise
        A().t(1)
        hash(A().t)
        A().t == A().t

    def test_method_no_reference_loops(self):
        # gh-127750: Created a strong reference to self
        class A:
            @functools.singledispatchmethod
            def t(self, arg):
                return weakref.ref(self)

        a = A()
        r = a.t(1)
        self.assertIsNotNone(r())
        del a  # delete a after a.t
        if not support.check_impl_detail(cpython=True):
            support.gc_collect()
        self.assertIsNone(r())

        a = A()
        t = a.t
        del a # delete a before a.t
        support.gc_collect()
        r = t(1)
        self.assertIsNotNone(r())
        del t
        if not support.check_impl_detail(cpython=True):
            support.gc_collect()
        self.assertIsNone(r())

    def test_signatures(self):
        @functools.singledispatch
        def func(item, arg: int) -> str:
            return str(item)
        @func.register
        def _(item: int, arg: bytes) -> str:
            return str(item)

        self.assertEqual(str(Signature.from_callable(func)),
                         "(item, arg: int) -> str")

    def test_method_signatures(self):
        class A:
            def m(self, item, arg: int) -> str:
                return str(item)
            @classmethod
            def cm(cls, item, arg: int) -> str:
                return str(item)
            @functools.singledispatchmethod
            def func(self, item, arg: int) -> str:
                return str(item)
            @func.register
            def _(self, item, arg: bytes) -> str:
                return str(item)

            @functools.singledispatchmethod
            @classmethod
            def cls_func(cls, item, arg: int) -> str:
                return str(arg)
            @func.register
            @classmethod
            def _(cls, item, arg: bytes) -> str:
                return str(item)

            @functools.singledispatchmethod
            @staticmethod
            def static_func(item, arg: int) -> str:
                return str(arg)
            @func.register
            @staticmethod
            def _(item, arg: bytes) -> str:
                return str(item)

        self.assertEqual(str(Signature.from_callable(A.func)),
                         "(self, item, arg: int) -> str")
        self.assertEqual(str(Signature.from_callable(A().func)),
                         "(self, item, arg: int) -> str")
        self.assertEqual(str(Signature.from_callable(A.cls_func)),
                         "(cls, item, arg: int) -> str")
        self.assertEqual(str(Signature.from_callable(A.static_func)),
                         "(item, arg: int) -> str")
