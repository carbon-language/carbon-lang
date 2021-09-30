#!/usr/bin/env python3

"""Argument passing algorithm experiments.

Problem: Given this interface definition:

interface C {
  let M:! C;
  let L:! C{.R = M};
  let R:! C{.L = M};
}

Want to find canonical form of `S(.L|.R)*` for binding `S:! C`.
Question is whether e.g. `S.L^M.R^N` equals `S.R^N.L^M`.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from collections import namedtuple


Decl = namedtuple("Decl", ["n", "t"])
Env = namedtuple("Env", ["bind", "iface"])


class Archetype:
    def __init__(self, n, t, b=()):
        self.n = n
        self.t = t
        self.b = b

    def is_arch(self):
        return True

    def is_point(self):
        return False

    def __repr__(self):
        return "Archetype(n=%s, t=%s, b=%s)" % (self.n, self.t, self.b)


class Pointer:
    def __init__(self, n):
        self.n = n

    def is_arch(self):
        return False

    def is_point(self):
        return True

    def __repr__(self):
        return "Pointer(n=%s)" % (self.n,)


def commute_example():
    return Env(
        bind={"S": Archetype("S", "C")},
        iface={
            "C": (
                ("M", ("C", ())),
                ("L", ("C", (("R", "M"),))),
                ("R", ("C", (("L", "M"),))),
            )
        },
    )


def graph_example():
    return Env(
        bind={"S": Archetype("S", "Graph")},
        iface={
            "Graph": (
                ("E", ("Edge", (("V", "V"),))),
                ("V", ("Vert", (("E", "E"),))),
            ),
            "Edge": (("V", ("Vert", ())),),
            "Vert": (("E", ("Edge", ())),),
        },
    )


def name(*args):
    return ".".join(a for a in args if a)


def parent(n):
    return ".".join(n.split(".")[:-1])


def expand(e, n, b=()):
    print("Expand: n =", n, "b =", b)
    assert n in e.bind, "%s not in %s" % (n, e.bind)
    if e.bind[n].is_point():
        print("...", n, "is a pointer to", e.bind[n].n)
        return e
    assert e.bind[n].is_arch()
    nb = e.bind[n]
    iface = e.iface[nb.t]
    for n_, v in b:
        pointerify(e, name(n, n_), member_lookup(n_, iface), name(parent(n), v))
    for n_, t in iface:
        nn = name(n, n_)
        if nn not in e.bind:
            na = Archetype(nn, t[0], t[1])
            print("Create", na)
            e.bind[nn] = na
    return e


def member_lookup(n, iface):
    for k, v in iface:
        if k == n:
            return v
    assert False


def type_lookup(e, n):
    s = n.split(".")
    ia = e.bind[s[0]]
    assert ia.is_arch()
    t = ia.t
    b = ia.b
    s = s[1:]
    for a in s:
        iface = e.iface[t]
        t, b = member_lookup(a, iface)
    print("type_lookup:", n, "to t:", t, "b:", b)
    return (t, b)


def pointerify(e, source, t, dest):
    print("Pointerify", source, t, "->", dest)
    if dest not in e.bind:
        print("Canonicalize dest:", dest)
        result = canonicalize(e, dest)
        print("Canonicalize result:", dest, "->", result)
        # Not required, but saves pointer-to-pointer steps:
        dest = result.n
    else:
        print("Dest exists:", e.bind[dest])
        # Not required, but saves pointer-to-pointer steps:
        dest = e.bind[dest].n
    assert source not in e.bind
    print("Set", source, "pointer to:", dest)
    e.bind[source] = Pointer(dest)
    for k, v in t[1]:
        new_source = name(parent(source), v)
        pointerify(e, new_source, type_lookup(e, new_source), name(dest, k))


def canonicalize(e, n):
    p = ""
    for a in n.split("."):
        s = name(p, a)
        if s not in e.bind:
            print("To expand:", p, "with binding:", e.bind[p])
            expand(e, p, e.bind[p].b)
            assert s in e.bind
        if e.bind[s].is_point():
            return canonicalize(e, e.bind[s].n + n[len(s) :])
        p = s
    return e.bind[s]


def resolve(n, f):
    print("Resolve:", n)
    e = f()
    b = canonicalize(e, n)
    assert b.is_arch()
    print("Resolved", n, "to", b.n)
    print()
    return b.n


# results = [(n, resolve(n, commute_example)) for n in
#            ('S.L.L', 'S.L.R', 'S.R.L', 'S.R.R')]
# results = [(n, resolve(n, commute_example)) for n in ('S.L.R',)]

# results = [(n, resolve(n, commute_example)) for n in
#            ('S.L.L.L.R.R', 'S.L.L.R.L.R', 'S.L.L.R.R.L', 'S.L.R.L.L.R',
#             'S.L.R.L.R.L', 'S.L.R.R.L.L', 'S.R.L.L.L.R', 'S.R.L.L.R.L',
#             'S.R.L.R.L.L', 'S.R.R.L.L.L')]

# import itertools
# trials = ['.'.join(('S',) + x) for x in itertools.product('LR', repeat=5)]
# results = [(n, resolve(n, commute_example)) for n in trials]

trials = (
    "S.E",
    "S.V",
    "S.E.V",
    "S.V.E",
    "S.E.V.E",
    "S.V.E.V",
    "S.E.V.E.V",
    "S.V.E.V.E",
    "S.E.V.E.V.E",
    "S.V.E.V.E.V",
)
results = [(n, resolve(n, graph_example)) for n in trials]

print()
print("RESULTS:")
for n, r in results:
    print("    ", n, "->", r)
