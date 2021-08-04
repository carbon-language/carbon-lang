#! /usr/bin/env python

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

fmt = "svg"


def escape(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
    )


def tablejoin(items, separator):
    data = ("<td>%s</td>" % separator).join(
        "<td>%s</td>" % item for item in items
    )
    return '<table border="0"><tr>%s</tr></table>' % data


def code(s):
    # FIXME: GraphViz's handling of font metrics appears to be pretty broken.
    # Add a little extra width to each character with a non-code-font space to
    # compensate.
    codefont = "".join(
        (
            '<font face="SFMono-Regular,Consolasl,Liberation Mono,'
            + 'Menlo,monospace" point-size="10.2">%s</font> '
        )
        % escape(part)
        for part in s
    )
    return (
        '<table border="0" bgcolor="#f2f3f3"><tr><td>%s</td></tr></table>'
        % codefont
    )


def math(s):
    # Render math in italics but otherwise unchanged.
    return "<i>%s</i>" % s


def raw(s):
    return s


LtR = ' shape="rarrow"'
RtL = ' shape="larrow"'
NonAssoc = ""

out = None
num = 0


def group(ops, assoc=NonAssoc, style=code):
    global num
    num = num + 1
    name = "op%d" % num
    print(
        "  %s [label=<%s>%s]"
        % (
            name,
            tablejoin((style(op) for op in ops), ", "),
            assoc,
        ),
        file=out,
    )
    return name


def edge(a, b):
    print("  %s -> %s" % (a, b), file=out)


def combine(name, items):
    if len(items) <= 1:
        return items
    print("  %s [label=<<i>%s</i>> shape=ellipse]" % (name, name), file=out)
    res = name
    for i in items:
        edge(i, name)
    return [res]


def graph(f):
    import subprocess

    outfile = open(f.__name__ + "." + fmt, "w")
    process = subprocess.Popen(
        ["dot", "-T" + fmt],
        stdin=subprocess.PIPE,
        stdout=outfile,
        encoding="utf8"
        # ["cat"], stdin=subprocess.PIPE, stdout=outfile, encoding='utf8'
    )
    global out
    out = process.stdin
    # print >>out, '  node [shape="rectangle" style="rounded" fontname="Arial"]'
    print(
        """
digraph {
  layout = dot
  rankdir = TB
  rank = "min"
  node [shape="none" fontsize="12" height="0"
        fontname="BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"]
  edge [dir="none"]
  """.strip(),
        file=out,
    )
    f()
    print("}", file=out)
    process.communicate()
    return f


@graph
def example():
    term = group(["(...)"], NonAssoc)
    mul = group(["a * b"], LtR)
    add = group(["a + b"], LtR)
    shl = group(["a << b"], NonAssoc)
    compare = group(["a == b"], NonAssoc)

    edge(term, mul)
    edge(mul, add)
    edge(term, shl)
    edge(add, compare)
    edge(shl, compare)
