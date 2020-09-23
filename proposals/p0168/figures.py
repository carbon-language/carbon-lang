#! /usr/bin/env python
import itertools

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
    # codefont = '<font face="SFMono-Regular,Consolasl,Liberation Mono,Menlo,monospace" point-size="10.2">%s</font>' % escape(s)
    codefont = "".join(
        '<font face="SFMono-Regular,Consolasl,Liberation Mono,Menlo,monospace" point-size="10.2">%s</font> '
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
    print >> out, "  %s [label=<%s>%s]" % (
        name,
        tablejoin((style(op) for op in ops), ", "),
        assoc,
    )
    return name


def edge(a, b):
    print >> out, "  %s -> %s" % (a, b)


def combine(name, items):
    if len(items) <= 1:
        return items
    print >> out, "  %s [label=<<i>%s</i>> shape=ellipse]" % (name, name)
    res = name
    for i in items:
        edge(i, name)
    return [res]


def graph(f):
    import subprocess

    outfile = file(f.func_name + "." + fmt, "w")
    process = subprocess.Popen(
        ["dot", "-T" + fmt], stdin=subprocess.PIPE, stdout=outfile
    )
    global out
    out = process.stdin
    print >> out, "digraph {"
    print >> out, "  layout = dot"
    print >> out, "  rankdir = TB"
    print >> out, '  rank = "min"'
    # print >>out, '  node [shape="rectangle" style="rounded" fontname="Arial"]'
    # Remove minimum height to avoid overly-tall nodes.
    print >> out, '  node [shape="none", fontname="BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" fontsize="12" height="0"]'
    print >> out, '  edge [dir="none"]'
    f()
    print >> out, "}"
    process.communicate()
    return f


@graph
def bodmas():
    brackets = group(["(&#8943;)"], style=raw)
    exponents = group(
        ['a<sup><font point-size="8">  b</font></sup>'], RtL, style=math
    )
    allplusminus = group(["-a", "a</i>  +<i>b", "a - b"], LtR, style=math)
    muldiv = group(["a  </i> x <i>b", "a &#247; b"], LtR, style=math)
    edge(brackets, exponents)
    edge(exponents, muldiv)
    edge(muldiv, allplusminus)


@graph
def arithmetic(term=None):
    unaryminus = group(["-a"], NonAssoc)
    incdec = group(["--a", "++a"], NonAssoc)
    muldiv = group(["a * b", "a / b"], LtR)
    mod = group(["a % b"], NonAssoc)
    addition = group(["a + b", "a - b"], LtR)

    if term:
        edge(term, unaryminus)
        edge(term, incdec)
    edge(unaryminus, muldiv)
    edge(unaryminus, mod)
    edge(incdec, muldiv)
    edge(incdec, mod)
    edge(muldiv, addition)
    return [addition, mod]


@graph
def bitwise(term=None):
    compl = group(["~a"], NonAssoc)
    bitand = group(["a & b"], LtR)
    bitor = group(["a | b"], LtR)
    bitxor = group(["a ^ b"], LtR)

    if term:
        edge(term, compl)
    edge(compl, bitand)
    edge(compl, bitor)
    edge(compl, bitxor)
    return [bitand, bitor, bitxor]


@graph
def logical(term=None, comparisons=[]):
    not_ = group(["not a"], NonAssoc)
    and_ = group(["a and b"], LtR)
    or_ = group(["a or b"], LtR)
    if term:
        edge(term, not_)
    for c in combine("predicate", [not_] + comparisons):
        edge(c, and_)
        edge(c, or_)
    return [and_, or_]


@graph
def comparisons(value=[]):
    rel = group(["a < b", "a <= b", "a > b", "a >= b"], NonAssoc)
    eq = group(["a == b", "a != b"], NonAssoc)
    for v in value:
        edge(v, rel)
        edge(v, eq)
    return [rel, eq]


@graph
def assignment(value=[]):
    assign = group(["a = b"], RtL)
    compassign = group(["a $= b"], NonAssoc)
    for v in value:
        edge(v, assign)
        edge(v, compassign)
    return [assign, compassign]


@graph
def all():
    paren = group(["(...)"], NonAssoc)
    term = group(["a(...)", "a[...]", "a.m"], LtR)
    edge(paren, term)

    value = combine("value", arithmetic(term) + bitwise(term))
    comp = comparisons(value)
    assign = assignment(value)
    andor = logical(term, comp)

    comma = group(["a, b"], LtR)
    for e in assign + andor:
        edge(e, comma)
