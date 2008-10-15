======================
Tutorial - Using LLVMC
======================

LLVMC is a generic compiler driver, which plays the same role for LLVM
as the ``gcc`` program does for GCC - the difference being that LLVMC
is designed to be more adaptable and easier to customize. Most of
LLVMC functionality is implemented via plugins, which can be loaded
dynamically or compiled in. This tutorial describes the basic usage
and configuration of LLVMC.


.. contents::


Compiling with LLVMC
====================

In general, LLVMC tries to be command-line compatible with ``gcc`` as
much as possible, so most of the familiar options work::

     $ llvmc2 -O3 -Wall hello.cpp
     $ ./a.out
     hello

For further help on command-line LLVMC usage, refer to the ``llvmc
--help`` output.

Using LLVMC to generate toolchain drivers
=========================================

LLVMC plugins are written mostly using TableGen [1]_, so you need to
be familiar with it to get anything done.

Start by compiling ``plugins/Simple/Simple.td``, which is a primitive
wrapper for ``gcc``::

    $ cd $LLVM_DIR/tools/llvmc2
    $ make DRIVER_NAME=mygcc BUILTIN_PLUGINS=Simple
    $ cat > hello.c
    [...]
    $ mygcc hello.c
    $ ./hello.out
    Hello

Here we link our plugin with the LLVMC core statically to form an
executable file called ``mygcc``. It is also possible to build our
plugin as a standalone dynamic library; this is described in the
reference manual.

Contents of the file ``Simple.td`` look like this::

    // Include common definitions
    include "Common.td"

    // Tool descriptions
    def gcc : Tool<
    [(in_language "c"),
     (out_language "executable"),
     (output_suffix "out"),
     (cmd_line "gcc $INFILE -o $OUTFILE"),
     (sink)
    ]>;

    // Language map
    def LanguageMap : LanguageMap<[LangToSuffixes<"c", ["c"]>]>;

    // Compilation graph
    def CompilationGraph : CompilationGraph<[Edge<root, gcc>]>;

As you can see, this file consists of three parts: tool descriptions,
language map, and the compilation graph definition.

At the heart of LLVMC is the idea of a compilation graph: vertices in
this graph are tools, and edges represent a transformation path
between two tools (for example, assembly source produced by the
compiler can be transformed into executable code by an assembler). The
compilation graph is basically a list of edges; a special node named
``root`` is used to mark graph entry points.

Tool descriptions are represented as property lists: most properties
in the example above should be self-explanatory; the ``sink`` property
means that all options lacking an explicit description should be
forwarded to this tool.

The ``LanguageMap`` associates a language name with a list of suffixes
and is used for deciding which toolchain corresponds to a given input
file.

To learn more about LLVMC customization, refer to the reference
manual and plugin source code in the ``plugins`` directory.

References
==========

.. [1] TableGen Fundamentals
       http://llvm.cs.uiuc.edu/docs/TableGenFundamentals.html

