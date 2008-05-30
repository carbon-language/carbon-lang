Tutorial - Using LLVMC
======================

LLVMC is a generic compiler driver, which plays the same role for LLVM
as the ``gcc`` program does for GCC - the difference being that LLVMC
is designed to be more adaptable and easier to customize. This
tutorial describes the basic usage and configuration of LLVMC.

Compiling with LLVMC
--------------------

In general, LLVMC tries to be command-line compatible with ``gcc`` as
much as possible, so most of the familiar options work::

     $ llvmc2 -O3 -Wall hello.cpp
     $ ./a.out
     hello

For further help on command-line LLVMC usage, refer to the ``llvmc
--help`` output.

Using LLVMC to generate toolchain drivers
-----------------------------------------

At the time of writing LLVMC does not support on-the-fly reloading of
configuration, so it will be necessary to recompile its source
code. LLVMC uses TableGen [1]_ as its configuration language, so
you'll need to familiar with it.

Start by compiling ``examples/Simple.td``, which is a simple wrapper
for ``gcc``::

    $ cd $LLVM_DIR/tools/llvmc2
    $ make TOOLNAME=mygcc GRAPH=examples/Simple.td
    $ edit hello.c
    $ mygcc hello.c
    $ ./hello.out
    Hello

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

At the heart of LLVMC is the idea of a transformation graph: vertices
in this graph are tools, and edges signify that there is a
transformation path between two tools (for example, assembly source
produced by the compiler can be transformed into executable code by an
assembler). A special node named ``root`` is used to mark the graph
entry points.

Tool descriptions are basically lists of properties: most properties
in the example above should be self-explanatory; the ``sink`` property
means that all options lacking an explicit description should be
forwarded to this tool.

``LanguageMap`` associates a language name with a list of suffixes and
is used for deciding which toolchain corresponds to a given input
file.

To learn more about LLVMC customization, refer to the reference
manual and sample configuration files in the ``examples`` directory.

References
==========

.. [1] TableGen Fundamentals
       http://llvm.cs.uiuc.edu/docs/TableGenFundamentals.html

