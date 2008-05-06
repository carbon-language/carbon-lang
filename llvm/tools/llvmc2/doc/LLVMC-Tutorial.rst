Tutorial - Writing LLVMC Configuration files
=============================================

LLVMC is a generic compiler driver, designed to be customizable and
extensible. It plays the same role for LLVM as the ``gcc`` program
does for GCC - LLVMC's job is essentially to transform a set of input
files into a set of targets depending on configuration rules and user
options. What makes LLVMC different is that these transformation rules
are completely customizable - in fact, LLVMC knows nothing about the
specifics of transformation (even the command-line options are mostly
not hard-coded) and regards the transformation structure as an
abstract graph. This makes it possible to adapt LLVMC for other
purposes - for example, as a build tool for game resources. This
tutorial describes the basic usage and configuration of LLVMC.

Because LLVMC employs TableGen [1]_ as its configuration language, you
need to be familiar with it to customize LLVMC.

Compiling with LLVMC
--------------------

In general, LLVMC tries to be command-line compatible with ``gcc`` as
much as possible, so most of the familiar options work::

     $ llvmc2 -O3 -Wall hello.cpp
     $ ./a.out
     hello

One nice feature of LLVMC is that you don't have to distinguish
between different compilers for different languages (think ``g++`` and
``gcc``) - the right toolchain is chosen automatically based on input
language names (which are, in turn, determined from file extension). If
you want to force files ending with ".c" compile as C++, use the
``-x`` option, just like you would do it with ``gcc``::

      $ llvmc2 -x c hello.cpp
      $ # hello.cpp is really a C file
      $ ./a.out
      hello

On the other hand, when using LLVMC as a linker to combine several C++
object files you should provide the ``--linker`` option since it's
impossible for LLVMC to choose the right linker in that case::

    $ llvmc2 -c hello.cpp
    $ llvmc2 hello.o
    [A lot of link-time errors skipped]
    $ llvmc2 --linker=c++ hello.o
    $ ./a.out
    hello

For further help on command-line LLVMC usage, refer to the ``llvmc
--help`` output.

Customizing LLVMC: the compilation graph
----------------------------------------

At the time of writing LLVMC does not support on-the-fly reloading of
configuration, so to customize LLVMC you'll have to edit and recompile
the source code (which lives under ``$LLVM_DIR/tools/llvmc2``). The
relevant files are ``Common.td``, ``Tools.td`` and ``Example.td``.

Internally, LLVMC stores information about possible transformations in
form of a graph. Nodes in this graph represent tools, and edges
between two nodes represent a transformation path. A special "root"
node represents entry points for the transformations. LLVMC also
assigns a weight to each edge (more on that below) to choose between
several alternative edges.

The definition of the compilation graph (see file ``Example.td``) is
just a list of edges::

    def CompilationGraph : CompilationGraph<[
        Edge<root, llvm_gcc_c>,
        Edge<root, llvm_gcc_assembler>,
        ...

        Edge<llvm_gcc_c, llc>,
        Edge<llvm_gcc_cpp, llc>,
        ...

        OptionalEdge<llvm_gcc_c, opt, [(switch_on "opt")]>,
        OptionalEdge<llvm_gcc_cpp, opt, [(switch_on "opt")]>,
        ...

        OptionalEdge<llvm_gcc_assembler, llvm_gcc_cpp_linker,
            [(if_input_languages_contain "c++"),
             (or (parameter_equals "linker", "g++"),
             (parameter_equals "linker", "c++"))]>,
        ...

        ]>;

As you can see, the edges can be either default or optional, where
optional edges are differentiated by sporting a list of patterns (or
edge properties) which are used to calculate the edge's weight. The
default edges are assigned a weight of 1, and optional edges get a
weight of 0 + 2*N where N is the number of succesful edge property
matches. When passing an input file through the graph, LLVMC picks the
edge with the maximum weight. To avoid ambiguity, there should be only
one default edge between two nodes (with the exception of the root
node, which gets a special treatment - there you are allowed to
specify one default edge *per language*).

* Possible edge properties are:

  - ``switch_on`` - Returns true if a given command-line option is
    provided by the user. Example: ``(switch_on "opt")``. Note that
    you have to define all possible command-line options separately in
    the tool descriptions. See the next section for the discussion of
    different kinds of command-line options.

  - ``parameter_equals`` - Returns true if a command-line parameter equals
    a given value. Example: ``(parameter_equals "W", "all")``.

  - ``element_in_list`` - Returns true if a command-line parameter list
    includes a given value. Example: ``(parameter_in_list "l", "pthread")``.

  - ``if_input_languages_contain`` - Returns true if a given input
    language belongs to the current input language set.

  - ``and`` - Edge property combinator. Returns true if all of its
    arguments return true. Used like this: ``(and (prop1), (prop2),
    ... (propN))``. Nesting is allowed, but not encouraged.

  - ``or`` - Edge property combinator that returns true if any one of its
    arguments returns true. Example: ``(or (prop1), (prop2), ... (propN))``.

  - ``weight`` - Makes it possible to explicitly specify the quantity
    added to the edge weight if this edge property matches. Used like
    this: ``(weight N, (prop))``. The inner property can include
    ``and`` and ``or`` combinators. When N is equal to 2, equivalent
    to ``(prop)``.

    Example: ``(weight 8, (and (switch_on "a"), (switch_on "b")))``.


To get a visual representation of the compilation graph (useful for
debugging), run ``llvmc2 --view-graph``. You will need ``dot`` and
``gsview`` installed for this to work properly.


Writing a tool description
--------------------------

As was said earlier, nodes in the compilation graph represent tools. A
tool definition looks like this (taken from the ``Tools.td`` file)::

  def llvm_gcc_cpp : Tool<[
      (in_language "c++"),
      (out_language "llvm-assembler"),
      (output_suffix "bc"),
      (cmd_line "llvm-g++ -c $INFILE -o $OUTFILE -emit-llvm"),
      (sink)
      ]>;

This defines a new tool called ``llvm_gcc_cpp``, which is an alias for
``llvm-g++``. As you can see, a tool definition is just a list of
properties; most of them should be self-evident. The ``sink`` property
means that this tool should be passed all command-line options that
aren't handled by the other tools.

The complete list of the currently implemented tool properties follows:

* Possible tool properties:

  - ``in_language`` - input language name.

  - ``out_language`` - output language name.

  - ``output_suffix`` - output file suffix.

  - ``cmd_line`` - the actual command used to run the tool. You can use
    ``$INFILE`` and ``$OUTFILE`` variables, as well as output
    redirection with ``>``.

  - ``join`` - this tool is a "join node" in the graph, i.e. it gets a
    list of input files and joins them together. Used for linkers.

  - ``sink`` - all command-line options that are not handled by other
    tools are passed to this tool.

The next tool definition is slightly more complex::

  def llvm_gcc_linker : Tool<[
      (in_language "object-code"),
      (out_language "executable"),
      (output_suffix "out"),
      (cmd_line "llvm-gcc $INFILE -o $OUTFILE"),
      (join),
      (prefix_list_option "L", (forward), (help "add a directory to link path")),
      (prefix_list_option "l", (forward), (help "search a library when linking")),
      (prefix_list_option "Wl", (unpack_values), (help "pass options to linker"))
      ]>;

This tool has a "join" property, which means that it behaves like a
linker (because of that this tool should be the last in the
toolchain). This tool also defines several command-line options: ``-l``,
``-L`` and ``-Wl`` which have their usual meaning. An option has two
attributes: a name and a (possibly empty) list of properties. All
currently implemented option types and properties are described below:

* Possible option types:

   - ``switch_option`` - a simple boolean switch, for example ``-time``.

   - ``parameter_option`` - option that takes an argument, for example
     ``-std=c99``;

   - ``parameter_list_option`` - same as the above, but more than one
     occurence of the option is allowed.

   - ``prefix_option`` - same as the parameter_option, but the option name
     and parameter value are not separated.

   - ``prefix_list_option`` - same as the above, but more than one
     occurence of the option is allowed; example: ``-lm -lpthread``.


* Possible option properties:

   - ``append_cmd`` - append a string to the tool invocation command.

   - ``forward`` - forward this option unchanged.

   - ``stop_compilation`` - stop compilation after this phase.

   - ``unpack_values`` - used for for splitting and forwarding
     comma-separated lists of options, e.g. ``-Wa,-foo=bar,-baz`` is
     converted to ``-foo=bar -baz`` and appended to the tool invocation
     command.

   - ``help`` - help string associated with this option.

   - ``required`` - this option is obligatory.


Language map
------------

One last thing that you need to modify when adding support for a new
language to LLVMC is the language map, which defines mappings from
file extensions to language names. It is used to choose the proper
toolchain based on the input. Language map definition is located in
the file ``Tools.td`` and looks like this::

    def LanguageMap : LanguageMap<
        [LangToSuffixes<"c++", ["cc", "cp", "cxx", "cpp", "CPP", "c++", "C"]>,
         LangToSuffixes<"c", ["c"]>,
         ...
        ]>;


References
==========

.. [1] TableGen Fundamentals
       http://llvm.cs.uiuc.edu/docs/TableGenFundamentals.html
