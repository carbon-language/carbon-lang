Tutorial - Writing LLVMCC Configuration files
=============================================

LLVMCC is a generic compiler driver (just like ``gcc``), designed to be
customizable and extensible. Its job is essentially to transform a set
of input files into a set of targets, depending on configuration rules
and user options. This tutorial describes how one can write
configuration files for ``llvmcc``.

Because LLVMCC uses TableGen [1]_ as the language of its configuration
files, you need to be familiar with it.

Describing a toolchain
----------------------

The main concept that ``llvmcc`` operates with is a *toolchain*, which
is just a list of tools that process input files in a pipeline-like
fashion. Toolchain definitions look like this::

   def ToolChains : ToolChains<[
       ToolChain<[llvm_gcc_c, llc, llvm_gcc_assembler, llvm_gcc_linker]>,
       ToolChain<[llvm_gcc_cpp, llc, llvm_gcc_assembler, llvm_gcc_linker]>,
       ...
       ]>;

Every configuration file should have a single toolchains list called
``ToolChains``.

At the time of writing, ``llvmcc`` does not support mixing various
toolchains together - in other words, all input files should be in the
same language.

Another temporary limitation is that every toolchain should end with a
"join" node - a linker-like program that combines its inputs into a
single output file.

Describing a tool
-----------------

A single element of a toolchain is a tool. A tool definition looks
like this (taken from the Tools.td file)::

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
  - in_language - input language name.

  - out_language - output language name.

  - output_suffix - output file suffix.

  - cmd_line - the actual command used to run the tool. You can use
    ``$INFILE`` and ``$OUTFILE`` variables.

  - join - this tool is a "join node" in the graph, i.e. it gets a
    list of input files and joins them together. Used for linkers.

  - sink - all command-line options that are not handled by other
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
   - switch_option - a simple boolean switch, for example ``-time``.

   - parameter_option - option that takes an argument, for example ``-std=c99``;

   - parameter_list_option - same as the above, but more than one
     occurence of the option is allowed.

   - prefix_option - same as the parameter_option, but the option name
     and parameter value are not separated.

   - prefix_list_option - same as the above, but more than one
     occurence of the option is allowed; example: ``-lm -lpthread``.

* Possible option properties:
   - append_cmd - append a string to the tool invocation command.

   - forward - forward this option unchanged.

   - stop_compilation - stop compilation after this phase.

   - unpack_values - used for for splitting and forwarding
     comma-separated lists of options, e.g. ``-Wa,-foo=bar,-baz`` is
     converted to ``-foo=bar -baz`` and appended to the tool invocation
     command.

   - help - help string associated with this option.

   - required - this option is obligatory.

Language map
------------

One last bit that you probably should change is the language map,
which defines mappings between language names and file extensions. It
is used internally to choose the proper toolchain based on the names
of the input files. Language map definition is located in the file
``Tools.td`` and looks like this::

    def LanguageMap : LanguageMap<
        [LangToSuffixes<"c++", ["cc", "cp", "cxx", "cpp", "CPP", "c++", "C"]>,
         LangToSuffixes<"c", ["c"]>,
         ...
        ]>;


Putting it all together
-----------------------

Since at the time of writing LLVMCC does not support on-the-fly
reloading of the configuration, the only way to test your changes is
to recompile the program. To do this, ``cd`` to the source code
directory and run ``make``.

References
==========

.. [1] TableGen Fundamentals
       http://llvm.cs.uiuc.edu/docs/TableGenFundamentals.html
