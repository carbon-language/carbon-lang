Introduction
============

Disclaimer: this document is currently somewhat out-of-date and is
retained for reference; for more recent documentation please refer to
LLVMC-Tutorial.rst.

A complete rewrite of the LLVMC compiler driver is proposed, aimed at
making it more configurable and useful.

Motivation
==========

As it stands, the current version of LLVMC does not meet its stated goals
of configurability and extensibility, and is therefore not used
much. The need for enhancements to LLVMC is also reflected in [1]_. The
proposed rewrite will fix the aforementioned deficiences and provide
an extensible, future-proof solution.

Design
======

A compiler driver's job is essentially to find a way to transform a set
of input files into a set of targets, depending on the user-provided
options. Since several methods of transformation can potentially exist,
it's natural to use a directed graph to represent all of them. In this
graph, nodes are tools -- e.g.,  ``gcc -S`` is a tool that generates
assembly from C language files -- and edges between the nodes indicate
that the output of one tool can be given as input to another -- i.e.,
``gcc -S -o - file.c | as``. We'll call this graph the compilation graph.

The proposed design revolves around the compilation graph and the
following core abstractions:

- Target - An (intermediate) compilation target.

- Action - A shell command template that represents a basic compilation
  transformation -- example: ``gcc -S $INPUT_FILE -o $OUTPUT_FILE``.

- Tool - Encapsulates information about a concrete tool used in the
  compilation process, produces Actions. Its operation depends on
  command-line options provided by the user.

- GraphBuilder - Constructs the compilation graph. Its operation
  depends on command-line options.

- GraphTraverser - Traverses the compilation graph and constructs a
  sequence of Actions needed to build the target file. Its operation
  depends on command-line options.

A high-level view of the compilation process:

  1. Configuration libraries (see below) are loaded in and the
     compilation graph is constructed from the tool descriptions.

  2. Information about possible options is gathered from (the nodes of)
     the compilation graph.

  3. Options are parsed based on data gathered in step 2.

  4. A sequence of Actions needed to build the target is constructed
     using the compilation graph and provided options.

  5. The resulting action sequence is executed.

Extensibility
==============

To make this design extensible, TableGen [2]_ will be used for
automatic generation of the Tool classes. Users wanting to customize
LLVMC need to write a configuration library consisting of a set of
TableGen descriptions of compilation tools plus a number of hooks
that influence compilation graph construction and traversal. LLVMC
will have the ability to load user configuration libraries at runtime;
in fact, its own basic functionality will be implemented as a
configuration library.

TableGen specification example
------------------------------

This small example specifies a Tool that converts C source to object
files. Note that it is only a mock-up of intended functionality, not a
final specification::

    def GCC : Tool<
     GCCProperties, // Properties of this tool
     GCCOptions     // Options description for this tool
    >;

    def GCCProperties : ToolProperties<[
     ToolName<"GCC">,
     InputLanguageName<"C">,
     OutputLanguageName<"Object-Code">
     InputFileExtension<"c">,
     OutputFileExtension<"o">,
     CommandFormat<"gcc -c $OPTIONS $FILES">
    ]>;

    def GCCOptions : ToolOptions<[
     Option<
       "-Wall",                 // Option name
       [None],                  // Allowed values
       [AddOption<"-Wall">]>,   // Action

     Option<
       "-Wextra",               // Option name
       [None],                  // Allowed values
       [AddOption<"-Wextra">]>, // Action

     Option<
       "-W",                 // Option name
       [None],               // Allowed values
       [AddOption<"-W">]>,   // Action

     Option<
       "-D",        // Option name
       [AnyString], // Allowed values

       [AddOptionWithArgument<"-D",GetOptionArgument<"-D">>]
       // Action:
       // If the driver was given option "-D<argument>", add
       // option "-D" with the same argument to the invocation string of
       // this tool.
       >

     ]>;

Example of generated code
-------------------------

The specification above compiles to the following code (again, it's a
mock-up)::

    class GCC : public Tool {

    public:

      GCC() { //... }

     // Properties

      static const char* ToolName = "GCC";
      static const char* InputLanguageName = "C";
      static const char* OutputLanguageName = "Object-Code";
      static const char* InputFileExtension = "c";
      static const char* OutputFileExtension = "o";
      static const char* CommandFormat = "gcc -c $OPTIONS $FILES";

     // Options

     OptionsDescription SupportedOptions() {
       OptionsDescription supportedOptions;

       supportedOptions.Add(Option("-Wall"));
       supportedOptions.Add(Option("-Wextra"));
       supportedOptions.Add(Option("-W"));
       supportedOptions.Add(Option("-D", AllowedArgs::ANY_STRING));

       return supportedOptions;
     }

     Action GenerateAction(Options providedOptions) {
       Action generatedAction(CommandFormat); Option curOpt;

       curOpt = providedOptions.Get("-D");
       if (curOpt) {
          assert(curOpt.HasArgument());
          generatedAction.AddOption(Option("-D", curOpt.GetArgument()));
       }

       curOpt = providedOptions.Get("-Wall");
       if (curOpt)
         generatedAction.AddOption(Option("-Wall"));

       curOpt = providedOptions.Get("-Wextra");
       if (curOpt)
         generatedAction.AddOption(Option("-Wall"));

       curOpt = providedOptions.Get("-W");
       if (curOpt)
         generatedAction.AddOption(Option("-Wall")); }

       return generatedAction;
     }

    };

    // defined somewhere...

    class Action { public: void AddOption(const Option& opt) {...}
    int Run(const Filenames& fnms) {...}

    }

Option handling
===============

Because one of the main tasks of the compiler driver is to correctly
handle user-provided options, it is important to define this process
in an exact way. The intent of the proposed scheme is to function as
a drop-in replacement for GCC.

Option syntax
-------------

The option syntax is specified by the following formal grammar::

        <command-line>      ::=  <option>*
        <option>            ::=  <positional-option> | <named-option>
        <named-option>      ::=  -[-]<option-name>[<delimeter><option-argument>]
        <delimeter>         ::=  ',' | '=' | ' '
        <positional-option> ::=  <string>
        <option-name>       ::=  <string>
        <option-argument>   ::=  <string>

This roughly corresponds to the GCC option syntax. Note that grouping
of short options (as in ``ls -la``) is forbidden.

Example::

        llvmc -O3 -Wa,-foo,-bar -pedantic -std=c++0x a.c b.c c.c

Option arguments can also have special forms. For example, an argument
can be a comma-separated list (like in -Wa,-foo,-bar). In such cases,
it's up to the option handler to parse the argument.

Option semantics
----------------

According to their meaning, options are classified into the following
categories:

- Global options - Options that influence compilation graph
  construction/traversal. Example: -E (stop after preprocessing).

- Local options - Options that influence one or several Actions in
  the generated action sequence. Example: -O3 (turn on optimization).

- Prefix options - Options that influence the meaning of the following
  command-line arguments. Example: -x language (specify language for
  the input files explicitly). Prefix options can be local or global.

- Built-in options - Options that are hard-coded into the
  driver. Examples: --help, -o file/-pipe (redirect output). Can be
  local or global.

Issues
======

1. Should global-options-influencing hooks be written by hand or
   auto-generated from TableGen specifications?

2. More?

References
==========

.. [1] LLVM Bug#686

       http://llvm.org/bugs/show_bug.cgi?id=686

.. [2] TableGen Fundamentals

       http://llvm.org/docs/TableGenFundamentals.html
