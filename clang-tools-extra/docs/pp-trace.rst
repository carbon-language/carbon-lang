.. index:: pp-trace

==================================
pp-trace User's Manual
==================================

.. toctree::
   :hidden:

:program:`pp-trace` is a standalone tool that traces preprocessor
activity.  It's also used as a test of Clang's PPCallbacks interface.
It runs a given source file through the Clang preprocessor, displaying
selected information from callback functions overidden in a PPCallbacks
derivation.  The output is in a high-level YAML format, described in
:ref:`OutputFormat`.

.. _Usage:

pp-trace Usage
==============

Command Line Format
-------------------

``pp-trace [<pp-trace-options>] <source-file> [<front-end-options>...]``

``<pp-trace-options>`` is a place-holder for options
specific to pp-trace, which are described below in
:ref:`CommandLineOptions`.

``<source-file>`` specifies the source file to run through the preprocessor.

``<front-end-options>`` is a place-holder for regular Clang
front-end arguments, which must follow the <source-file>.

.. _CommandLineOptions:

Command Line Options
--------------------

.. option:: -ignore <callback-name-list>

  This option specifies a comma-seperated list of names of callbacks
  that shouldn't be traced.  It can be used to eliminate unwanted
  trace output.  The callback names are the name of the actual
  callback function names in the PPCallbacks class:

  * FileChanged
  * FileSkipped
  * FileNotFound
  * InclusionDirective
  * moduleImport
  * EndOfMainFile
  * Ident
  * PragmaDirective
  * PragmaComment
  * PragmaDetectMismatch
  * PragmaDebug
  * PragmaMessage
  * PragmaDiagnosticPush
  * PragmaDiagnosticPop
  * PragmaDiagnostic
  * PragmaOpenCLExtension
  * PragmaWarning
  * PragmaWarningPush
  * PragmaWarningPop
  * MacroExpands
  * MacroDefined
  * MacroUndefined
  * Defined
  * SourceRangeSkipped
  * If
  * Elif
  * Ifdef
  * Ifndef
  * Else
  * Endif

.. option:: -output <output-file>

  By default, pp-trace outputs the trace information to stdout.  Use this
  option to output the trace information to a file.

.. _OutputFormat:

pp-trace Output Format
======================

The pp-trace output is formatted as YAML.  See http://yaml.org/ for general
YAML information.  It's arranged as a sequence of information about the
callback call, include the callback name and argument information, for
example:::

  ---
  - Callback: Name
    Argument1: Value1
    Argument2: Value2
  (etc.)
  ...

With real data:::

  ---
  - Callback: FileChanged
    Loc: "c:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-include.cpp:1:1"
    Reason: EnterFile
    FileType: C_User
    PrevFID: (invalid)
    (etc.)
  - Callback: FileChanged
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-include.cpp:5:1"
    Reason: ExitFile
    FileType: C_User
    PrevFID: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/Input/Level1B.h"
  - Callback: EndOfMainFile
  ...

In all but one case (MacroDirective) the "Argument" scalars have the same
name as the argument in the corresponding PPCallbacks callback function.

Callback Details
----------------

The following sections describe the output format for each callback.

FileChanged Callback
^^^^^^^^^^^^^^^^^^^^

+----------------+-----------------------------------------------------+------------------------------+-----------------------+
| Argument Name  | Argument Value Syntax                               | Clang C++ Type               | Description           |
+----------------+-----------------------------------------------------+------------------------------+-----------------------+
| Loc            | "(file):(line):(col)"                               | SourceLocation               | Where in the file.    |
+----------------+-----------------------------------------------------+------------------------------+-----------------------+
| Reason         | (EnterFile|ExitFile|SystemHeaderPragma|RenameFile)  | PPCallbacks::FileChangeReason| Reason for change.    |
+----------------+-----------------------------------------------------+------------------------------+-----------------------+
| FileType       | (C_User|C_System|C_ExternCSystem)                   | SrcMgr::CharacteristicKind   | Include type.         |
+----------------+-----------------------------------------------------+------------------------------+-----------------------+
| PrevFID        | ((file)|(invalid))                                  | FileID                       | Previous file, if any.|
+----------------+-----------------------------------------------------+------------------------------+-----------------------+

Example:::

  - Callback: FileChanged
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-include.cpp:1:1"
    Reason: EnterFile
    FileType: C_User
    PrevFID: (invalid)

(More callback documentation to come...)

Building pp-trace
=================

To build from source:

1. Read `Getting Started with the LLVM System`_ and `Clang Tools
   Documentation`_ for information on getting sources for LLVM, Clang, and
   Clang Extra Tools.

2. `Getting Started with the LLVM System`_ and `Building LLVM with CMake`_ give
   directions for how to build. With sources all checked out into the
   right place the LLVM build will build Clang Extra Tools and their
   dependencies automatically.

   * If using CMake, you can also use the ``pp-trace`` target to build
     just the pp-trace tool and its dependencies.

.. _Getting Started with the LLVM System: http://llvm.org/docs/GettingStarted.html
.. _Building LLVM with CMake: http://llvm.org/docs/CMake.html
.. _Clang Tools Documentation: http://clang.llvm.org/docs/ClangTools.html

