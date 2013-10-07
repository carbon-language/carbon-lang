=====================
clang-modernize Usage
=====================

``clang-modernize [options] [<sources>...] [-- [args]]``

``<source#>`` specifies the path to the source to migrate. This path may be
relative to the current directory. If no sources are provided, a compilation
database provided with `-p`_ can be used to provide sources together with the
`include/exclude options`_.

By default all transformations are applied. There are two ways to enable a
subset of the transformations:

1. Explicitly, by referring to the transform options directly, see
   :ref:`transform-specific-command-line-options`.
2. Implicitly, based on the compilers to support, see
   :ref:`-for-compilers=\<string\> <for-compilers-option>`.

If both ways of specifying transforms are used only explicitly specified
transformations that are supported by the given compilers will be applied.

General Command Line Options
============================

.. option:: -help

  Displays tool usage instructions and command line options.

.. option:: -version

  Displays the version information of this tool.

.. _-p:

.. option:: -p=<build-path>

  ``<build-path>`` is the directory containing a *compilation databasefile*, a
  file named ``compile_commands.json``, which provides compiler arguments for
  building each source file. CMake can generate this file by specifying
  ``-DCMAKE_EXPORT_COMPILE_COMMANDS`` when running CMake. Ninja_, since v1.2 can
  also generate this file with ``ninja -t compdb``. If the compilation database
  cannot be used for any reason, an error is reported.

  This option is ignored if ``--`` is present.

  Files in the compilation database will be transformed if no sources are
  provided and paths to files are explicitly included using ``-include`` or
  ``-include-from``.
  In order to transform all files in a compilation database the following
  command line can be used:

    ``clang-modernize -p=<build-path> -include=<project_root>``

  Use ``-exclude`` or ``-exclude-from`` to limit the scope of ``-include``.

.. _Ninja: http://martine.github.io/ninja/

.. option:: -- [args]

  Another way to provide compiler arguments is to specify all arguments on the
  command line following ``--``. Arguments provided this way are used for
  *every* source file.

  If neither ``--`` nor ``-p`` are specified a compilation database is
  searched for starting with the path of the first-provided source file and
  proceeding through parent directories. If no compilation database is found or
  one is found and cannot be used for any reason then ``-std=c++11`` is used as
  the only compiler argument.

.. option:: -risk=<risk-level>

  Some transformations may cause a change in semantics. In such cases the
  maximum acceptable risk level specified through the ``-risk`` command
  line option decides whether or not a transformation is applied.

  Three different risk level options are available:

    ``-risk=safe``
      Perform only safe transformations.
    ``-risk=reasonable`` (default)
      Enable transformations that may change semantics.
    ``-risk=risky``
      Enable transformations that are likely to change semantics.

  The meaning of risk is handled differently for each transform. See
  :ref:`transform documentation <transforms>` for details.

.. option:: -final-syntax-check

  After applying the final transform to a file, parse the file to ensure the
  last transform did not introduce syntax errors. Syntax errors introduced by
  earlier transforms are already caught when subsequent transforms parse the
  file.

.. option:: -summary

  Displays a summary of the number of changes each transform made or could have
  made to each source file immediately after each transform is applied.
  **Accepted** changes are those actually made. **Rejected** changes are those
  that could have been made if the acceptable risk level were higher.
  **Deferred** changes are those that might be possible but they might conflict
  with other accepted changes. Re-applying the transform will resolve deferred
  changes.

.. _for-compilers-option:

.. option:: -for-compilers=<string>

  Select transforms targeting the intersection of language features supported by
  the given compilers.

  Four compilers are supported. The transforms are enabled according to this
  table:

  ===============  =====  ===  ====  ====
  Transforms       clang  gcc  icc   mscv
  ===============  =====  ===  ====  ====
  AddOverride (1)  3.0    4.7  14    8
  LoopConvert      3.0    4.6  13    11
  PassByValue      3.0    4.6  13    11
  ReplaceAutoPtr   3.0    4.6  13    11
  UseAuto          2.9    4.4  12    10
  UseNullptr       3.0    4.6  12.1  10
  ===============  =====  ===  ====  ====

  (1): if *-override-macros* is provided it's assumed that the macros are C++11
  aware and the transform is enabled without regard to the supported compilers.

  The structure of the argument to the `-for-compilers` option is
  **<compiler>-<major ver>[.<minor ver>]** where **<compiler>** is one of the
  compilers from the above table.

  Some examples:

  1. To support `Clang >= 3.0`, `gcc >= 4.6` and `MSVC >= 11`:

     ``clang-modernize -for-compilers=clang-3.0,gcc-4.6,msvc-11 <args..>``

     Enables LoopConvert, ReplaceAutoPtr, UseAuto, UseNullptr.

  2. To support `icc >= 12` while using a C++11-aware macro for the `override`
     virtual specifier:

     ``clang-modernize -for-compilers=icc-12 -override-macros <args..>``

     Enables AddOverride and UseAuto.

  .. warning::

    If your version of Clang depends on the GCC headers (e.g: when `libc++` is
    not used), then you probably want to add the GCC version to the targeted
    platforms as well.

.. option:: -perf[=<directory>]

  Turns on performance measurement and output functionality. The time it takes to
  apply each transform is recorded by the migrator and written in JSON format
  to a uniquely named file in the given ``<directory>``. All sources processed
  by a single Modernizer process are written to the same output file. If
  ``<directory>`` is not provided the default is ``./migrate_perf/``.

  The time recorded for a transform includes parsing and creating source code
  replacements.

.. option:: -serialize-replacements

  Causes the modernizer to generate replacements and serialize them to disk but
  not apply them. This can be useful for debugging or for manually running
  ``clang-apply-replacements``. Replacements are serialized in YAML_ format.
  By default serialzied replacements are written to a temporary directory whose
  name is written to stderr when serialization is complete.

.. _YAML: http://www.yaml.org/

.. option:: -serialize-dir=<string>

  Choose a directory to serialize replacements to. The directory must exist.

.. _include/exclude options:

Path Inclusion/Exclusion Options
================================

.. option:: -include=<path1>,<path2>,...,<pathN>

  Use this option to indicate which directories contain files that can be
  changed by the modernizer. Inidividual files may be specified if desired.
  Multiple paths can be specified as a comma-separated list. Sources mentioned
  explicitly on the command line are always included so this option controls
  which other files (e.g. headers) may be changed while transforming
  translation units.

.. option:: -exclude=<path1>,<path2>,...,<pathN>

  Used with ``-include`` to provide finer control over which files and
  directories can be transformed. Individual files and files within directories
  specified by this option **will not** be transformed. Multiple paths can be
  specified as a comma-separated list.

.. option:: -include-from=<filename>

  Like ``-include`` but read paths from the given file. Paths should be one per
  line.

.. option:: -exclude-from=<filename>

  Like ``-exclude`` but read paths from the given file. Paths are listed one
  per line.

Formatting Command Line Options
===============================

.. option:: -format

  Enable reformatting of code changed by transforms. Formatting is done after
  every transform.

.. option:: -style=<string>

  Specifies how formatting should be done. The behaviour of this option is
  identical to the same option provided by clang-format_. Refer to
  `clang-format's style options`_ for more details.

.. option:: -style-config=<dir>

  When using ``-style=file``, the default behaviour is to look for
  ``.clang-format`` starting in the current directory and then in ancestors. To
  specify a directory to find the style configuration file, use this option.

Example:

.. code-block:: c++
  :emphasize-lines: 10-12,18

    // file.cpp
    for (std::vector<int>::const_iterator I = my_container.begin(),
                                          E = my_container.end();
         I != E; ++I) {
      std::cout << *I << std::endl;
    }

    // No reformatting:
    //     clang-modernize -use-auto file.cpp
    for (auto I = my_container.begin(),
                                          E = my_container.end();
         I != E; ++I) {
      std::cout << *I << std::endl;
    }

    // With reformatting enabled:
    //     clang-modernize -format -use-auto file.cpp
    for (auto I = my_container.begin(), E = my_container.end(); I != E; ++I) {
      std::cout << *I << std::endl;
    }

.. _clang-format: http://clang.llvm.org/docs/ClangFormat.html
.. _clang-format's style options: http://clang.llvm.org/docs/ClangFormatStyleOptions.html


.. _transform-specific-command-line-options:

Transform-Specific Command Line Options
=======================================

.. option:: -loop-convert

  Makes use of C++11 range-based for loops where possible. See
  :doc:`LoopConvertTransform`.

.. option:: -use-nullptr

  Makes use of the new C++11 keyword ``nullptr`` where possible.
  See :doc:`UseNullptrTransform`.

.. option:: -user-null-macros=<string>

  ``<string>`` is a comma-separated list of user-defined macros that behave like
  the ``NULL`` macro. The :option:`-use-nullptr` transform will replace these
  macros along with ``NULL``. See :doc:`UseNullptrTransform`.

.. option:: -use-auto

  Replace the type specifier of variable declarations with the ``auto`` type
  specifier. See :doc:`UseAutoTransform`.

.. option:: -add-override

  Adds the override specifier to member functions where it is appropriate. That
  is, the override specifier is added to member functions that override a
  virtual function in a base class and that don't already have the specifier.
  See :doc:`AddOverrideTransform`.

.. option:: -override-macros

  Tells the Add Override Transform to locate a macro that expands to
  ``override`` and use that macro instead of the ``override`` keyword directly.
  If no such macro is found, ``override`` is still used. This option enables
  projects that use such macros to maintain build compatibility with non-C++11
  code.

.. option:: -pass-by-value

  Replace const-reference parameters by values in situations where it can be
  beneficial.
  See :doc:`PassByValueTransform`.

.. option:: -replace-auto_ptr

  Replace ``std::auto_ptr`` (deprecated in C++11) by ``std::unique_ptr`` and
  wrap calls to the copy constructor and assignment operator with
  ``std::move()``.
  See :doc:`ReplaceAutoPtrTransform`.
