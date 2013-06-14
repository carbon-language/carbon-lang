===================
cpp11-migrate Usage
===================

``cpp11-migrate [options] <source0> [... <sourceN>] [-- [args]]``

``<source#>`` specifies the path to the source to migrate. This path may be
relative to the current directory.

At least one transform must be enabled.

General Command Line Options
============================

.. option:: -help

  Displays tool usage instructions and command line options.

.. option:: -version

  Displays the version information of this tool.

.. option:: -p[=<build-path>]

  ``<build-path>`` is the directory containing a file named
  ``compile_commands.json`` which provides compiler arguments for building each
  source file. CMake can generate this file by specifying
  ``-DCMAKE_EXPORT_COMPILE_COMMANDS`` when running CMake. Ninja_, since v1.2
  can also generate this file with ``ninja -t compdb``. If ``<build-path>`` is
  not provided the ``compile_commands.json`` file is searched for through all
  parent directories.

.. option:: -- [args]

  Another way to provide compiler arguments is to specify all arguments on the
  command line following ``--``. Arguments provided this way are used for
  *every* source file.
  
  If ``-p`` is not specified, ``--`` is necessary, even if no compiler
  arguments are required.

.. _Ninja: http://martine.github.io/ninja/

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

.. option:: -perf[=<directory>]

  Turns on performance measurement and output functionality. The time it takes to
  apply each transform is recorded by the migrator and written in JSON format
  to a uniquely named file in the given ``<directory>``. All sources processed
  by a single Migrator process are written to the same output file. If ``<directory>`` is
  not provided the default is ``./migrate_perf/``.

  The time recorded for a transform includes parsing and creating source code
  replacements.

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

