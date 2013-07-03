.. index:: cpp11-migrate

============================
C++11 Migrator User's Manual
============================

.. toctree::
   :hidden:

   UseAutoTransform
   UseNullptrTransform
   LoopConvertTransform
   AddOverrideTransform
   ReplaceAutoPtrTransform
   MigratorUsage

:program:`cpp11-migrate` is a standalone tool used to automatically convert
C++98 and C++03 code to use features of the new C++11 standard where
appropriate.

Getting Started
===============

To build from source:

1. Read `Getting Started with the LLVM System`_ and `Clang Tools
   Documentation`_ for information on getting sources for LLVM, Clang, and
   Clang Extra Tools.

2. `Getting Started with the LLVM System`_ and `Building LLVM with CMake`_ give
   directions for how to build. With sources all checked out into the
   right place the LLVM build will build Clang Extra Tools and their
   dependencies automatically.

   * If using CMake, you can also use the ``cpp11-migrate`` target to build
     just the Migrator and its dependencies.

Before continuing, take a look at :doc:`MigratorUsage` to see how to invoke the
Migrator.

Before running the Migrator on code you'll need the arguments you'd normally
pass to the compiler. If you're migrating a single file with few compiler
arguments, it might be easier to pass the compiler args on the command line
after ``--``. If you're working with multiple files or even a single file
with many compiler args, it's probably best to use a *compilation database*.

A `compilation database`_ contains the command-line arguments for multiple
files. If the code you want to transform can be built with CMake, you can
generate this database easily by running CMake with the
``-DCMAKE_EXPORT_COMPILE_COMMANDS`` option. The Ninja_ build system, since
v1.2, can create this file too using the *compdb* tool: ``ninja -t compdb``. If
you're not already using either of these tools or cannot easily make use of
them you might consider looking into Bear_.

In addition to the compiler arguments you usually build your code with, you must
provide the option for enabling C++11 features. For clang and versions of gcc
≥ v4.8 this is ``-std=c++11``.

Now with compiler arguments, the Migrator can be applied to source. Sources are
transformed in place and changes are only written to disk if compilation errors
aren't caused by the transforms. Each transform will re-parse the output from
the previous transform. The output from the last transform is not checked
unless ``-final-syntax-check`` is enabled.


.. _Ninja: http://martine.github.io/ninja/
.. _Bear: https://github.com/rizsotto/Bear
.. _compilation database: http://clang.llvm.org/docs/JSONCompilationDatabase.html
.. _Getting Started with the LLVM System: http://llvm.org/docs/GettingStarted.html
.. _Building LLVM with CMake: http://llvm.org/docs/CMake.html
.. _Clang Tools Documentation: http://clang.llvm.org/docs/ClangTools.html

Getting Involved
================

If you find a bug

.. raw:: html

  <input type="button" id="logbug" value="Log a Bug!" />
  <script type="text/javascript" src="https://cpp11-migrate.atlassian.net/s/en_USpfg3b3-1988229788/6095/34/1.4.0-m2/_/download/batch/com.atlassian.jira.collector.plugin.jira-issue-collector-plugin:issuecollector/com.atlassian.jira.collector.plugin.jira-issue-collector-plugin:issuecollector.js?collectorId=50813874"></script>
  <script type="text/javascript">window.ATL_JQ_PAGE_PROPS =  {
    "triggerFunction": function(showCollectorDialog) {
      //Requries that jQuery is available! 
      jQuery("#logbug").click(function(e) {
        e.preventDefault();
        showCollectorDialog();
      });
    }};
  </script>

Bugs and feature development of the Migrator are tracked at
http://cpp11-migrate.atlassian.net. If you want to get involved the front page
shows a list of outstanding issues or you can browse around the project to get
familiar. To take on issues or contribute feature requests and/or bug reports
you need to sign up for an account from the `log in page`_. An account also
lets you sign up for notifications on issues or vote for unassigned issues to
be completed.

.. _log in page: https://cpp11-migrate.atlassian.net/login

.. _transforms:

Transformations
===============

The Migrator is a collection of independent transforms which can be
independently enabled. The transforms currently implemented are:

* :doc:`LoopConvertTransform`

* :doc:`UseNullptrTransform`

* :doc:`UseAutoTransform`

* :doc:`AddOverrideTransform`

* :doc:`ReplaceAutoPtrTransform`
