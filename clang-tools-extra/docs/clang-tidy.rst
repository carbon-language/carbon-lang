==========
Clang-Tidy
==========

:program:`clang-tidy` is a clang-based C++ linter tool. Its purpose is to
provide an extensible framework for diagnosing and fixing typical programming
errors, like style violations, interface misuse, or bugs that can be deduced via
static analysis. :program:`clang-tidy` is modular and provides a convenient
interface for writing new checks.


Using clang-tidy
================

:program:`clang-tidy` is a `LibTooling`_-based tool, and it's easier to work
with if you set up a compile command database for your project (for an example
of how to do this see `How To Setup Tooling For LLVM`_ ). You can also specify
compilation options on the command line after ``--``:

.. code-block:: bash

  $ clang-tidy test.cpp -- -Imy_project/include -DMY_DEFINES ...

:program:`clang-tidy` has its own checks and can also run Clang static analyzer
checks. Each check has a name and the checks to run can be chosen using the
``-checks=`` and ``-disable-checks=`` options. :program:`clang-tidy` selects the
checks with names matching the regular expression specified by the ``-checks=``
option and not matching the one specified by the ``-disable-checks=`` option.

The ``-list-checks`` option lists all the enabled checks. It can be used with or
without ``-checks=`` and/or ``-disable-checks=``.

There are currently three groups of checks:

* Checks related to the LLVM coding conventions have names starting with
  ``llvm-``.

* Checks related to the Google coding conventions have names starting with
  ``google-``.

* Clang static analyzer checks are named starting with ``clang-analyzer-``.


The ``-fix`` flag instructs :program:`clang-format` to fix found errors if
supported by corresponding checks.

An overview of all the command-line options:

.. code-block:: bash

  $ clang-tidy -help
  USAGE: clang-tidy [options] <source0> [... <sourceN>]

  OPTIONS:

  General options:

    -help                    - Display available options (-help-hidden
                               for more)
    -help-list               - Display list of available options
                               (-help-list-hidden for more)
    -version                 - Display the version of this program

  clang-tidy options:

    -checks=<string>         - Regular expression matching the names of
                               the checks to be run.
    -disable-checks=<string> - Regular expression matching the names of
                               the checks to disable.
    -fix                     - Fix detected errors if possible.
    -list-checks             - List all enabled checks and exit.
    -p=<string>              - Build path

  -p <build-path> is used to read a compile command database.

    For example, it can be a CMake build directory in which a file named
    compile_commands.json exists (use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    CMake option to get this output). When no build path is specified,
    a search for compile_commands.json will be attempted through all
    parent paths of the first input file . See:
    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html for an
    example of setting up Clang Tooling on a source tree.

  <source0> ... specify the paths of source files. These paths are
    looked up in the compile command database. If the path of a file is
    absolute, it needs to point into CMake's source tree. If the path is
    relative, the current working directory needs to be in the CMake
    source tree and the file must be in a subdirectory of the current
    working directory. "./" prefixes in the relative files will be
    automatically removed, but the rest of a relative path must be a
    suffix of a path in the compile command database.


.. _LibTooling: http://clang.llvm.org/docs/LibTooling.html
.. _How To Setup Tooling For LLVM: http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html


Getting Involved
================

:program:`clang-tidy` has several own checks and can run Clang static analyzer
checks, but its power is in the ability to easily write custom checks.

Checks are organized in modules, which can be linked into :program:`clang-tidy`
with minimal or no code changes in clang-tidy.

Checks can plug the analysis on the preprocessor level using `PPCallbacks`_ or
on the AST level using `AST Matchers`_. When an error is found, checks can
report them in a way similar to how Clang diagnostics work. A fix-it hint can be
attached to a diagnostic message.

The interface provided by clang-tidy makes it easy to write useful and precise
checks in just a few lines of code. If you have an idea for a good check, the
rest of this document explains how to do this.

.. _AST Matchers: http://clang.llvm.org/docs/LibASTMatchers.html
.. _PPCallbacks: http://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html


Choosing the Right Place for your Check
---------------------------------------

If you have an idea of a check, you should decide whether it should be
implemented as a:

+ *Clang diagnostic*: if the check is generic enough, targets code patterns that
  most probably are bugs (rather than style or readability issues), can be
  implemented effectively and with extremely low false positive rate, it may
  make a good Clang diagnostic.

+ *Clang static analyzer check*: if the check requires some sort of control flow
  analysis, it should probably be implemented as a static analyzer check.

+ *clang-tidy check* is a good choice for linter-style checks, checks that are
  related to a certain coding style, checks that address code readability, etc.


Preparing your Workspace
------------------------

If you are new to LLVM development, you should read the `Getting Started with
the LLVM System`_, `Using Clang Tools`_ and `How To Setup Tooling For LLVM`_
documents to check out and build LLVM, Clang and Clang Extra Tools with CMake.

Once you are done, change to the ``llvm/tools/clang/tools/extra`` directory, and
let's start!

.. _Getting Started with the LLVM System: http://llvm.org/docs/GettingStarted.html
.. _Using Clang Tools: http://clang.llvm.org/docs/ClangTools.html


The Directory Structure
-----------------------

:program:`clang-tidy` source code resides in the
``llvm/tools/clang/tools/extra`` directory and is structured as follows:

::

  clang-tidy/                       # Clang-tidy core.
  ├── ClangTidy.h                   # Interfaces for users and checks.
  ├── ClangTidyModule.h             # Interface for clang-tidy modules.
  ├── ClangTidyModuleRegistry.h     # Interface for registering of modules.
     ...
  ├── google/                       # Google clang-tidy module.
  │   ├── GoogleTidyModule.cpp
  │   ├── GoogleTidyModule.h
          ...
  ├── llvm/                         # LLVM clang-tidy module.
  │   ├── LLVMTidyModule.cpp
  │   ├── LLVMTidyModule.h
          ...
  └── tool/                         # Sources of the clang-tidy binary.
          ...
  test/clang-tidy/                  # Integration tests.
      ...
  unittests/clang-tidy/
  ├── ClangTidyTest.h
  ├── GoogleModuleTest.cpp
  ├── LLVMModuleTest.cpp
      ...


Writing a clang-tidy Check
--------------------------

So you have an idea of a useful check for :program:`clang-tidy`.

You need to decide which module the check belongs to. If the check verifies
conformance of the code to a certain coding style, it probably deserves a
separate module and a directory in ``clang-tidy/`` (there are LLVM and Google
modules already).

After choosing the module, you need to create a class for your check:

.. code-block:: c++

  #include "../ClangTidy.h"

  namespace clang {
  namespace tidy {

  class MyCheck : public ClangTidyCheck {
  };

  } // namespace tidy
  } // namespace clang

Next, you need to decide whether it should operate on the preprocessor level or
on the AST level. Let's imagine that we need to work with the AST in our check.
In this case we need to override two methods:

.. code-block:: c++

  ...
  class ExplicitConstructorCheck : public ClangTidyCheck {
  public:
    void registerMatchers(ast_matchers::MatchFinder *Finder);
    void check(ast_matchers::MatchFinder::MatchResult &Result);
  };

In the ``registerMatchers`` method we create an AST Matcher (see `AST Matchers`_
for more information) that will find the pattern in the AST that we want to
inspect. The results of the matching are passed to the ``check`` method, which
can further inspect them and report diagnostics.

.. code-block:: c++

  using namespace ast_matchers;

  void ExplicitConstructorCheck::registerMatchers(MatchFinder *Finder) {
    Finder->addMatcher(constructorDecl().bind("ctor"), this);
  }

  void ExplicitConstructorCheck::check(const MatchFinder::MatchResult &Result) {
    const CXXConstructorDecl *Ctor =
        Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
    // Do not be confused: isExplicit means 'explicit' keyword is present,
    // isImplicit means that it's a compiler-generated constructor.
    if (Ctor->isOutOfLine() || Ctor->isExplicit() || Ctor->isImplicit())
      return;
    if (Ctor->getNumParams() == 0 || Ctor->getMinRequiredArguments() > 1)
      return;
    SourceLocation Loc = Ctor->getLocation();
    diag(Loc, "Single-argument constructors must be explicit")
        << FixItHint::CreateInsertion(Loc, "explicit ");
  }

(The full code for this check resides in
``clang-tidy/google/GoogleTidyModule.cpp``).


Registering your Check
----------------------

The check should be registered in the corresponding module with a distinct name:

.. code-block:: c++

  class MyModule : public ClangTidyModule {
   public:
    virtual void
    addCheckFactories(ClangTidyCheckFactories &CheckFactories) LLVM_OVERRIDE {
      CheckFactories.addCheckFactory(
          "my-explicit-constructor",
          new ClangTidyCheckFactory<ExplicitConstructorCheck>());
    }
  };

Now we need to register the module in the ``ClangTidyModuleRegistry`` using a
statically initialized variable:

.. code-block:: c++

  static ClangTidyModuleRegistry::Add<MyModule> X("my-module",
                                                  "Adds my lint checks.");


When using LLVM build system, we need to use the following hack to ensure the
module is linked into the clang-tidy binary:

Add this near the ``ClangTidyModuleRegistry::Add<MyModule>`` variable:

.. code-block:: c++

  // This anchor is used to force the linker to link in the generated object file
  // and thus register the MyModule.
  volatile int MyModuleAnchorSource = 0;

And this to the main translation unit of the clang-tidy binary (or the binary
you link the ``clang-tidy`` library in) ``clang-tidy/tool/ClangTidyMain.cpp``:

.. code-block:: c++

  // This anchor is used to force the linker to link the MyModule.
  extern volatile int MyModuleAnchorSource;
  static int MyModuleAnchorDestination = MyModuleAnchorSource;

