=======
Modules
=======

.. contents::
   :local:

Introduction
============
Most software is built using a number of software libraries, including libraries supplied by the platform, internal libraries built as part of the software itself to provide structure, and third-party libraries. For each library, one needs to access both its interface (API) and its implementation. In the C family of languages, the interface to a library is accessed by including the appropriate header files(s):

.. code-block:: c

  #include <SomeLib.h>

The implementation is handled separately by linking against the appropriate library. For example, by passing ``-lSomeLib`` to the linker.

Modules provide an alternative, simpler way to use software libraries that provides better compile-time scalability and eliminates many of the problems inherent to using the C preprocessor to access the API of a library.

Problems with the current model
-------------------------------
The ``#include`` mechanism provided by the C preprocessor is a very poor way to access the API of a library, for a number of reasons:

* **Compile-time scalability**: Each time a header is included, the
  compiler must preprocess and parse the text in that header and every
  header it includes, transitively. This process must be repeated for
  every translation unit in the application, which involves a huge
  amount of redundant work. In a project with *N* translation units
  and *M* headers included in each translation unit, the compiler is
  performing *M x N* work even though most of the *M* headers are
  shared among multiple translation units. C++ is particularly bad,
  because the compilation model for templates forces a huge amount of
  code into headers.

* **Fragility**: ``#include`` directives are treated as textual
  inclusion by the preprocessor, and are therefore subject to any  
  active macro definitions at the time of inclusion. If any of the 
  active macro definitions happens to collide with a name in the 
  library, it can break the library API or cause compilation failures 
  in the library header itself. For an extreme example, 
  ``#define std "The C++ Standard"`` and then include a standard  
  library header: the result is a horrific cascade of failures in the
  C++ Standard Library's implementation. More subtle real-world
  problems occur when the headers for two different libraries interact
  due to macro collisions, and users are forced to reorder
  ``#include`` directives or introduce ``#undef`` directives to break
  the (unintended) dependency.

* **Conventional workarounds**: C programmers have
  adopted a number of conventions to work around the fragility of the
  C preprocessor model. Include guards, for example, are required for
  the vast majority of headers to ensure that multiple inclusion
  doesn't break the compile. Macro names are written with
  ``LONG_PREFIXED_UPPERCASE_IDENTIFIERS`` to avoid collisions, and some
  library/framework developers even use ``__underscored`` names
  in headers to avoid collisions with "normal" names that (by
  convention) shouldn't even be macros. These conventions are a
  barrier to entry for developers coming from non-C languages, are
  boilerplate for more experienced developers, and make our headers
  far uglier than they should be.

* **Tool confusion**: In a C-based language, it is hard to build tools
  that work well with software libraries, because the boundaries of
  the libraries are not clear. Which headers belong to a particular
  library, and in what order should those headers be included to
  guarantee that they compile correctly? Are the headers C, C++,
  Objective-C++, or one of the variants of these languages? What
  declarations in those headers are actually meant to be part of the
  API, and what declarations are present only because they had to be
  written as part of the header file?

Semantic import
---------------
Modules improve access to the API of software libraries by replacing the textual preprocessor inclusion model with a more robust, more efficient semantic model. From the user's perspective, the code looks only slightly different, because one uses an ``import`` declaration rather than a ``#include`` preprocessor directive:

.. code-block:: c

  import std.io; // pseudo-code; see below for syntax discussion

However, this module import behaves quite differently from the corresponding ``#include <stdio.h>``: when the compiler sees the module import above, it loads a binary representation of the ``std.io`` module and makes its API available to the application directly. Preprocessor definitions that precede the import declaration have no impact on the API provided by ``std.io``, because the module itself was compiled as a separate, standalone module. Additionally, any linker flags required to use the ``std.io`` module will automatically be provided when the module is imported [#]_
This semantic import model addresses many of the problems of the preprocessor inclusion model:

* **Compile-time scalability**: The ``std.io`` module is only compiled once, and importing the module into a translation unit is a constant-time operation (independent of module system). Thus, the API of each software library is only parsed once, reducing the *M x N* compilation problem to an *M + N* problem.

* **Fragility**: Each module is parsed as a standalone entity, so it has a consistent preprocessor environment. This completely eliminates the need for ``__underscored`` names and similarly defensive tricks. Moreover, the current preprocessor definitions when an import declaration is encountered are ignored, so one software library can not affect how another software library is compiled, eliminating include-order dependencies.

* **Tool confusion**: Modules describe the API of software libraries, and tools can reason about and present a module as a representation of that API. Because modules can only be built standalone, tools can rely on the module definition to ensure that they get the complete API for the library. Moreover, modules can specify which languages they work with, so, e.g., one can not accidentally attempt to load a C++ module into a C program.

Problems modules do not solve
-----------------------------
Many programming languages have a module or package system, and because of the variety of features provided by these languages it is important to define what modules do *not* do. In particular, all of the following are considered out-of-scope for modules:

* **Rewrite the world's code**: It is not realistic to require applications or software libraries to make drastic or non-backward-compatible changes, nor is it feasible to completely eliminate headers. Modules must interoperate with existing software libraries and allow a gradual transition.

* **Versioning**: Modules have no notion of version information. Programmers must still rely on the existing versioning mechanisms of the underlying language (if any exist) to version software libraries.

* **Namespaces**: Unlike in some languages, modules do not imply any notion of namespaces. Thus, a struct declared in one module will still conflict with a struct of the same name declared in a different module, just as they would if declared in two different headers. This aspect is important for backward compatibility, because (for example) the mangled names of entities in software libraries must not change when introducing modules.

* **Binary distribution of modules**: Headers (particularly C++ headers) expose the full complexity of the language. Maintaining a stable binary module format across architectures, compiler versions, and compiler vendors is technically infeasible.

Supported syntaxes
==================

Clang supports four different source-level syntaxes for modules, with full
interoperability between the different syntaxes. They are:

 * :ref:`Objective-C import declaration`\s
 * :ref:`C++ Modules TS syntax`
 * :ref:`#include translation <includes-as-imports>`
 * :ref:`#pragma syntax <module pragmas>`

All syntaxes other than the Modules TS syntax are enabled by ``-fmodules``.
The C++ Modules TS syntax is enabled by ``-fmodules-ts``.
Additional `command-line parameters`_ are described in a separate section later.

At present, there is no first-class C syntax for import declarations.

Objective-C import declaration
------------------------------
Objective-C provides syntax for importing a module via an *@import declaration*, which imports the named module:

.. parsed-literal::

  @import std;

The ``@import`` declaration above imports the entire contents of the ``std`` module (which would contain, e.g., the entire C or C++ standard library) and make its API available within the current translation unit. To import only part of a module, one may use dot syntax to specific a particular submodule, e.g.,

.. parsed-literal::

  @import std.io;

Redundant import declarations are ignored, and one is free to import modules at any point within the translation unit, so long as the import declaration is at global scope.

C++ Modules TS syntax
---------------------
Clang supports the module syntax described in the draft C++ Technical
Specification for Modules. Modules can be imported using the syntax:

.. parsed-literal::

  import std;

.. warning::

  The C++ Modules TS and Clang's other supported module systems use two
  different naming systems for modules. Outside the Modules TS, a module
  name ``foo.bar.baz`` describes the ``bar.baz`` submodule of the ``foo``
  ("top-level") module, where a top-level module is a single compiled entity
  that is intended to represent the interface of a complete library, and
  submodules describe partitions of the module whose visibility can be
  separately controlled through ``import``\s.
  Within the Modules TS, there is no support for subdividing or grouping the
  interface of a library, and the name ``foo.bar.baz`` is effectively a
  top-level module name, naming a module that has no relation to modules
  ``foo`` or ``foo.bar`` (if such modules exist).

  This means there are two different ways to interpret a declaration such
  as ``import std.io;`` if the Modules TS is enabled: as either the submodule
  ``io`` of the top-level module ``std``, or as the top-level module
  ``std.io``. Clang currently only considers the Modules TS interpretation
  when using the Modules TS import syntax, so can only import top-level
  modules. This is subject to change.

Includes as imports
-------------------
The primary user-level feature of modules is the import operation, which provides access to the API of software libraries. However, today's programs make extensive use of ``#include``, and it is unrealistic to assume that all of this code will change overnight. Instead, modules can automatically translate ``#include`` directives into the corresponding module import. For example, the include directive

.. code-block:: c

  #include <stdio.h>

will be automatically mapped to an import of the module ``std.io``. Even with specific ``import`` syntax in the language, this particular feature is important for both adoption and backward compatibility: automatic translation of ``#include`` to ``import`` allows an application to get the benefits of modules (for all modules-enabled libraries) without any changes to the application itself. Thus, users can easily use modules with one compiler while falling back to the preprocessor-inclusion mechanism with other compilers.

While building a module, ``#include_next`` is also supported, with one caveat.
The usual behavior of ``#include_next`` is to search for the specified filename
in the list of include paths, starting from the path *after* the one
in which the current file was found.
Because files listed in module maps are not found through include paths, a
different strategy is used for ``#include_next`` directives in such files: the
list of include paths is searched for the specified header name, to find the
first include path that would refer to the current file. ``#include_next`` is
interpreted as if the current file had been found in that path.
If this search finds a file named by a module map, the ``#include_next``
directive is translated into an import, just like for a ``#include``
directive.

``#include``\s are mapped to module imports using :doc:`module maps <ModuleMaps>`.

Compilation models
==================

Modules can be built implicitly (on demand), or explicitly, as described below.
Implicit module builds integrate more smoothly into existing build systems, but
are not well-suited to distributed compilation or highly-parallel builds.
Explicit module builds are more flexible but require more work to configure.

Implicit module builds
----------------------
The binary representation of modules is automatically generated by the compiler on an as-needed basis. When a module is imported (e.g., by an ``#include`` of one of the module's headers), the compiler will spawn a second instance of itself [#]_, with a fresh preprocessing context [#]_, to parse just the headers in that module. The resulting Abstract Syntax Tree (AST) is then persisted into the binary representation of the module that is then loaded into translation unit where the module import was encountered.

The binary representation of modules is persisted in the *module cache*. Imports of a module will first query the module cache and, if a binary representation of the required module is already available, will load that representation directly. Thus, a module's headers will only be parsed once per language configuration, rather than once per translation unit that uses the module.

Modules maintain references to each of the headers that were part of the module build. If any of those headers changes, or if any of the modules on which a module depends change, then the module will be (automatically) recompiled. The process should never require any user intervention.

Implicit module builds are currently only supported for modules described by :doc:`module maps <ModuleMaps>`, and not for :doc:`C++ Modules TS modules <ModulesTS>`.

Explicit module builds
----------------------
The compilation action that implicit module builds invoke to compile a module can also be invoked explicitly. The module map or module interface unit is provided as an input to the compilation action, and the ``--precompile`` flag is used to indicate that compilation should stop after producing the AST. As with an implicit module build, the output file is a serialized form of the compiled AST.

In this model, there is no module cache, so compiled module files must be passed to downstream compilations in another way. Typically, this is achieved by use of either the ``-fmodule-file=`` parameter or the ``-fprebuit-module-path=`` parameter, which are described below.

Unlike for implicit module builds, explicitly compiled modules are not rebuilt automatically if their inputs change; instead, an attempt to use a module that is out of date will result in an error. However, some differences between configuration are permitted between the build of an explicitly-built module and the use of the module -- for example, different warning settings and macro definitions are allowed, as are changes to some minor language settings. The settings in force when a module was compiled govern its behavior.

.. warning::

  Explicitly compiling a module from a module map currently requires the use of
  ``-cc1`` flags, which are intentionally undocumented and subject to change at
  any time, because the "module map" input type is not currently exposed.

.. _modules-command-line-parameters:

Command-line parameters
=======================
``-fmodules``
  Enable the modules feature. Implies ``-fimplicit-modules`` and ``-fimplicit-module-maps``.

``-fmodules-ts``
  Enable support for the C++ Modules TS syntax and its semantic model.

``-module-file-info <module file name>``
  Debugging aid that prints information about a given module file (with a ``.pcm`` extension), including the language and preprocessor options that particular module variant was built with.

Parameters controlling module maps
----------------------------------

``-fimplicit-module-maps``
  Enable implicit search for module map files named ``module.modulemap`` and similar. This option is implied by ``-fmodules``. If this is disabled with ``-fno-implicit-module-maps``, module map files will only be loaded if they are explicitly specified via ``-fmodule-map-file`` or transitively used by another module map file.

``-fmodule-map-file=<file>``
  Load the given module map file if a header from its directory or one of its subdirectories is loaded.

``-fbuiltin-module-map``
  Load the Clang builtins module map file. (Equivalent to ``-fmodule-map-file=<resource dir>/include/module.modulemap``)

Parameters controlling module compilation mode
----------------------------------------------

``-fimplicit-modules``
  Implicitly build modules using the module cache. This option is implied by ``-fmodules``. If this is disabled with ``-fno-implicit-modules``, all modules used by the build must be specified with ``-fmodule-file`` or ``-fprebuilt-module-path``.

``-fmodules-cache-path=<directory>``
  Specify the path to the modules cache. If not provided, Clang will select a system-appropriate default.

``-fmodules-ignore-macro=macroname``
  Instruct modules to ignore the named macro when selecting an appropriate module variant. Use this for macros defined on the command line that don't affect how modules are built, to improve sharing of compiled module files.

``-fmodules-prune-interval=seconds``
  Specify the minimum delay (in seconds) between attempts to prune the module cache. Module cache pruning attempts to clear out old, unused module files so that the module cache itself does not grow without bound. The default delay is large (604,800 seconds, or 7 days) because this is an expensive operation. Set this value to 0 to turn off pruning.

``-fmodules-prune-after=seconds``
  Specify the minimum time (in seconds) for which a file in the module cache must be unused (according to access time) before module pruning will remove it. The default delay is large (2,678,400 seconds, or 31 days) to avoid excessive module rebuilding.

``-fmodule-file=<file>``
  Load the given precompiled module file.

``-fprebuilt-module-path=<directory>``
  Specify the path to the prebuilt modules. If specified, we will look for modules in this directory for a given top-level module name. We don't need a module map for loading prebuilt modules in this directory and the compiler will not try to rebuild these modules. This can be specified multiple times.

Parameters controlling module semantics
---------------------------------------

``-fno-autolink``
  Disable automatic linking against the libraries associated with imported modules.

``-fmodules-decluse``
``-fmodules-strict-decluse``
  Enable checking of module ``use`` declarations, which requires dependencies between modules to be explicitly declared in module maps. With ``-fmodules-strict-decluse``, ``#include``\s of headers not within any module are also rejected.

``-fmodule-name=module-id``
  Consider the compiled source file to be a part of the given module.

``-fmodules-search-all``
  If a symbol is not found, search modules referenced in the current module maps but not imported for symbols, so the error message can reference the module by name.  Note that if the global module index has not been built before, this might take some time as it needs to build all the modules.  Note that this option doesn't apply in module builds, to avoid the recursion.

Future Directions
=================
Modules support is under active development, and there are many opportunities remaining to improve it. Here are a few ideas:

**Detect unused module imports**
  Unlike with ``#include`` directives, it should be fairly simple to track whether a directly-imported module has ever been used. By doing so, Clang can emit ``unused import`` or ``unused #include`` diagnostics, including Fix-Its to remove the useless imports/includes.

**Fix-Its for missing imports**
  It's fairly common for one to make use of some API while writing code, only to get a compiler error about "unknown type" or "no function named" because the corresponding header has not been included. Clang can detect such cases and auto-import the required module, but should provide a Fix-It to add the import.

**Improve modularize**
  The modularize tool is both extremely important (for deployment) and extremely crude. It needs better UI, better detection of problems (especially for C++), and perhaps an assistant mode to help write module maps for you.

Where To Learn More About Modules
=================================
The Clang source code provides additional information about modules:

``clang/lib/Headers/module.modulemap``
  Module map for Clang's compiler-specific header files.

``clang/test/Modules/``
  Tests specifically related to modules functionality.

``clang/include/clang/Basic/Module.h``
  The ``Module`` class in this header describes a module, and is used throughout the compiler to implement modules.

``clang/include/clang/Lex/ModuleMap.h``
  The ``ModuleMap`` class in this header describes the full module map, consisting of all of the module map files that have been parsed, and providing facilities for looking up module maps and mapping between modules and headers (in both directions).

PCHInternals_
  Information about the serialized AST format used for precompiled headers and modules. The actual implementation is in the ``clangSerialization`` library.

.. [#] Automatic linking against the libraries of modules requires specific linker support, which is not widely available.

.. [#] The second instance is actually a new thread within the current process, not a separate process. However, the original compiler instance is blocked on the execution of this thread.

.. [#] The preprocessing context in which the modules are parsed is actually dependent on the command-line options provided to the compiler, including the language dialect and any ``-D`` options. However, the compiled modules for different command-line options are kept distinct, and any preprocessor directives that occur within the translation unit are ignored. See the section on the `Configuration macros declaration`_ for more information.

.. _PCHInternals: PCHInternals.html
 
