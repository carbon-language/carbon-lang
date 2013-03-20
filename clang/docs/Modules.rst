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

Problems with the Current Model
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

Semantic Import
---------------
Modules improve access to the API of software libraries by replacing the textual preprocessor inclusion model with a more robust, more efficient semantic model. From the user's perspective, the code looks only slightly different, because one uses an ``import`` declaration rather than a ``#include`` preprocessor directive:

.. code-block:: c

  import std.io; // pseudo-code; see below for syntax discussion

However, this module import behaves quite differently from the corresponding ``#include <stdio.h>``: when the compiler sees the module import above, it loads a binary representation of the ``std.io`` module and makes its API available to the application directly. Preprocessor definitions that precede the import declaration have no impact on the API provided by ``std.io``, because the module itself was compiled as a separate, standalone module. Additionally, any linker flags required to use the ``std.io`` module will automatically be provided when the module is imported [#]_
This semantic import model addresses many of the problems of the preprocessor inclusion model:

* **Compile-time scalability**: The ``std.io`` module is only compiled once, and importing the module into a translation unit is a constant-time operation (independent of module system). Thus, the API of each software library is only parsed once, reducing the *M x N* compilation problem to an *M + N* problem.

* **Fragility**: Each module is parsed as a standalone entity, so it has a consistent preprocessor environment. This completely eliminates the need for ``__underscored`` names and similarly defensive tricks. Moreover, the current preprocessor definitions when an import declaration is encountered are ignored, so one software library can not affect how another software library is compiled, eliminating include-order dependencies.

* **Tool confusion**: Modules describe the API of software libraries, and tools can reason about and present a module as a representation of that API. Because modules can only be built standalone, tools can rely on the module definition to ensure that they get the complete API for the library. Moreover, modules can specify which languages they work with, so, e.g., one can not accidentally attempt to load a C++ module into a C program.

Problems Modules Do Not Solve
-----------------------------
Many programming languages have a module or package system, and because of the variety of features provided by these languages it is important to define what modules do *not* do. In particular, all of the following are considered out-of-scope for modules:

* **Rewrite the world's code**: It is not realistic to require applications or software libraries to make drastic or non-backward-compatible changes, nor is it feasible to completely eliminate headers. Modules must interoperate with existing software libraries and allow a gradual transition.

* **Versioning**: Modules have no notion of version information. Programmers must still rely on the existing versioning mechanisms of the underlying language (if any exist) to version software libraries.

* **Namespaces**: Unlike in some languages, modules do not imply any notion of namespaces. Thus, a struct declared in one module will still conflict with a struct of the same name declared in a different module, just as they would if declared in two different headers. This aspect is important for backward compatibility, because (for example) the mangled names of entities in software libraries must not change when introducing modules.

* **Binary distribution of modules**: Headers (particularly C++ headers) expose the full complexity of the language. Maintaining a stable binary module format across archectures, compiler versions, and compiler vendors is technically infeasible.

Using Modules
=============
To enable modules, pass the command-line flag ``-fmodules`` [#]_. This will make any modules-enabled software libraries available as modules as well as introducing any modules-specific syntax. Additional command-line parameters are described later.

Includes as Imports
-------------------
The primary user-level feature of modules is the import operation, which provides access to the API of software libraries. However, Clang does not provide a specific syntax for importing modules within the language itself [#]_. Instead, Clang translates ``#include`` directives into the corresponding module import. For example, the include directive

.. code-block:: c

  #include <stdio.h>

will be automatically mapped to an import of the module ``std.io``. Even with specific ``import`` syntax in the language, this particular feature is important for both adoption and backward compatibility: automatic translation of ``#include`` to ``import`` allows an application to get the benefits of modules (for any modules-enabled libraries) without any changes to the application itself. Thus, users can easily use modules with one compiler while falling back to the preprocessor-inclusion mechanism with other compilers.

Module Maps
-----------
The crucial link between modules and headers is described by a *module map*, which describes how a collection of existing headers maps on to the (logical) structure of a module. For example, one could imagine a module ``std`` covering the C standard library. Each of the C standard library headers (``<stdio.h>``, ``<stdlib.h>``, ``<math.h>``, etc.) would contribute to the ``std`` module, by placing their respective APIs into the corresponding submodule (``std.io``, ``std.lib``, ``std.math``, etc.). Having a list of the headers that are part of the ``std`` module allows the compiler to build the ``std`` module as a standalone entity, and having the mapping from header names to (sub)modules allows the automatic translation of ``#include`` directives to module imports.

Module maps are specified as separate files (each named ``module.map``) alongside the headers they describe, which allows them to be added to existing software libraries without having to change the library headers themselves (in most cases [#]_). The actual `Module Map Language`_ is described in a later section.

Compilation Model
-----------------
The binary representation of modules is automatically generated by the compiler on an as-needed basis. When a module is imported (e.g., by an ``#include`` of one of the module's headers), the compiler will spawn a second instance of itself, with a fresh preprocessing context [#]_, to parse just the headers in that module. The resulting Abstract Syntax Tree (AST) is then persisted into the binary representation of the module that is then loaded into translation unit where the module import was encountered.

The binary representation of modules is persisted in the *module cache*. Imports of a module will first query the module cache and, if a binary representation of the required module is already available, will load that representation directly. Thus, a module's headers will only be parsed once per language configuration, rather than once per translation unit that uses the module.

Modules maintain references to each of the headers that were part of the module build. If any of those headers changes, or if any of the modules on which a module depends change, then the module will be (automatically) recompiled. The process should never require any user intervention.

Command-line parameters
-----------------------
``-fmodules``
  Enable the modules feature (EXPERIMENTAL).

``-fcxx-modules``
  Enable the modules feature for C++ (EXPERIMENTAL and VERY BROKEN).

``-fmodules-cache-path=<directory>``
  Specify the path to the modules cache. If not provided, Clang will select a system-appropriate default.

``-f[no-]modules-autolink``
  Enable of disable automatic linking against the libraries associated with imported modules.

``-fmodules-ignore-macro=macroname``
  Instruct modules to ignore the named macro when selecting an appropriate module variant. Use this for macros defined on the command line that don't affect how modules are built, to improve sharing of compiled module files.

Module Map Language
===================
TBD


.. [#] Automatic linking against the libraries of modules requires specific linker support, which is not widely available.

.. [#] Modules are only available in C and Objective-C; a separate flag ``-fcxx-modules`` enables modules support for C++, which is even more experimental and broken.

.. [#] The ``import modulename;`` syntax described earlier in the document is a straw man proposal. Actual syntax will be pursued within the C++ committee and implemented in Clang.

.. [#] There are certain anti-patterns that occur in headers, particularly system headers, that cause problems for modules.

.. [#] The preprocessing context in which the modules are parsed is actually dependent on the command-line options provided to the compiler, including the language dialect and any ``-D`` options. However, the compiled modules for different command-line options are kept distinct, and any preprocessor directives that occur within the translation unit are ignored. 
