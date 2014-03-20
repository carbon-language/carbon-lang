=================
TableGen BackEnds
=================

.. contents::
   :local:

Introduction
============

TableGen backends are at the core of TableGen's functionality. The source files
provide the semantics to a generated (in memory) structure, but it's up to the
backend to print this out in a way that is meaningful to the user (normally a
C program including a file or a textual list of warnings, options and error
messages).

TableGen is used by both LLVM and Clang with very different goals. LLVM uses it
as a way to automate the generation of massive amounts of information regarding
instructions, schedules, cores and architecture features. Some backends generate
output that is consumed by more than one source file, so they need to be created
in a way that is easy to use pre-processor tricks. Some backends can also print
C code structures, so that they can be directly included as-is.

Clang, on the other hand, uses it mainly for diagnostic messages (errors,
warnings, tips) and attributes, so more on the textual end of the scale.

LLVM BackEnds
=============

.. warning::
   This document is raw. Each section below needs three sub-sections: description
   of its purpose with a list of users, output generated from generic input, and
   finally why it needed a new backend (in case there's something similar).

Emitter
-------

Generate machine code emitter.

RegisterInfo
------------

Generate registers and register classes info.

InstrInfo
---------

Generate instruction descriptions.

AsmWriter
---------

Generate calling convention descriptions.

AsmMatcher
----------

Generate assembly writer.

Disassembler
------------

Generate disassembler.

PseudoLowering
--------------

Generate pseudo instruction lowering.

CallingConv
-----------

Generate assembly instruction matcher.

DAGISel
-------

Generate a DAG instruction selector.

DFAPacketizer
-------------

Generate DFA Packetizer for VLIW targets.

FastISel
--------

Generate a "fast" instruction selector.

Subtarget
---------

Generate subtarget enumerations.

Intrinsic
---------

Generate intrinsic information.

TgtIntrinsic
------------

Generate target intrinsic information.

OptParserDefs
-------------

Print enum values for a class.

CTags
-----

Generate ctags-compatible index.


Clang BackEnds
==============

ClangAttrClasses
----------------

Generate clang attribute clases.

ClangAttrParserStringSwitches
-----------------------------

Generate all parser-related attribute string switches.

ClangAttrImpl
-------------

Generate clang attribute implementations.

ClangAttrList
-------------

Generate a clang attribute list.

ClangAttrPCHRead
----------------

Generate clang PCH attribute reader.

ClangAttrPCHWrite
-----------------

Generate clang PCH attribute writer.

ClangAttrSpellingList
---------------------

Generate a clang attribute spelling list.

ClangAttrSpellingListIndex
--------------------------

Generate a clang attribute spelling index.

ClangAttrASTVisitor
-------------------

Generate a recursive AST visitor for clang attribute.

ClangAttrTemplateInstantiate
----------------------------

Generate a clang template instantiate code.

ClangAttrParsedAttrList
-----------------------

Generate a clang parsed attribute list.

ClangAttrParsedAttrImpl
-----------------------

Generate the clang parsed attribute helpers.

ClangAttrParsedAttrKinds
------------------------

Generate a clang parsed attribute kinds.

ClangAttrDump
-------------

Generate clang attribute dumper.

ClangDiagsDefs
--------------

Generate Clang diagnostics definitions.

ClangDiagGroups
---------------

Generate Clang diagnostic groups.

ClangDiagsIndexName
-------------------

Generate Clang diagnostic name index.

ClangCommentNodes
-----------------

Generate Clang AST comment nodes.

ClangDeclNodes
--------------

Generate Clang AST declaration nodes.

ClangStmtNodes
--------------

Generate Clang AST statement nodes.

ClangSACheckers
---------------

Generate Clang Static Analyzer checkers.

ClangCommentHTMLTags
--------------------

Generate efficient matchers for HTML tag names that are used in documentation comments.

ClangCommentHTMLTagsProperties
------------------------------

Generate efficient matchers for HTML tag properties.

ClangCommentHTMLNamedCharacterReferences
----------------------------------------

Generate function to translate named character references to UTF-8 sequences.

ClangCommentCommandInfo
-----------------------

Generate command properties for commands that are used in documentation comments.

ClangCommentCommandList
-----------------------

Generate list of commands that are used in documentation comments.

ArmNeon
-------

Generate arm_neon.h for clang.

ArmNeonSema
-----------

Generate ARM NEON sema support for clang.

ArmNeonTest
-----------

Generate ARM NEON tests for clang.

AttrDocs
--------

Generate attribute documentation.

How to write a back-end
=======================

TODO.

Until we get a step-by-step HowTo for writing TableGen backends, you can at
least grab the boilerplate (build system, new files, etc.) from Clang's
r173931.

TODO: How they work, how to write one.  This section should not contain details
about any particular backend, except maybe ``-print-enums`` as an example.  This
should highlight the APIs in ``TableGen/Record.h``.

