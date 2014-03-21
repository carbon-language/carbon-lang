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

Overall, each backend will take the same TableGen file type and transform into
similar output for different targets/uses. There is an implicit contract between
the TableGen files, the back-ends and their users.

For instance, a global contract is that each back-end produces macro-guarded
sections. Based on whether the file is included by a header or a source file,
or even in which context of each file the include is being used, you have
todefine a macro just before including it, to get the right output:

.. code-block:: c++

  #define GET_REGINFO_TARGET_DESC
  #include "ARMGenRegisterInfo.inc"

And just part of the generated file would be included. This is useful if
you need the same information in multiple formats (instantiation, initialization,
getter/setter functions, etc) from the same source TableGen file without having
to re-compile the TableGen file multiple times.

Sometimes, multiple macros might be defined before the same include file to
output multiple blocks:

.. code-block:: c++

  #define GET_REGISTER_MATCHER
  #define GET_SUBTARGET_FEATURE_NAME
  #define GET_MATCHER_IMPLEMENTATION
  #include "ARMGenAsmMatcher.inc"

The macros will be undef'd automatically as they're used, in the include file.

On all LLVM back-ends, the ``llvm-tblgen`` binary will be executed on the root
TableGen file ``<Target>.td``, which should include all others. This guarantees
that all information needed is accessible, and that no duplication is needed
in the TbleGen files.

CodeEmitter
-----------

**Purpose**: CodeEmitterGen uses the descriptions of instructions and their fields to
construct an automated code emitter: a function that, given a MachineInstr,
returns the (currently, 32-bit unsigned) value of the instruction.

**Output**: C++ code, implementing the target's CodeEmitter
class by overriding the virtual functions as ``<Target>CodeEmitter::function()``.

**Usage**: Used to include directly at the end of ``<Target>CodeEmitter.cpp``, and
with option `-mc-emitter` to be included in ``<Target>MCCodeEmitter.cpp``.

RegisterInfo
------------

**Purpose**: This tablegen backend is responsible for emitting a description of a target
register file for a code generator.  It uses instances of the Register,
RegisterAliases, and RegisterClass classes to gather this information.

**Output**: C++ code with enums and structures representing the register mappings,
properties, masks, etc.

**Usage**: Both on ``<Target>BaseRegisterInfo`` and ``<Target>MCTargetDesc`` (headers
and source files) with macros defining in which they are for declaration vs.
initialization issues.

InstrInfo
---------

**Purpose**: This tablegen backend is responsible for emitting a description of the target
instruction set for the code generator. (what are the differences from CodeEmitter?)

**Output**: C++ code with enums and structures representing the register mappings,
properties, masks, etc.

**Usage**: Both on ``<Target>BaseInstrInfo`` and ``<Target>MCTargetDesc`` (headers
and source files) with macros defining in which they are for declaration vs.

AsmWriter
---------

**Purpose**: Emits an assembly printer for the current target.

**Output**: Implementation of ``<Target>InstPrinter::printInstruction()``, among
other things.

**Usage**: Included directly into ``InstPrinter/<Target>InstPrinter.cpp``.

AsmMatcher
----------

**Purpose**: Emits a target specifier matcher for
converting parsed assembly operands in the MCInst structures. It also
emits a matcher for custom operand parsing. Extensive documentation is
written on the ``AsmMatcherEmitter.cpp`` file.

**Output**: Assembler parsers' matcher functions, declarations, etc.

**Usage**: Used in back-ends' ``AsmParser/<Target>AsmParser.cpp`` for
building the AsmParser class.

Disassembler
------------

**Purpose**: Contains disassembler table emitters for various
architectures. Extensive documentation is written on the
``DisassemblerEmitter.cpp`` file.

**Output**: Decoding tables, static decoding functions, etc.

**Usage**: Directly included in ``Disassembler/<Target>Disassembler.cpp``
to cater for all default decodings, after all hand-made ones.

PseudoLowering
--------------

**Purpose**: Generate pseudo instruction lowering.

**Output**: Implements ``ARMAsmPrinter::emitPseudoExpansionLowering()``.

**Usage**: Included directly into ``<Target>AsmPrinter.cpp``.

CallingConv
-----------

**Purpose**: Responsible for emitting descriptions of the calling
conventions supported by this target.

**Output**: Implement static functions to deal with calling conventions
chained by matching styles, returning false on no match.

**Usage**: Used in ISelLowering and FastIsel as function pointers to
implementation returned by a CC sellection function.

DAGISel
-------

**Purpose**: Generate a DAG instruction selector.

**Output**: Creates huge functions for automating DAG selection.

**Usage**: Included in ``<Target>ISelDAGToDAG.cpp`` inside the target's
implementation of ``SelectionDAGISel``.

DFAPacketizer
-------------

**Purpose**: This class parses the Schedule.td file and produces an API that
can be used to reason about whether an instruction can be added to a packet
on a VLIW architecture. The class internally generates a deterministic finite
automaton (DFA) that models all possible mappings of machine instructions
to functional units as instructions are added to a packet.

**Output**: Scheduling tables for GPU back-ends (Hexagon, AMD).

**Usage**: Included directly on ``<Target>InstrInfo.cpp``.

FastISel
--------

**Purpose**: This tablegen backend emits code for use by the "fast"
instruction selection algorithm. See the comments at the top of
lib/CodeGen/SelectionDAG/FastISel.cpp for background. This file
scans through the target's tablegen instruction-info files
and extracts instructions with obvious-looking patterns, and it emits
code to look up these instructions by type and operator.

**Output**: Generates ``Predicate`` and ``FastEmit`` methods.

**Usage**: Implements private methods of the targets' implementation
of ``FastISel`` class.

Subtarget
---------

**Purpose**: Generate subtarget enumerations.

**Output**: Enums, globals, local tables for sub-target information.

**Usage**: Populates ``<Target>Subtarget`` and
``MCTargetDesc/<Target>MCTargetDesc`` files (both headers and source).

Intrinsic
---------

**Purpose**: Generate (target) intrinsic information.

OptParserDefs
-------------

**Purpose**: Print enum values for a class.

CTags
-----

**Purpose**: This tablegen backend emits an index of definitions in ctags(1)
format. A helper script, utils/TableGen/tdtags, provides an easier-to-use
interface; run 'tdtags -H' for documentation.

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

