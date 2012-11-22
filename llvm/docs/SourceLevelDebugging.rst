================================
Source Level Debugging with LLVM
================================

.. sectionauthor:: Chris Lattner <sabre@nondot.org> and Jim Laskey <jlaskey@mac.com>

.. contents::
   :local:

Introduction
============

This document is the central repository for all information pertaining to debug
information in LLVM.  It describes the :ref:`actual format that the LLVM debug
information takes <format>`, which is useful for those interested in creating
front-ends or dealing directly with the information.  Further, this document
provides specific examples of what debug information for C/C++ looks like.

Philosophy behind LLVM debugging information
--------------------------------------------

The idea of the LLVM debugging information is to capture how the important
pieces of the source-language's Abstract Syntax Tree map onto LLVM code.
Several design aspects have shaped the solution that appears here.  The
important ones are:

* Debugging information should have very little impact on the rest of the
  compiler.  No transformations, analyses, or code generators should need to
  be modified because of debugging information.

* LLVM optimizations should interact in :ref:`well-defined and easily described
  ways <intro_debugopt>` with the debugging information.

* Because LLVM is designed to support arbitrary programming languages,
  LLVM-to-LLVM tools should not need to know anything about the semantics of
  the source-level-language.

* Source-level languages are often **widely** different from one another.
  LLVM should not put any restrictions of the flavor of the source-language,
  and the debugging information should work with any language.

* With code generator support, it should be possible to use an LLVM compiler
  to compile a program to native machine code and standard debugging
  formats.  This allows compatibility with traditional machine-code level
  debuggers, like GDB or DBX.

The approach used by the LLVM implementation is to use a small set of
:ref:`intrinsic functions <format_common_intrinsics>` to define a mapping
between LLVM program objects and the source-level objects.  The description of
the source-level program is maintained in LLVM metadata in an
:ref:`implementation-defined format <ccxx_frontend>` (the C/C++ front-end
currently uses working draft 7 of the `DWARF 3 standard
<http://www.eagercon.com/dwarf/dwarf3std.htm>`_).

When a program is being debugged, a debugger interacts with the user and turns
the stored debug information into source-language specific information.  As
such, a debugger must be aware of the source-language, and is thus tied to a
specific language or family of languages.

Debug information consumers
---------------------------

The role of debug information is to provide meta information normally stripped
away during the compilation process.  This meta information provides an LLVM
user a relationship between generated code and the original program source
code.

Currently, debug information is consumed by DwarfDebug to produce dwarf
information used by the gdb debugger.  Other targets could use the same
information to produce stabs or other debug forms.

It would also be reasonable to use debug information to feed profiling tools
for analysis of generated code, or, tools for reconstructing the original
source from generated code.

TODO - expound a bit more.

.. _intro_debugopt:

Debugging optimized code
------------------------

An extremely high priority of LLVM debugging information is to make it interact
well with optimizations and analysis.  In particular, the LLVM debug
information provides the following guarantees:

* LLVM debug information **always provides information to accurately read
  the source-level state of the program**, regardless of which LLVM
  optimizations have been run, and without any modification to the
  optimizations themselves.  However, some optimizations may impact the
  ability to modify the current state of the program with a debugger, such
  as setting program variables, or calling functions that have been
  deleted.

* As desired, LLVM optimizations can be upgraded to be aware of the LLVM
  debugging information, allowing them to update the debugging information
  as they perform aggressive optimizations.  This means that, with effort,
  the LLVM optimizers could optimize debug code just as well as non-debug
  code.

* LLVM debug information does not prevent optimizations from
  happening (for example inlining, basic block reordering/merging/cleanup,
  tail duplication, etc).

* LLVM debug information is automatically optimized along with the rest of
  the program, using existing facilities.  For example, duplicate
  information is automatically merged by the linker, and unused information
  is automatically removed.

Basically, the debug information allows you to compile a program with
"``-O0 -g``" and get full debug information, allowing you to arbitrarily modify
the program as it executes from a debugger.  Compiling a program with
"``-O3 -g``" gives you full debug information that is always available and
accurate for reading (e.g., you get accurate stack traces despite tail call
elimination and inlining), but you might lose the ability to modify the program
and call functions where were optimized out of the program, or inlined away
completely.

:ref:`LLVM test suite <test-suite-quickstart>` provides a framework to test
optimizer's handling of debugging information.  It can be run like this:

.. code-block:: bash

  % cd llvm/projects/test-suite/MultiSource/Benchmarks  # or some other level
  % make TEST=dbgopt

This will test impact of debugging information on optimization passes.  If
debugging information influences optimization passes then it will be reported
as a failure.  See :doc:`TestingGuide` for more information on LLVM test
infrastructure and how to run various tests.

.. _format:

Debugging information format
============================

LLVM debugging information has been carefully designed to make it possible for
the optimizer to optimize the program and debugging information without
necessarily having to know anything about debugging information.  In
particular, the use of metadata avoids duplicated debugging information from
the beginning, and the global dead code elimination pass automatically deletes
debugging information for a function if it decides to delete the function.

To do this, most of the debugging information (descriptors for types,
variables, functions, source files, etc) is inserted by the language front-end
in the form of LLVM metadata.

Debug information is designed to be agnostic about the target debugger and
debugging information representation (e.g. DWARF/Stabs/etc).  It uses a generic
pass to decode the information that represents variables, types, functions,
namespaces, etc: this allows for arbitrary source-language semantics and
type-systems to be used, as long as there is a module written for the target
debugger to interpret the information.

To provide basic functionality, the LLVM debugger does have to make some
assumptions about the source-level language being debugged, though it keeps
these to a minimum.  The only common features that the LLVM debugger assumes
exist are :ref:`source files <format_files>`, and :ref:`program objects
<format_global_variables>`.  These abstract objects are used by a debugger to
form stack traces, show information about local variables, etc.

This section of the documentation first describes the representation aspects
common to any source-language.  :ref:`ccxx_frontend` describes the data layout
conventions used by the C and C++ front-ends.

Debug information descriptors
-----------------------------

In consideration of the complexity and volume of debug information, LLVM
provides a specification for well formed debug descriptors.

Consumers of LLVM debug information expect the descriptors for program objects
to start in a canonical format, but the descriptors can include additional
information appended at the end that is source-language specific.  All LLVM
debugging information is versioned, allowing backwards compatibility in the
case that the core structures need to change in some way.  Also, all debugging
information objects start with a tag to indicate what type of object it is.
The source-language is allowed to define its own objects, by using unreserved
tag numbers.  We recommend using with tags in the range 0x1000 through 0x2000
(there is a defined ``enum DW_TAG_user_base = 0x1000``.)

The fields of debug descriptors used internally by LLVM are restricted to only
the simple data types ``i32``, ``i1``, ``float``, ``double``, ``mdstring`` and
``mdnode``.

.. code-block:: llvm

  !1 = metadata !{
    i32,   ;; A tag
    ...
  }

<a name="LLVMDebugVersion">The first field of a descriptor is always an
``i32`` containing a tag value identifying the content of the descriptor.
The remaining fields are specific to the descriptor.  The values of tags are
loosely bound to the tag values of DWARF information entries.  However, that
does not restrict the use of the information supplied to DWARF targets.  To
facilitate versioning of debug information, the tag is augmented with the
current debug version (``LLVMDebugVersion = 8 << 16`` or 0x80000 or
524288.)

The details of the various descriptors follow.

Compile unit descriptors
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !0 = metadata !{
    i32,       ;; Tag = 17 + LLVMDebugVersion (DW_TAG_compile_unit)
    i32,       ;; Unused field.
    i32,       ;; DWARF language identifier (ex. DW_LANG_C89)
    metadata,  ;; Source file name
    metadata,  ;; Source file directory (includes trailing slash)
    metadata   ;; Producer (ex. "4.0.1 LLVM (LLVM research group)")
    i1,        ;; True if this is a main compile unit.
    i1,        ;; True if this is optimized.
    metadata,  ;; Flags
    i32        ;; Runtime version
    metadata   ;; List of enums types
    metadata   ;; List of retained types
    metadata   ;; List of subprograms
    metadata   ;; List of global variables
  }

These descriptors contain a source language ID for the file (we use the DWARF
3.0 ID numbers, such as ``DW_LANG_C89``, ``DW_LANG_C_plus_plus``,
``DW_LANG_Cobol74``, etc), three strings describing the filename, working
directory of the compiler, and an identifier string for the compiler that
produced it.

Compile unit descriptors provide the root context for objects declared in a
specific compilation unit.  File descriptors are defined using this context.
These descriptors are collected by a named metadata ``!llvm.dbg.cu``.  Compile
unit descriptor keeps track of subprograms, global variables and type
information.

.. _format_files:

File descriptors
^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !0 = metadata !{
    i32,       ;; Tag = 41 + LLVMDebugVersion (DW_TAG_file_type)
    metadata,  ;; Source file name
    metadata,  ;; Source file directory (includes trailing slash)
    metadata   ;; Unused
  }

These descriptors contain information for a file.  Global variables and top
level functions would be defined using this context.  File descriptors also
provide context for source line correspondence.

Each input file is encoded as a separate file descriptor in LLVM debugging
information output.

.. _format_global_variables:

Global variable descriptors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !1 = metadata !{
    i32,      ;; Tag = 52 + LLVMDebugVersion (DW_TAG_variable)
    i32,      ;; Unused field.
    metadata, ;; Reference to context descriptor
    metadata, ;; Name
    metadata, ;; Display name (fully qualified C++ name)
    metadata, ;; MIPS linkage name (for C++)
    metadata, ;; Reference to file where defined
    i32,      ;; Line number where defined
    metadata, ;; Reference to type descriptor
    i1,       ;; True if the global is local to compile unit (static)
    i1,       ;; True if the global is defined in the compile unit (not extern)
    {}*       ;; Reference to the global variable
  }

These descriptors provide debug information about globals variables.  The
provide details such as name, type and where the variable is defined.  All
global variables are collected inside the named metadata ``!llvm.dbg.cu``.

.. _format_subprograms:

Subprogram descriptors
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32,      ;; Tag = 46 + LLVMDebugVersion (DW_TAG_subprogram)
    i32,      ;; Unused field.
    metadata, ;; Reference to context descriptor
    metadata, ;; Name
    metadata, ;; Display name (fully qualified C++ name)
    metadata, ;; MIPS linkage name (for C++)
    metadata, ;; Reference to file where defined
    i32,      ;; Line number where defined
    metadata, ;; Reference to type descriptor
    i1,       ;; True if the global is local to compile unit (static)
    i1,       ;; True if the global is defined in the compile unit (not extern)
    i32,      ;; Line number where the scope of the subprogram begins
    i32,      ;; Virtuality, e.g. dwarf::DW_VIRTUALITY__virtual
    i32,      ;; Index into a virtual function
    metadata, ;; indicates which base type contains the vtable pointer for the
              ;; derived class
    i32,      ;; Flags - Artifical, Private, Protected, Explicit, Prototyped.
    i1,       ;; isOptimized
    Function * , ;; Pointer to LLVM function
    metadata, ;; Lists function template parameters
    metadata, ;; Function declaration descriptor
    metadata  ;; List of function variables
  }

These descriptors provide debug information about functions, methods and
subprograms.  They provide details such as name, return types and the source
location where the subprogram is defined.

Block descriptors
^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !3 = metadata !{
    i32,     ;; Tag = 11 + LLVMDebugVersion (DW_TAG_lexical_block)
    metadata,;; Reference to context descriptor
    i32,     ;; Line number
    i32,     ;; Column number
    metadata,;; Reference to source file
    i32      ;; Unique ID to identify blocks from a template function
  }

This descriptor provides debug information about nested blocks within a
subprogram.  The line number and column numbers are used to dinstinguish two
lexical blocks at same depth.

.. code-block:: llvm

  !3 = metadata !{
    i32,     ;; Tag = 11 + LLVMDebugVersion (DW_TAG_lexical_block)
    metadata ;; Reference to the scope we're annotating with a file change
    metadata,;; Reference to the file the scope is enclosed in.
  }

This descriptor provides a wrapper around a lexical scope to handle file
changes in the middle of a lexical block.

.. _format_basic_type:

Basic type descriptors
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !4 = metadata !{
    i32,      ;; Tag = 36 + LLVMDebugVersion (DW_TAG_base_type)
    metadata, ;; Reference to context
    metadata, ;; Name (may be "" for anonymous types)
    metadata, ;; Reference to file where defined (may be NULL)
    i32,      ;; Line number where defined (may be 0)
    i64,      ;; Size in bits
    i64,      ;; Alignment in bits
    i64,      ;; Offset in bits
    i32,      ;; Flags
    i32       ;; DWARF type encoding
  }

These descriptors define primitive types used in the code.  Example ``int``,
``bool`` and ``float``.  The context provides the scope of the type, which is
usually the top level.  Since basic types are not usually user defined the
context and line number can be left as NULL and 0.  The size, alignment and
offset are expressed in bits and can be 64 bit values.  The alignment is used
to round the offset when embedded in a :ref:`composite type
<format_composite_type>` (example to keep float doubles on 64 bit boundaries).
The offset is the bit offset if embedded in a :ref:`composite type
<format_composite_type>`.

The type encoding provides the details of the type.  The values are typically
one of the following:

.. code-block:: llvm

  DW_ATE_address       = 1
  DW_ATE_boolean       = 2
  DW_ATE_float         = 4
  DW_ATE_signed        = 5
  DW_ATE_signed_char   = 6
  DW_ATE_unsigned      = 7
  DW_ATE_unsigned_char = 8

.. _format_derived_type:

Derived type descriptors
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !5 = metadata !{
    i32,      ;; Tag (see below)
    metadata, ;; Reference to context
    metadata, ;; Name (may be "" for anonymous types)
    metadata, ;; Reference to file where defined (may be NULL)
    i32,      ;; Line number where defined (may be 0)
    i64,      ;; Size in bits
    i64,      ;; Alignment in bits
    i64,      ;; Offset in bits
    i32,      ;; Flags to encode attributes, e.g. private
    metadata, ;; Reference to type derived from
    metadata, ;; (optional) Name of the Objective C property associated with
              ;; Objective-C an ivar
    metadata, ;; (optional) Name of the Objective C property getter selector.
    metadata, ;; (optional) Name of the Objective C property setter selector.
    i32       ;; (optional) Objective C property attributes.
  }

These descriptors are used to define types derived from other types.  The value
of the tag varies depending on the meaning.  The following are possible tag
values:

.. code-block:: llvm

  DW_TAG_formal_parameter = 5
  DW_TAG_member           = 13
  DW_TAG_pointer_type     = 15
  DW_TAG_reference_type   = 16
  DW_TAG_typedef          = 22
  DW_TAG_const_type       = 38
  DW_TAG_volatile_type    = 53
  DW_TAG_restrict_type    = 55

``DW_TAG_member`` is used to define a member of a :ref:`composite type
<format_composite_type>` or :ref:`subprogram <format_subprograms>`.  The type
of the member is the :ref:`derived type <format_derived_type>`.
``DW_TAG_formal_parameter`` is used to define a member which is a formal
argument of a subprogram.

``DW_TAG_typedef`` is used to provide a name for the derived type.

``DW_TAG_pointer_type``, ``DW_TAG_reference_type``, ``DW_TAG_const_type``,
``DW_TAG_volatile_type`` and ``DW_TAG_restrict_type`` are used to qualify the
:ref:`derived type <format_derived_type>`.

:ref:`Derived type <format_derived_type>` location can be determined from the
context and line number.  The size, alignment and offset are expressed in bits
and can be 64 bit values.  The alignment is used to round the offset when
embedded in a :ref:`composite type <format_composite_type>`  (example to keep
float doubles on 64 bit boundaries.) The offset is the bit offset if embedded
in a :ref:`composite type <format_composite_type>`.

Note that the ``void *`` type is expressed as a type derived from NULL.

.. _format_composite_type:

Composite type descriptors
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !6 = metadata !{
    i32,      ;; Tag (see below)
    metadata, ;; Reference to context
    metadata, ;; Name (may be "" for anonymous types)
    metadata, ;; Reference to file where defined (may be NULL)
    i32,      ;; Line number where defined (may be 0)
    i64,      ;; Size in bits
    i64,      ;; Alignment in bits
    i64,      ;; Offset in bits
    i32,      ;; Flags
    metadata, ;; Reference to type derived from
    metadata, ;; Reference to array of member descriptors
    i32       ;; Runtime languages
  }

These descriptors are used to define types that are composed of 0 or more
elements.  The value of the tag varies depending on the meaning.  The following
are possible tag values:

.. code-block:: llvm

  DW_TAG_array_type       = 1
  DW_TAG_enumeration_type = 4
  DW_TAG_structure_type   = 19
  DW_TAG_union_type       = 23
  DW_TAG_vector_type      = 259
  DW_TAG_subroutine_type  = 21
  DW_TAG_inheritance      = 28

The vector flag indicates that an array type is a native packed vector.

The members of array types (tag = ``DW_TAG_array_type``) or vector types (tag =
``DW_TAG_vector_type``) are :ref:`subrange descriptors <format_subrange>`, each
representing the range of subscripts at that level of indexing.

The members of enumeration types (tag = ``DW_TAG_enumeration_type``) are
:ref:`enumerator descriptors <format_enumerator>`, each representing the
definition of enumeration value for the set.  All enumeration type descriptors
are collected inside the named metadata ``!llvm.dbg.cu``.

The members of structure (tag = ``DW_TAG_structure_type``) or union (tag =
``DW_TAG_union_type``) types are any one of the :ref:`basic
<format_basic_type>`, :ref:`derived <format_derived_type>` or :ref:`composite
<format_composite_type>` type descriptors, each representing a field member of
the structure or union.

For C++ classes (tag = ``DW_TAG_structure_type``), member descriptors provide
information about base classes, static members and member functions.  If a
member is a :ref:`derived type descriptor <format_derived_type>` and has a tag
of ``DW_TAG_inheritance``, then the type represents a base class.  If the member
of is a :ref:`global variable descriptor <format_global_variables>` then it
represents a static member.  And, if the member is a :ref:`subprogram
descriptor <format_subprograms>` then it represents a member function.  For
static members and member functions, ``getName()`` returns the members link or
the C++ mangled name.  ``getDisplayName()`` the simplied version of the name.

The first member of subroutine (tag = ``DW_TAG_subroutine_type``) type elements
is the return type for the subroutine.  The remaining elements are the formal
arguments to the subroutine.

:ref:`Composite type <format_composite_type>` location can be determined from
the context and line number.  The size, alignment and offset are expressed in
bits and can be 64 bit values.  The alignment is used to round the offset when
embedded in a :ref:`composite type <format_composite_type>` (as an example, to
keep float doubles on 64 bit boundaries).  The offset is the bit offset if
embedded in a :ref:`composite type <format_composite_type>`.

.. _format_subrange:

Subrange descriptors
^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !42 = metadata !{
    i32,    ;; Tag = 33 + LLVMDebugVersion (DW_TAG_subrange_type)
    i64,    ;; Low value
    i64     ;; High value
  }

These descriptors are used to define ranges of array subscripts for an array
:ref:`composite type <format_composite_type>`.  The low value defines the lower
bounds typically zero for C/C++.  The high value is the upper bounds.  Values
are 64 bit.  ``High - Low + 1`` is the size of the array.  If ``Low > High``
the array bounds are not included in generated debugging information.

.. _format_enumerator:

Enumerator descriptors
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !6 = metadata !{
    i32,      ;; Tag = 40 + LLVMDebugVersion (DW_TAG_enumerator)
    metadata, ;; Name
    i64       ;; Value
  }

These descriptors are used to define members of an enumeration :ref:`composite
type <format_composite_type>`, it associates the name to the value.

Local variables
^^^^^^^^^^^^^^^

.. code-block:: llvm

  !7 = metadata !{
    i32,      ;; Tag (see below)
    metadata, ;; Context
    metadata, ;; Name
    metadata, ;; Reference to file where defined
    i32,      ;; 24 bit - Line number where defined
              ;; 8 bit - Argument number. 1 indicates 1st argument.
    metadata, ;; Type descriptor
    i32,      ;; flags
    metadata  ;; (optional) Reference to inline location
  }

These descriptors are used to define variables local to a sub program.  The
value of the tag depends on the usage of the variable:

.. code-block:: llvm

  DW_TAG_auto_variable   = 256
  DW_TAG_arg_variable    = 257
  DW_TAG_return_variable = 258

An auto variable is any variable declared in the body of the function.  An
argument variable is any variable that appears as a formal argument to the
function.  A return variable is used to track the result of a function and has
no source correspondent.

The context is either the subprogram or block where the variable is defined.
Name the source variable name.  Context and line indicate where the variable
was defined.  Type descriptor defines the declared type of the variable.

.. _format_common_intrinsics:

Debugger intrinsic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LLVM uses several intrinsic functions (name prefixed with "``llvm.dbg``") to
provide debug information at various points in generated code.

``llvm.dbg.declare``
^^^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  void %llvm.dbg.declare(metadata, metadata)

This intrinsic provides information about a local element (e.g., variable).
The first argument is metadata holding the alloca for the variable.  The second
argument is metadata containing a description of the variable.

``llvm.dbg.value``
^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  void %llvm.dbg.value(metadata, i64, metadata)

This intrinsic provides information when a user source variable is set to a new
value.  The first argument is the new value (wrapped as metadata).  The second
argument is the offset in the user source variable where the new value is
written.  The third argument is metadata containing a description of the user
source variable.

Object lifetimes and scoping
============================

In many languages, the local variables in functions can have their lifetimes or
scopes limited to a subset of a function.  In the C family of languages, for
example, variables are only live (readable and writable) within the source
block that they are defined in.  In functional languages, values are only
readable after they have been defined.  Though this is a very obvious concept,
it is non-trivial to model in LLVM, because it has no notion of scoping in this
sense, and does not want to be tied to a language's scoping rules.

In order to handle this, the LLVM debug format uses the metadata attached to
llvm instructions to encode line number and scoping information.  Consider the
following C fragment, for example:

.. code-block:: c

  1.  void foo() {
  2.    int X = 21;
  3.    int Y = 22;
  4.    {
  5.      int Z = 23;
  6.      Z = X;
  7.    }
  8.    X = Y;
  9.  }

Compiled to LLVM, this function would be represented like this:

.. code-block:: llvm

  define void @foo() nounwind ssp {
  entry:
    %X = alloca i32, align 4                        ; <i32*> [#uses=4]
    %Y = alloca i32, align 4                        ; <i32*> [#uses=4]
    %Z = alloca i32, align 4                        ; <i32*> [#uses=3]
    %0 = bitcast i32* %X to {}*                     ; <{}*> [#uses=1]
    call void @llvm.dbg.declare(metadata !{i32 * %X}, metadata !0), !dbg !7
    store i32 21, i32* %X, !dbg !8
    %1 = bitcast i32* %Y to {}*                     ; <{}*> [#uses=1]
    call void @llvm.dbg.declare(metadata !{i32 * %Y}, metadata !9), !dbg !10
    store i32 22, i32* %Y, !dbg !11
    %2 = bitcast i32* %Z to {}*                     ; <{}*> [#uses=1]
    call void @llvm.dbg.declare(metadata !{i32 * %Z}, metadata !12), !dbg !14
    store i32 23, i32* %Z, !dbg !15
    %tmp = load i32* %X, !dbg !16                   ; <i32> [#uses=1]
    %tmp1 = load i32* %Y, !dbg !16                  ; <i32> [#uses=1]
    %add = add nsw i32 %tmp, %tmp1, !dbg !16        ; <i32> [#uses=1]
    store i32 %add, i32* %Z, !dbg !16
    %tmp2 = load i32* %Y, !dbg !17                  ; <i32> [#uses=1]
    store i32 %tmp2, i32* %X, !dbg !17
    ret void, !dbg !18
  }

  declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

  !0 = metadata !{i32 459008, metadata !1, metadata !"X",
                  metadata !3, i32 2, metadata !6}; [ DW_TAG_auto_variable ]
  !1 = metadata !{i32 458763, metadata !2}; [DW_TAG_lexical_block ]
  !2 = metadata !{i32 458798, i32 0, metadata !3, metadata !"foo", metadata !"foo",
                 metadata !"foo", metadata !3, i32 1, metadata !4,
                 i1 false, i1 true}; [DW_TAG_subprogram ]
  !3 = metadata !{i32 458769, i32 0, i32 12, metadata !"foo.c",
                  metadata !"/private/tmp", metadata !"clang 1.1", i1 true,
                  i1 false, metadata !"", i32 0}; [DW_TAG_compile_unit ]
  !4 = metadata !{i32 458773, metadata !3, metadata !"", null, i32 0, i64 0, i64 0,
                  i64 0, i32 0, null, metadata !5, i32 0}; [DW_TAG_subroutine_type ]
  !5 = metadata !{null}
  !6 = metadata !{i32 458788, metadata !3, metadata !"int", metadata !3, i32 0,
                  i64 32, i64 32, i64 0, i32 0, i32 5}; [DW_TAG_base_type ]
  !7 = metadata !{i32 2, i32 7, metadata !1, null}
  !8 = metadata !{i32 2, i32 3, metadata !1, null}
  !9 = metadata !{i32 459008, metadata !1, metadata !"Y", metadata !3, i32 3,
                  metadata !6}; [ DW_TAG_auto_variable ]
  !10 = metadata !{i32 3, i32 7, metadata !1, null}
  !11 = metadata !{i32 3, i32 3, metadata !1, null}
  !12 = metadata !{i32 459008, metadata !13, metadata !"Z", metadata !3, i32 5,
                   metadata !6}; [ DW_TAG_auto_variable ]
  !13 = metadata !{i32 458763, metadata !1}; [DW_TAG_lexical_block ]
  !14 = metadata !{i32 5, i32 9, metadata !13, null}
  !15 = metadata !{i32 5, i32 5, metadata !13, null}
  !16 = metadata !{i32 6, i32 5, metadata !13, null}
  !17 = metadata !{i32 8, i32 3, metadata !1, null}
  !18 = metadata !{i32 9, i32 1, metadata !2, null}

This example illustrates a few important details about LLVM debugging
information.  In particular, it shows how the ``llvm.dbg.declare`` intrinsic and
location information, which are attached to an instruction, are applied
together to allow a debugger to analyze the relationship between statements,
variable definitions, and the code used to implement the function.

.. code-block:: llvm

  call void @llvm.dbg.declare(metadata, metadata !0), !dbg !7

The first intrinsic ``%llvm.dbg.declare`` encodes debugging information for the
variable ``X``.  The metadata ``!dbg !7`` attached to the intrinsic provides
scope information for the variable ``X``.

.. code-block:: llvm

  !7 = metadata !{i32 2, i32 7, metadata !1, null}
  !1 = metadata !{i32 458763, metadata !2}; [DW_TAG_lexical_block ]
  !2 = metadata !{i32 458798, i32 0, metadata !3, metadata !"foo",
                  metadata !"foo", metadata !"foo", metadata !3, i32 1,
                  metadata !4, i1 false, i1 true}; [DW_TAG_subprogram ]

Here ``!7`` is metadata providing location information.  It has four fields:
line number, column number, scope, and original scope.  The original scope
represents inline location if this instruction is inlined inside a caller, and
is null otherwise.  In this example, scope is encoded by ``!1``. ``!1``
represents a lexical block inside the scope ``!2``, where ``!2`` is a
:ref:`subprogram descriptor <format_subprograms>`.  This way the location
information attached to the intrinsics indicates that the variable ``X`` is
declared at line number 2 at a function level scope in function ``foo``.

Now lets take another example.

.. code-block:: llvm

  call void @llvm.dbg.declare(metadata, metadata !12), !dbg !14

The second intrinsic ``%llvm.dbg.declare`` encodes debugging information for
variable ``Z``.  The metadata ``!dbg !14`` attached to the intrinsic provides
scope information for the variable ``Z``.

.. code-block:: llvm

  !13 = metadata !{i32 458763, metadata !1}; [DW_TAG_lexical_block ]
  !14 = metadata !{i32 5, i32 9, metadata !13, null}

Here ``!14`` indicates that ``Z`` is declared at line number 5 and
column number 9 inside of lexical scope ``!13``.  The lexical scope itself
resides inside of lexical scope ``!1`` described above.

The scope information attached with each instruction provides a straightforward
way to find instructions covered by a scope.

.. _ccxx_frontend:

C/C++ front-end specific debug information
==========================================

The C and C++ front-ends represent information about the program in a format
that is effectively identical to `DWARF 3.0
<http://www.eagercon.com/dwarf/dwarf3std.htm>`_ in terms of information
content.  This allows code generators to trivially support native debuggers by
generating standard dwarf information, and contains enough information for
non-dwarf targets to translate it as needed.

This section describes the forms used to represent C and C++ programs.  Other
languages could pattern themselves after this (which itself is tuned to
representing programs in the same way that DWARF 3 does), or they could choose
to provide completely different forms if they don't fit into the DWARF model.
As support for debugging information gets added to the various LLVM
source-language front-ends, the information used should be documented here.

The following sections provide examples of various C/C++ constructs and the
debug information that would best describe those constructs.

C/C++ source file information
-----------------------------

Given the source files ``MySource.cpp`` and ``MyHeader.h`` located in the
directory ``/Users/mine/sources``, the following code:

.. code-block:: c

  #include "MyHeader.h"

  int main(int argc, char *argv[]) {
    return 0;
  }

a C/C++ front-end would generate the following descriptors:

.. code-block:: llvm

  ...
  ;;
  ;; Define the compile unit for the main source file "/Users/mine/sources/MySource.cpp".
  ;;
  !2 = metadata !{
    i32 524305,    ;; Tag
    i32 0,         ;; Unused
    i32 4,         ;; Language Id
    metadata !"MySource.cpp",
    metadata !"/Users/mine/sources",
    metadata !"4.2.1 (Based on Apple Inc. build 5649) (LLVM build 00)",
    i1 true,       ;; Main Compile Unit
    i1 false,      ;; Optimized compile unit
    metadata !"",  ;; Compiler flags
    i32 0}         ;; Runtime version

  ;;
  ;; Define the file for the file "/Users/mine/sources/MySource.cpp".
  ;;
  !1 = metadata !{
    i32 524329,    ;; Tag
    metadata !"MySource.cpp",
    metadata !"/Users/mine/sources",
    metadata !2    ;; Compile unit
  }

  ;;
  ;; Define the file for the file "/Users/mine/sources/Myheader.h"
  ;;
  !3 = metadata !{
    i32 524329,    ;; Tag
    metadata !"Myheader.h"
    metadata !"/Users/mine/sources",
    metadata !2    ;; Compile unit
  }

  ...

``llvm::Instruction`` provides easy access to metadata attached with an
instruction.  One can extract line number information encoded in LLVM IR using
``Instruction::getMetadata()`` and ``DILocation::getLineNumber()``.

.. code-block:: c++

  if (MDNode *N = I->getMetadata("dbg")) {  // Here I is an LLVM instruction
    DILocation Loc(N);                      // DILocation is in DebugInfo.h
    unsigned Line = Loc.getLineNumber();
    StringRef File = Loc.getFilename();
    StringRef Dir = Loc.getDirectory();
  }

C/C++ global variable information
---------------------------------

Given an integer global variable declared as follows:

.. code-block:: c

  int MyGlobal = 100;

a C/C++ front-end would generate the following descriptors:

.. code-block:: llvm

  ;;
  ;; Define the global itself.
  ;;
  %MyGlobal = global int 100
  ...
  ;;
  ;; List of debug info of globals
  ;;
  !llvm.dbg.cu = !{!0}

  ;; Define the compile unit.
  !0 = metadata !{
    i32 786449,                       ;; Tag
    i32 0,                            ;; Context
    i32 4,                            ;; Language
    metadata !"foo.cpp",              ;; File
    metadata !"/Volumes/Data/tmp",    ;; Directory
    metadata !"clang version 3.1 ",   ;; Producer
    i1 true,                          ;; Deprecated field
    i1 false,                         ;; "isOptimized"?
    metadata !"",                     ;; Flags
    i32 0,                            ;; Runtime Version
    metadata !1,                      ;; Enum Types
    metadata !1,                      ;; Retained Types
    metadata !1,                      ;; Subprograms
    metadata !3                       ;; Global Variables
  } ; [ DW_TAG_compile_unit ]

  ;; The Array of Global Variables
  !3 = metadata !{
    metadata !4
  }

  !4 = metadata !{
    metadata !5
  }

  ;;
  ;; Define the global variable itself.
  ;;
  !5 = metadata !{
    i32 786484,                        ;; Tag
    i32 0,                             ;; Unused
    null,                              ;; Unused
    metadata !"MyGlobal",              ;; Name
    metadata !"MyGlobal",              ;; Display Name
    metadata !"",                      ;; Linkage Name
    metadata !6,                       ;; File
    i32 1,                             ;; Line
    metadata !7,                       ;; Type
    i32 0,                             ;; IsLocalToUnit
    i32 1,                             ;; IsDefinition
    i32* @MyGlobal                     ;; LLVM-IR Value
  } ; [ DW_TAG_variable ]

  ;;
  ;; Define the file
  ;;
  !6 = metadata !{
    i32 786473,                        ;; Tag
    metadata !"foo.cpp",               ;; File
    metadata !"/Volumes/Data/tmp",     ;; Directory
    null                               ;; Unused
  } ; [ DW_TAG_file_type ]

  ;;
  ;; Define the type
  ;;
  !7 = metadata !{
    i32 786468,                         ;; Tag
    null,                               ;; Unused
    metadata !"int",                    ;; Name
    null,                               ;; Unused
    i32 0,                              ;; Line
    i64 32,                             ;; Size in Bits
    i64 32,                             ;; Align in Bits
    i64 0,                              ;; Offset
    i32 0,                              ;; Flags
    i32 5                               ;; Encoding
  } ; [ DW_TAG_base_type ]

C/C++ function information
--------------------------

Given a function declared as follows:

.. code-block:: c

  int main(int argc, char *argv[]) {
    return 0;
  }

a C/C++ front-end would generate the following descriptors:

.. code-block:: llvm

  ;;
  ;; Define the anchor for subprograms.  Note that the second field of the
  ;; anchor is 46, which is the same as the tag for subprograms
  ;; (46 = DW_TAG_subprogram.)
  ;;
  !6 = metadata !{
    i32 524334,        ;; Tag
    i32 0,             ;; Unused
    metadata !1,       ;; Context
    metadata !"main",  ;; Name
    metadata !"main",  ;; Display name
    metadata !"main",  ;; Linkage name
    metadata !1,       ;; File
    i32 1,             ;; Line number
    metadata !4,       ;; Type
    i1 false,          ;; Is local
    i1 true,           ;; Is definition
    i32 0,             ;; Virtuality attribute, e.g. pure virtual function
    i32 0,             ;; Index into virtual table for C++ methods
    i32 0,             ;; Type that holds virtual table.
    i32 0,             ;; Flags
    i1 false,          ;; True if this function is optimized
    Function *,        ;; Pointer to llvm::Function
    null               ;; Function template parameters
  }
  ;;
  ;; Define the subprogram itself.
  ;;
  define i32 @main(i32 %argc, i8** %argv) {
  ...
  }

C/C++ basic types
-----------------

The following are the basic type descriptors for C/C++ core types:

bool
^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"bool",  ;; Name
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 8,             ;; Size in Bits
    i64 8,             ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 2              ;; Encoding
  }

char
^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"char",  ;; Name
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 8,             ;; Size in Bits
    i64 8,             ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 6              ;; Encoding
  }

unsigned char
^^^^^^^^^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"unsigned char",
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 8,             ;; Size in Bits
    i64 8,             ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 8              ;; Encoding
  }

short
^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"short int",
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 16,            ;; Size in Bits
    i64 16,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 5              ;; Encoding
  }

unsigned short
^^^^^^^^^^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"short unsigned int",
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 16,            ;; Size in Bits
    i64 16,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 7              ;; Encoding
  }

int
^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"int",   ;; Name
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 32,            ;; Size in Bits
    i64 32,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 5              ;; Encoding
  }

unsigned int
^^^^^^^^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"unsigned int",
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 32,            ;; Size in Bits
    i64 32,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 7              ;; Encoding
  }

long long
^^^^^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"long long int",
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 64,            ;; Size in Bits
    i64 64,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 5              ;; Encoding
  }

unsigned long long
^^^^^^^^^^^^^^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"long long unsigned int",
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 64,            ;; Size in Bits
    i64 64,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 7              ;; Encoding
  }

float
^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"float",
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 32,            ;; Size in Bits
    i64 32,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 4              ;; Encoding
  }

double
^^^^^^

.. code-block:: llvm

  !2 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"double",;; Name
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 64,            ;; Size in Bits
    i64 64,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 4              ;; Encoding
  }

C/C++ derived types
-------------------

Given the following as an example of C/C++ derived type:

.. code-block:: c

  typedef const int *IntPtr;

a C/C++ front-end would generate the following descriptors:

.. code-block:: llvm

  ;;
  ;; Define the typedef "IntPtr".
  ;;
  !2 = metadata !{
    i32 524310,          ;; Tag
    metadata !1,         ;; Context
    metadata !"IntPtr",  ;; Name
    metadata !3,         ;; File
    i32 0,               ;; Line number
    i64 0,               ;; Size in bits
    i64 0,               ;; Align in bits
    i64 0,               ;; Offset in bits
    i32 0,               ;; Flags
    metadata !4          ;; Derived From type
  }
  ;;
  ;; Define the pointer type.
  ;;
  !4 = metadata !{
    i32 524303,          ;; Tag
    metadata !1,         ;; Context
    metadata !"",        ;; Name
    metadata !1,         ;; File
    i32 0,               ;; Line number
    i64 64,              ;; Size in bits
    i64 64,              ;; Align in bits
    i64 0,               ;; Offset in bits
    i32 0,               ;; Flags
    metadata !5          ;; Derived From type
  }
  ;;
  ;; Define the const type.
  ;;
  !5 = metadata !{
    i32 524326,          ;; Tag
    metadata !1,         ;; Context
    metadata !"",        ;; Name
    metadata !1,         ;; File
    i32 0,               ;; Line number
    i64 32,              ;; Size in bits
    i64 32,              ;; Align in bits
    i64 0,               ;; Offset in bits
    i32 0,               ;; Flags
    metadata !6          ;; Derived From type
  }
  ;;
  ;; Define the int type.
  ;;
  !6 = metadata !{
    i32 524324,          ;; Tag
    metadata !1,         ;; Context
    metadata !"int",     ;; Name
    metadata !1,         ;; File
    i32 0,               ;; Line number
    i64 32,              ;; Size in bits
    i64 32,              ;; Align in bits
    i64 0,               ;; Offset in bits
    i32 0,               ;; Flags
    5                    ;; Encoding
  }

C/C++ struct/union types
------------------------

Given the following as an example of C/C++ struct type:

.. code-block:: c

  struct Color {
    unsigned Red;
    unsigned Green;
    unsigned Blue;
  };

a C/C++ front-end would generate the following descriptors:

.. code-block:: llvm

  ;;
  ;; Define basic type for unsigned int.
  ;;
  !5 = metadata !{
    i32 524324,        ;; Tag
    metadata !1,       ;; Context
    metadata !"unsigned int",
    metadata !1,       ;; File
    i32 0,             ;; Line number
    i64 32,            ;; Size in Bits
    i64 32,            ;; Align in Bits
    i64 0,             ;; Offset in Bits
    i32 0,             ;; Flags
    i32 7              ;; Encoding
  }
  ;;
  ;; Define composite type for struct Color.
  ;;
  !2 = metadata !{
    i32 524307,        ;; Tag
    metadata !1,       ;; Context
    metadata !"Color", ;; Name
    metadata !1,       ;; Compile unit
    i32 1,             ;; Line number
    i64 96,            ;; Size in bits
    i64 32,            ;; Align in bits
    i64 0,             ;; Offset in bits
    i32 0,             ;; Flags
    null,              ;; Derived From
    metadata !3,       ;; Elements
    i32 0              ;; Runtime Language
  }

  ;;
  ;; Define the Red field.
  ;;
  !4 = metadata !{
    i32 524301,        ;; Tag
    metadata !1,       ;; Context
    metadata !"Red",   ;; Name
    metadata !1,       ;; File
    i32 2,             ;; Line number
    i64 32,            ;; Size in bits
    i64 32,            ;; Align in bits
    i64 0,             ;; Offset in bits
    i32 0,             ;; Flags
    metadata !5        ;; Derived From type
  }

  ;;
  ;; Define the Green field.
  ;;
  !6 = metadata !{
    i32 524301,        ;; Tag
    metadata !1,       ;; Context
    metadata !"Green", ;; Name
    metadata !1,       ;; File
    i32 3,             ;; Line number
    i64 32,            ;; Size in bits
    i64 32,            ;; Align in bits
    i64 32,             ;; Offset in bits
    i32 0,             ;; Flags
    metadata !5        ;; Derived From type
  }

  ;;
  ;; Define the Blue field.
  ;;
  !7 = metadata !{
    i32 524301,        ;; Tag
    metadata !1,       ;; Context
    metadata !"Blue",  ;; Name
    metadata !1,       ;; File
    i32 4,             ;; Line number
    i64 32,            ;; Size in bits
    i64 32,            ;; Align in bits
    i64 64,             ;; Offset in bits
    i32 0,             ;; Flags
    metadata !5        ;; Derived From type
  }

  ;;
  ;; Define the array of fields used by the composite type Color.
  ;;
  !3 = metadata !{metadata !4, metadata !6, metadata !7}

C/C++ enumeration types
-----------------------

Given the following as an example of C/C++ enumeration type:

.. code-block:: c

  enum Trees {
    Spruce = 100,
    Oak = 200,
    Maple = 300
  };

a C/C++ front-end would generate the following descriptors:

.. code-block:: llvm

  ;;
  ;; Define composite type for enum Trees
  ;;
  !2 = metadata !{
    i32 524292,        ;; Tag
    metadata !1,       ;; Context
    metadata !"Trees", ;; Name
    metadata !1,       ;; File
    i32 1,             ;; Line number
    i64 32,            ;; Size in bits
    i64 32,            ;; Align in bits
    i64 0,             ;; Offset in bits
    i32 0,             ;; Flags
    null,              ;; Derived From type
    metadata !3,       ;; Elements
    i32 0              ;; Runtime language
  }

  ;;
  ;; Define the array of enumerators used by composite type Trees.
  ;;
  !3 = metadata !{metadata !4, metadata !5, metadata !6}

  ;;
  ;; Define Spruce enumerator.
  ;;
  !4 = metadata !{i32 524328, metadata !"Spruce", i64 100}

  ;;
  ;; Define Oak enumerator.
  ;;
  !5 = metadata !{i32 524328, metadata !"Oak", i64 200}

  ;;
  ;; Define Maple enumerator.
  ;;
  !6 = metadata !{i32 524328, metadata !"Maple", i64 300}

Debugging information format
============================

Debugging Information Extension for Objective C Properties
----------------------------------------------------------

Introduction
^^^^^^^^^^^^

Objective C provides a simpler way to declare and define accessor methods using
declared properties.  The language provides features to declare a property and
to let compiler synthesize accessor methods.

The debugger lets developer inspect Objective C interfaces and their instance
variables and class variables.  However, the debugger does not know anything
about the properties defined in Objective C interfaces.  The debugger consumes
information generated by compiler in DWARF format.  The format does not support
encoding of Objective C properties.  This proposal describes DWARF extensions to
encode Objective C properties, which the debugger can use to let developers
inspect Objective C properties.

Proposal
^^^^^^^^

Objective C properties exist separately from class members.  A property can be
defined only by "setter" and "getter" selectors, and be calculated anew on each
access.  Or a property can just be a direct access to some declared ivar.
Finally it can have an ivar "automatically synthesized" for it by the compiler,
in which case the property can be referred to in user code directly using the
standard C dereference syntax as well as through the property "dot" syntax, but
there is no entry in the ``@interface`` declaration corresponding to this ivar.

To facilitate debugging, these properties we will add a new DWARF TAG into the
``DW_TAG_structure_type`` definition for the class to hold the description of a
given property, and a set of DWARF attributes that provide said description.
The property tag will also contain the name and declared type of the property.

If there is a related ivar, there will also be a DWARF property attribute placed
in the ``DW_TAG_member`` DIE for that ivar referring back to the property TAG
for that property.  And in the case where the compiler synthesizes the ivar
directly, the compiler is expected to generate a ``DW_TAG_member`` for that
ivar (with the ``DW_AT_artificial`` set to 1), whose name will be the name used
to access this ivar directly in code, and with the property attribute pointing
back to the property it is backing.

The following examples will serve as illustration for our discussion:

.. code-block:: objc

  @interface I1 {
    int n2;
  }

  @property int p1;
  @property int p2;
  @end

  @implementation I1
  @synthesize p1;
  @synthesize p2 = n2;
  @end

This produces the following DWARF (this is a "pseudo dwarfdump" output):

.. code-block:: none

  0x00000100:  TAG_structure_type [7] *
                 AT_APPLE_runtime_class( 0x10 )
                 AT_name( "I1" )
                 AT_decl_file( "Objc_Property.m" )
                 AT_decl_line( 3 )

  0x00000110    TAG_APPLE_property
                  AT_name ( "p1" )
                  AT_type ( {0x00000150} ( int ) )

  0x00000120:   TAG_APPLE_property
                  AT_name ( "p2" )
                  AT_type ( {0x00000150} ( int ) )

  0x00000130:   TAG_member [8]
                  AT_name( "_p1" )
                  AT_APPLE_property ( {0x00000110} "p1" )
                  AT_type( {0x00000150} ( int ) )
                  AT_artificial ( 0x1 )

  0x00000140:    TAG_member [8]
                   AT_name( "n2" )
                   AT_APPLE_property ( {0x00000120} "p2" )
                   AT_type( {0x00000150} ( int ) )

  0x00000150:  AT_type( ( int ) )

Note, the current convention is that the name of the ivar for an
auto-synthesized property is the name of the property from which it derives
with an underscore prepended, as is shown in the example.  But we actually
don't need to know this convention, since we are given the name of the ivar
directly.

Also, it is common practice in ObjC to have different property declarations in
the @interface and @implementation - e.g. to provide a read-only property in
the interface,and a read-write interface in the implementation.  In that case,
the compiler should emit whichever property declaration will be in force in the
current translation unit.

Developers can decorate a property with attributes which are encoded using
``DW_AT_APPLE_property_attribute``.

.. code-block:: objc

  @property (readonly, nonatomic) int pr;

.. code-block:: none

  TAG_APPLE_property [8]
    AT_name( "pr" )
    AT_type ( {0x00000147} (int) )
    AT_APPLE_property_attribute (DW_APPLE_PROPERTY_readonly, DW_APPLE_PROPERTY_nonatomic)

The setter and getter method names are attached to the property using
``DW_AT_APPLE_property_setter`` and ``DW_AT_APPLE_property_getter`` attributes.

.. code-block:: objc

  @interface I1
  @property (setter=myOwnP3Setter:) int p3;
  -(void)myOwnP3Setter:(int)a;
  @end

  @implementation I1
  @synthesize p3;
  -(void)myOwnP3Setter:(int)a{ }
  @end

The DWARF for this would be:

.. code-block:: none

  0x000003bd: TAG_structure_type [7] *
                AT_APPLE_runtime_class( 0x10 )
                AT_name( "I1" )
                AT_decl_file( "Objc_Property.m" )
                AT_decl_line( 3 )

  0x000003cd      TAG_APPLE_property
                    AT_name ( "p3" )
                    AT_APPLE_property_setter ( "myOwnP3Setter:" )
                    AT_type( {0x00000147} ( int ) )

  0x000003f3:     TAG_member [8]
                    AT_name( "_p3" )
                    AT_type ( {0x00000147} ( int ) )
                    AT_APPLE_property ( {0x000003cd} )
                    AT_artificial ( 0x1 )

New DWARF Tags
^^^^^^^^^^^^^^

+-----------------------+--------+
| TAG                   | Value  |
+=======================+========+
| DW_TAG_APPLE_property | 0x4200 |
+-----------------------+--------+

New DWARF Attributes
^^^^^^^^^^^^^^^^^^^^

+--------------------------------+--------+-----------+
| Attribute                      | Value  | Classes   |
+================================+========+===========+
| DW_AT_APPLE_property           | 0x3fed | Reference |
+--------------------------------+--------+-----------+
| DW_AT_APPLE_property_getter    | 0x3fe9 | String    |
+--------------------------------+--------+-----------+
| DW_AT_APPLE_property_setter    | 0x3fea | String    |
+--------------------------------+--------+-----------+
| DW_AT_APPLE_property_attribute | 0x3feb | Constant  |
+--------------------------------+--------+-----------+

New DWARF Constants
^^^^^^^^^^^^^^^^^^^

+--------------------------------+-------+
| Name                           | Value |
+================================+=======+
| DW_AT_APPLE_PROPERTY_readonly  | 0x1   |
+--------------------------------+-------+
| DW_AT_APPLE_PROPERTY_readwrite | 0x2   |
+--------------------------------+-------+
| DW_AT_APPLE_PROPERTY_assign    | 0x4   |
+--------------------------------+-------+
| DW_AT_APPLE_PROPERTY_retain    | 0x8   |
+--------------------------------+-------+
| DW_AT_APPLE_PROPERTY_copy      | 0x10  |
+--------------------------------+-------+
| DW_AT_APPLE_PROPERTY_nonatomic | 0x20  |
+--------------------------------+-------+

Name Accelerator Tables
-----------------------

Introduction
^^^^^^^^^^^^

The "``.debug_pubnames``" and "``.debug_pubtypes``" formats are not what a
debugger needs.  The "``pub``" in the section name indicates that the entries
in the table are publicly visible names only.  This means no static or hidden
functions show up in the "``.debug_pubnames``".  No static variables or private
class variables are in the "``.debug_pubtypes``".  Many compilers add different
things to these tables, so we can't rely upon the contents between gcc, icc, or
clang.

The typical query given by users tends not to match up with the contents of
these tables.  For example, the DWARF spec states that "In the case of the name
of a function member or static data member of a C++ structure, class or union,
the name presented in the "``.debug_pubnames``" section is not the simple name
given by the ``DW_AT_name attribute`` of the referenced debugging information
entry, but rather the fully qualified name of the data or function member."
So the only names in these tables for complex C++ entries is a fully
qualified name.  Debugger users tend not to enter their search strings as
"``a::b::c(int,const Foo&) const``", but rather as "``c``", "``b::c``" , or
"``a::b::c``".  So the name entered in the name table must be demangled in
order to chop it up appropriately and additional names must be manually entered
into the table to make it effective as a name lookup table for debuggers to
se.

All debuggers currently ignore the "``.debug_pubnames``" table as a result of
its inconsistent and useless public-only name content making it a waste of
space in the object file.  These tables, when they are written to disk, are not
sorted in any way, leaving every debugger to do its own parsing and sorting.
These tables also include an inlined copy of the string values in the table
itself making the tables much larger than they need to be on disk, especially
for large C++ programs.

Can't we just fix the sections by adding all of the names we need to this
table? No, because that is not what the tables are defined to contain and we
won't know the difference between the old bad tables and the new good tables.
At best we could make our own renamed sections that contain all of the data we
need.

These tables are also insufficient for what a debugger like LLDB needs.  LLDB
uses clang for its expression parsing where LLDB acts as a PCH.  LLDB is then
often asked to look for type "``foo``" or namespace "``bar``", or list items in
namespace "``baz``".  Namespaces are not included in the pubnames or pubtypes
tables.  Since clang asks a lot of questions when it is parsing an expression,
we need to be very fast when looking up names, as it happens a lot.  Having new
accelerator tables that are optimized for very quick lookups will benefit this
type of debugging experience greatly.

We would like to generate name lookup tables that can be mapped into memory
from disk, and used as is, with little or no up-front parsing.  We would also
be able to control the exact content of these different tables so they contain
exactly what we need.  The Name Accelerator Tables were designed to fix these
issues.  In order to solve these issues we need to:

* Have a format that can be mapped into memory from disk and used as is
* Lookups should be very fast
* Extensible table format so these tables can be made by many producers
* Contain all of the names needed for typical lookups out of the box
* Strict rules for the contents of tables

Table size is important and the accelerator table format should allow the reuse
of strings from common string tables so the strings for the names are not
duplicated.  We also want to make sure the table is ready to be used as-is by
simply mapping the table into memory with minimal header parsing.

The name lookups need to be fast and optimized for the kinds of lookups that
debuggers tend to do.  Optimally we would like to touch as few parts of the
mapped table as possible when doing a name lookup and be able to quickly find
the name entry we are looking for, or discover there are no matches.  In the
case of debuggers we optimized for lookups that fail most of the time.

Each table that is defined should have strict rules on exactly what is in the
accelerator tables and documented so clients can rely on the content.

Hash Tables
^^^^^^^^^^^

Standard Hash Tables
""""""""""""""""""""

Typical hash tables have a header, buckets, and each bucket points to the
bucket contents:

.. code-block:: none

  .------------.
  |  HEADER    |
  |------------|
  |  BUCKETS   |
  |------------|
  |  DATA      |
  `------------'

The BUCKETS are an array of offsets to DATA for each hash:

.. code-block:: none

  .------------.
  | 0x00001000 | BUCKETS[0]
  | 0x00002000 | BUCKETS[1]
  | 0x00002200 | BUCKETS[2]
  | 0x000034f0 | BUCKETS[3]
  |            | ...
  | 0xXXXXXXXX | BUCKETS[n_buckets]
  '------------'

So for ``bucket[3]`` in the example above, we have an offset into the table
0x000034f0 which points to a chain of entries for the bucket.  Each bucket must
contain a next pointer, full 32 bit hash value, the string itself, and the data
for the current string value.

.. code-block:: none

              .------------.
  0x000034f0: | 0x00003500 | next pointer
              | 0x12345678 | 32 bit hash
              | "erase"    | string value
              | data[n]    | HashData for this bucket
              |------------|
  0x00003500: | 0x00003550 | next pointer
              | 0x29273623 | 32 bit hash
              | "dump"     | string value
              | data[n]    | HashData for this bucket
              |------------|
  0x00003550: | 0x00000000 | next pointer
              | 0x82638293 | 32 bit hash
              | "main"     | string value
              | data[n]    | HashData for this bucket
              `------------'

The problem with this layout for debuggers is that we need to optimize for the
negative lookup case where the symbol we're searching for is not present.  So
if we were to lookup "``printf``" in the table above, we would make a 32 hash
for "``printf``", it might match ``bucket[3]``.  We would need to go to the
offset 0x000034f0 and start looking to see if our 32 bit hash matches.  To do
so, we need to read the next pointer, then read the hash, compare it, and skip
to the next bucket.  Each time we are skipping many bytes in memory and
touching new cache pages just to do the compare on the full 32 bit hash.  All
of these accesses then tell us that we didn't have a match.

Name Hash Tables
""""""""""""""""

To solve the issues mentioned above we have structured the hash tables a bit
differently: a header, buckets, an array of all unique 32 bit hash values,
followed by an array of hash value data offsets, one for each hash value, then
the data for all hash values:

.. code-block:: none

  .-------------.
  |  HEADER     |
  |-------------|
  |  BUCKETS    |
  |-------------|
  |  HASHES     |
  |-------------|
  |  OFFSETS    |
  |-------------|
  |  DATA       |
  `-------------'

The ``BUCKETS`` in the name tables are an index into the ``HASHES`` array.  By
making all of the full 32 bit hash values contiguous in memory, we allow
ourselves to efficiently check for a match while touching as little memory as
possible.  Most often checking the 32 bit hash values is as far as the lookup
goes.  If it does match, it usually is a match with no collisions.  So for a
table with "``n_buckets``" buckets, and "``n_hashes``" unique 32 bit hash
values, we can clarify the contents of the ``BUCKETS``, ``HASHES`` and
``OFFSETS`` as:

.. code-block:: none

  .-------------------------.
  |  HEADER.magic           | uint32_t
  |  HEADER.version         | uint16_t
  |  HEADER.hash_function   | uint16_t
  |  HEADER.bucket_count    | uint32_t
  |  HEADER.hashes_count    | uint32_t
  |  HEADER.header_data_len | uint32_t
  |  HEADER_DATA            | HeaderData
  |-------------------------|
  |  BUCKETS                | uint32_t[bucket_count] // 32 bit hash indexes
  |-------------------------|
  |  HASHES                 | uint32_t[hashes_count] // 32 bit hash values
  |-------------------------|
  |  OFFSETS                | uint32_t[hashes_count] // 32 bit offsets to hash value data
  |-------------------------|
  |  ALL HASH DATA          |
  `-------------------------'

So taking the exact same data from the standard hash example above we end up
with:

.. code-block:: none

              .------------.
              | HEADER     |
              |------------|
              |          0 | BUCKETS[0]
              |          2 | BUCKETS[1]
              |          5 | BUCKETS[2]
              |          6 | BUCKETS[3]
              |            | ...
              |        ... | BUCKETS[n_buckets]
              |------------|
              | 0x........ | HASHES[0]
              | 0x........ | HASHES[1]
              | 0x........ | HASHES[2]
              | 0x........ | HASHES[3]
              | 0x........ | HASHES[4]
              | 0x........ | HASHES[5]
              | 0x12345678 | HASHES[6]    hash for BUCKETS[3]
              | 0x29273623 | HASHES[7]    hash for BUCKETS[3]
              | 0x82638293 | HASHES[8]    hash for BUCKETS[3]
              | 0x........ | HASHES[9]
              | 0x........ | HASHES[10]
              | 0x........ | HASHES[11]
              | 0x........ | HASHES[12]
              | 0x........ | HASHES[13]
              | 0x........ | HASHES[n_hashes]
              |------------|
              | 0x........ | OFFSETS[0]
              | 0x........ | OFFSETS[1]
              | 0x........ | OFFSETS[2]
              | 0x........ | OFFSETS[3]
              | 0x........ | OFFSETS[4]
              | 0x........ | OFFSETS[5]
              | 0x000034f0 | OFFSETS[6]   offset for BUCKETS[3]
              | 0x00003500 | OFFSETS[7]   offset for BUCKETS[3]
              | 0x00003550 | OFFSETS[8]   offset for BUCKETS[3]
              | 0x........ | OFFSETS[9]
              | 0x........ | OFFSETS[10]
              | 0x........ | OFFSETS[11]
              | 0x........ | OFFSETS[12]
              | 0x........ | OFFSETS[13]
              | 0x........ | OFFSETS[n_hashes]
              |------------|
              |            |
              |            |
              |            |
              |            |
              |            |
              |------------|
  0x000034f0: | 0x00001203 | .debug_str ("erase")
              | 0x00000004 | A 32 bit array count - number of HashData with name "erase"
              | 0x........ | HashData[0]
              | 0x........ | HashData[1]
              | 0x........ | HashData[2]
              | 0x........ | HashData[3]
              | 0x00000000 | String offset into .debug_str (terminate data for hash)
              |------------|
  0x00003500: | 0x00001203 | String offset into .debug_str ("collision")
              | 0x00000002 | A 32 bit array count - number of HashData with name "collision"
              | 0x........ | HashData[0]
              | 0x........ | HashData[1]
              | 0x00001203 | String offset into .debug_str ("dump")
              | 0x00000003 | A 32 bit array count - number of HashData with name "dump"
              | 0x........ | HashData[0]
              | 0x........ | HashData[1]
              | 0x........ | HashData[2]
              | 0x00000000 | String offset into .debug_str (terminate data for hash)
              |------------|
  0x00003550: | 0x00001203 | String offset into .debug_str ("main")
              | 0x00000009 | A 32 bit array count - number of HashData with name "main"
              | 0x........ | HashData[0]
              | 0x........ | HashData[1]
              | 0x........ | HashData[2]
              | 0x........ | HashData[3]
              | 0x........ | HashData[4]
              | 0x........ | HashData[5]
              | 0x........ | HashData[6]
              | 0x........ | HashData[7]
              | 0x........ | HashData[8]
              | 0x00000000 | String offset into .debug_str (terminate data for hash)
              `------------'

So we still have all of the same data, we just organize it more efficiently for
debugger lookup.  If we repeat the same "``printf``" lookup from above, we
would hash "``printf``" and find it matches ``BUCKETS[3]`` by taking the 32 bit
hash value and modulo it by ``n_buckets``.  ``BUCKETS[3]`` contains "6" which
is the index into the ``HASHES`` table.  We would then compare any consecutive
32 bit hashes values in the ``HASHES`` array as long as the hashes would be in
``BUCKETS[3]``.  We do this by verifying that each subsequent hash value modulo
``n_buckets`` is still 3.  In the case of a failed lookup we would access the
memory for ``BUCKETS[3]``, and then compare a few consecutive 32 bit hashes
before we know that we have no match.  We don't end up marching through
multiple words of memory and we really keep the number of processor data cache
lines being accessed as small as possible.

The string hash that is used for these lookup tables is the Daniel J.
Bernstein hash which is also used in the ELF ``GNU_HASH`` sections.  It is a
very good hash for all kinds of names in programs with very few hash
collisions.

Empty buckets are designated by using an invalid hash index of ``UINT32_MAX``.

Details
^^^^^^^

These name hash tables are designed to be generic where specializations of the
table get to define additional data that goes into the header ("``HeaderData``"),
how the string value is stored ("``KeyType``") and the content of the data for each
hash value.

Header Layout
"""""""""""""

The header has a fixed part, and the specialized part.  The exact format of the
header is:

.. code-block:: c

  struct Header
  {
    uint32_t   magic;           // 'HASH' magic value to allow endian detection
    uint16_t   version;         // Version number
    uint16_t   hash_function;   // The hash function enumeration that was used
    uint32_t   bucket_count;    // The number of buckets in this hash table
    uint32_t   hashes_count;    // The total number of unique hash values and hash data offsets in this table
    uint32_t   header_data_len; // The bytes to skip to get to the hash indexes (buckets) for correct alignment
                                // Specifically the length of the following HeaderData field - this does not
                                // include the size of the preceding fields
    HeaderData header_data;     // Implementation specific header data
  };

The header starts with a 32 bit "``magic``" value which must be ``'HASH'``
encoded as an ASCII integer.  This allows the detection of the start of the
hash table and also allows the table's byte order to be determined so the table
can be correctly extracted.  The "``magic``" value is followed by a 16 bit
``version`` number which allows the table to be revised and modified in the
future.  The current version number is 1. ``hash_function`` is a ``uint16_t``
enumeration that specifies which hash function was used to produce this table.
The current values for the hash function enumerations include:

.. code-block:: c

  enum HashFunctionType
  {
    eHashFunctionDJB = 0u, // Daniel J Bernstein hash function
  };

``bucket_count`` is a 32 bit unsigned integer that represents how many buckets
are in the ``BUCKETS`` array.  ``hashes_count`` is the number of unique 32 bit
hash values that are in the ``HASHES`` array, and is the same number of offsets
are contained in the ``OFFSETS`` array.  ``header_data_len`` specifies the size
in bytes of the ``HeaderData`` that is filled in by specialized versions of
this table.

Fixed Lookup
""""""""""""

The header is followed by the buckets, hashes, offsets, and hash value data.

.. code-block:: c

  struct FixedTable
  {
    uint32_t buckets[Header.bucket_count];  // An array of hash indexes into the "hashes[]" array below
    uint32_t hashes [Header.hashes_count];  // Every unique 32 bit hash for the entire table is in this table
    uint32_t offsets[Header.hashes_count];  // An offset that corresponds to each item in the "hashes[]" array above
  };

``buckets`` is an array of 32 bit indexes into the ``hashes`` array.  The
``hashes`` array contains all of the 32 bit hash values for all names in the
hash table.  Each hash in the ``hashes`` table has an offset in the ``offsets``
array that points to the data for the hash value.

This table setup makes it very easy to repurpose these tables to contain
different data, while keeping the lookup mechanism the same for all tables.
This layout also makes it possible to save the table to disk and map it in
later and do very efficient name lookups with little or no parsing.

DWARF lookup tables can be implemented in a variety of ways and can store a lot
of information for each name.  We want to make the DWARF tables extensible and
able to store the data efficiently so we have used some of the DWARF features
that enable efficient data storage to define exactly what kind of data we store
for each name.

The ``HeaderData`` contains a definition of the contents of each HashData chunk.
We might want to store an offset to all of the debug information entries (DIEs)
for each name.  To keep things extensible, we create a list of items, or
Atoms, that are contained in the data for each name.  First comes the type of
the data in each atom:

.. code-block:: c

  enum AtomType
  {
    eAtomTypeNULL       = 0u,
    eAtomTypeDIEOffset  = 1u,   // DIE offset, check form for encoding
    eAtomTypeCUOffset   = 2u,   // DIE offset of the compiler unit header that contains the item in question
    eAtomTypeTag        = 3u,   // DW_TAG_xxx value, should be encoded as DW_FORM_data1 (if no tags exceed 255) or DW_FORM_data2
    eAtomTypeNameFlags  = 4u,   // Flags from enum NameFlags
    eAtomTypeTypeFlags  = 5u,   // Flags from enum TypeFlags
  };

The enumeration values and their meanings are:

.. code-block:: none

  eAtomTypeNULL       - a termination atom that specifies the end of the atom list
  eAtomTypeDIEOffset  - an offset into the .debug_info section for the DWARF DIE for this name
  eAtomTypeCUOffset   - an offset into the .debug_info section for the CU that contains the DIE
  eAtomTypeDIETag     - The DW_TAG_XXX enumeration value so you don't have to parse the DWARF to see what it is
  eAtomTypeNameFlags  - Flags for functions and global variables (isFunction, isInlined, isExternal...)
  eAtomTypeTypeFlags  - Flags for types (isCXXClass, isObjCClass, ...)

Then we allow each atom type to define the atom type and how the data for each
atom type data is encoded:

.. code-block:: c

  struct Atom
  {
    uint16_t type;  // AtomType enum value
    uint16_t form;  // DWARF DW_FORM_XXX defines
  };

The ``form`` type above is from the DWARF specification and defines the exact
encoding of the data for the Atom type.  See the DWARF specification for the
``DW_FORM_`` definitions.

.. code-block:: c

  struct HeaderData
  {
    uint32_t die_offset_base;
    uint32_t atom_count;
    Atoms    atoms[atom_count0];
  };

``HeaderData`` defines the base DIE offset that should be added to any atoms
that are encoded using the ``DW_FORM_ref1``, ``DW_FORM_ref2``,
``DW_FORM_ref4``, ``DW_FORM_ref8`` or ``DW_FORM_ref_udata``.  It also defines
what is contained in each ``HashData`` object -- ``Atom.form`` tells us how large
each field will be in the ``HashData`` and the ``Atom.type`` tells us how this data
should be interpreted.

For the current implementations of the "``.apple_names``" (all functions +
globals), the "``.apple_types``" (names of all types that are defined), and
the "``.apple_namespaces``" (all namespaces), we currently set the ``Atom``
array to be:

.. code-block:: c

  HeaderData.atom_count = 1;
  HeaderData.atoms[0].type = eAtomTypeDIEOffset;
  HeaderData.atoms[0].form = DW_FORM_data4;

This defines the contents to be the DIE offset (eAtomTypeDIEOffset) that is
  encoded as a 32 bit value (DW_FORM_data4).  This allows a single name to have
  multiple matching DIEs in a single file, which could come up with an inlined
  function for instance.  Future tables could include more information about the
  DIE such as flags indicating if the DIE is a function, method, block,
  or inlined.

The KeyType for the DWARF table is a 32 bit string table offset into the
  ".debug_str" table.  The ".debug_str" is the string table for the DWARF which
  may already contain copies of all of the strings.  This helps make sure, with
  help from the compiler, that we reuse the strings between all of the DWARF
  sections and keeps the hash table size down.  Another benefit to having the
  compiler generate all strings as DW_FORM_strp in the debug info, is that
  DWARF parsing can be made much faster.

After a lookup is made, we get an offset into the hash data.  The hash data
  needs to be able to deal with 32 bit hash collisions, so the chunk of data
  at the offset in the hash data consists of a triple:

.. code-block:: c

  uint32_t str_offset
  uint32_t hash_data_count
  HashData[hash_data_count]

If "str_offset" is zero, then the bucket contents are done. 99.9% of the
  hash data chunks contain a single item (no 32 bit hash collision):

.. code-block:: none

  .------------.
  | 0x00001023 | uint32_t KeyType (.debug_str[0x0001023] => "main")
  | 0x00000004 | uint32_t HashData count
  | 0x........ | uint32_t HashData[0] DIE offset
  | 0x........ | uint32_t HashData[1] DIE offset
  | 0x........ | uint32_t HashData[2] DIE offset
  | 0x........ | uint32_t HashData[3] DIE offset
  | 0x00000000 | uint32_t KeyType (end of hash chain)
  `------------'

If there are collisions, you will have multiple valid string offsets:

.. code-block:: none

  .------------.
  | 0x00001023 | uint32_t KeyType (.debug_str[0x0001023] => "main")
  | 0x00000004 | uint32_t HashData count
  | 0x........ | uint32_t HashData[0] DIE offset
  | 0x........ | uint32_t HashData[1] DIE offset
  | 0x........ | uint32_t HashData[2] DIE offset
  | 0x........ | uint32_t HashData[3] DIE offset
  | 0x00002023 | uint32_t KeyType (.debug_str[0x0002023] => "print")
  | 0x00000002 | uint32_t HashData count
  | 0x........ | uint32_t HashData[0] DIE offset
  | 0x........ | uint32_t HashData[1] DIE offset
  | 0x00000000 | uint32_t KeyType (end of hash chain)
  `------------'

Current testing with real world C++ binaries has shown that there is around 1
32 bit hash collision per 100,000 name entries.

Contents
^^^^^^^^

As we said, we want to strictly define exactly what is included in the
different tables.  For DWARF, we have 3 tables: "``.apple_names``",
"``.apple_types``", and "``.apple_namespaces``".

"``.apple_names``" sections should contain an entry for each DWARF DIE whose
``DW_TAG`` is a ``DW_TAG_label``, ``DW_TAG_inlined_subroutine``, or
``DW_TAG_subprogram`` that has address attributes: ``DW_AT_low_pc``,
``DW_AT_high_pc``, ``DW_AT_ranges`` or ``DW_AT_entry_pc``.  It also contains
``DW_TAG_variable`` DIEs that have a ``DW_OP_addr`` in the location (global and
static variables).  All global and static variables should be included,
including those scoped within functions and classes.  For example using the
following code:

.. code-block:: c

  static int var = 0;

  void f ()
  {
    static int var = 0;
  }

Both of the static ``var`` variables would be included in the table.  All
functions should emit both their full names and their basenames.  For C or C++,
the full name is the mangled name (if available) which is usually in the
``DW_AT_MIPS_linkage_name`` attribute, and the ``DW_AT_name`` contains the
function basename.  If global or static variables have a mangled name in a
``DW_AT_MIPS_linkage_name`` attribute, this should be emitted along with the
simple name found in the ``DW_AT_name`` attribute.

"``.apple_types``" sections should contain an entry for each DWARF DIE whose
tag is one of:

* DW_TAG_array_type
* DW_TAG_class_type
* DW_TAG_enumeration_type
* DW_TAG_pointer_type
* DW_TAG_reference_type
* DW_TAG_string_type
* DW_TAG_structure_type
* DW_TAG_subroutine_type
* DW_TAG_typedef
* DW_TAG_union_type
* DW_TAG_ptr_to_member_type
* DW_TAG_set_type
* DW_TAG_subrange_type
* DW_TAG_base_type
* DW_TAG_const_type
* DW_TAG_constant
* DW_TAG_file_type
* DW_TAG_namelist
* DW_TAG_packed_type
* DW_TAG_volatile_type
* DW_TAG_restrict_type
* DW_TAG_interface_type
* DW_TAG_unspecified_type
* DW_TAG_shared_type

Only entries with a ``DW_AT_name`` attribute are included, and the entry must
not be a forward declaration (``DW_AT_declaration`` attribute with a non-zero
value).  For example, using the following code:

.. code-block:: c

  int main ()
  {
    int *b = 0;
    return *b;
  }

We get a few type DIEs:

.. code-block:: none

  0x00000067:     TAG_base_type [5]
                  AT_encoding( DW_ATE_signed )
                  AT_name( "int" )
                  AT_byte_size( 0x04 )

  0x0000006e:     TAG_pointer_type [6]
                  AT_type( {0x00000067} ( int ) )
                  AT_byte_size( 0x08 )

The DW_TAG_pointer_type is not included because it does not have a ``DW_AT_name``.

"``.apple_namespaces``" section should contain all ``DW_TAG_namespace`` DIEs.
If we run into a namespace that has no name this is an anonymous namespace, and
the name should be output as "``(anonymous namespace)``" (without the quotes).
Why?  This matches the output of the ``abi::cxa_demangle()`` that is in the
standard C++ library that demangles mangled names.


Language Extensions and File Format Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Objective-C Extensions
""""""""""""""""""""""

"``.apple_objc``" section should contain all ``DW_TAG_subprogram`` DIEs for an
Objective-C class.  The name used in the hash table is the name of the
Objective-C class itself.  If the Objective-C class has a category, then an
entry is made for both the class name without the category, and for the class
name with the category.  So if we have a DIE at offset 0x1234 with a name of
method "``-[NSString(my_additions) stringWithSpecialString:]``", we would add
an entry for "``NSString``" that points to DIE 0x1234, and an entry for
"``NSString(my_additions)``" that points to 0x1234.  This allows us to quickly
track down all Objective-C methods for an Objective-C class when doing
expressions.  It is needed because of the dynamic nature of Objective-C where
anyone can add methods to a class.  The DWARF for Objective-C methods is also
emitted differently from C++ classes where the methods are not usually
contained in the class definition, they are scattered about across one or more
compile units.  Categories can also be defined in different shared libraries.
So we need to be able to quickly find all of the methods and class functions
given the Objective-C class name, or quickly find all methods and class
functions for a class + category name.  This table does not contain any
selector names, it just maps Objective-C class names (or class names +
category) to all of the methods and class functions.  The selectors are added
as function basenames in the "``.debug_names``" section.

In the "``.apple_names``" section for Objective-C functions, the full name is
the entire function name with the brackets ("``-[NSString
stringWithCString:]``") and the basename is the selector only
("``stringWithCString:``").

Mach-O Changes
""""""""""""""

The sections names for the apple hash tables are for non mach-o files.  For
mach-o files, the sections should be contained in the ``__DWARF`` segment with
names as follows:

* "``.apple_names``" -> "``__apple_names``"
* "``.apple_types``" -> "``__apple_types``"
* "``.apple_namespaces``" -> "``__apple_namespac``" (16 character limit)
* "``.apple_objc``" -> "``__apple_objc``"

