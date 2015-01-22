=====================================
Garbage Collection Safepoints in LLVM
=====================================

.. contents::
   :local:
   :depth: 2

Status
=======

This document describes a set of experimental extensions to LLVM. Use
with caution.  Because the intrinsics have experimental status,
compatibility across LLVM releases is not guaranteed.

LLVM currently supports an alternate mechanism for conservative
garbage collection support using the gc_root intrinsic.  The mechanism
described here shares little in common with the alternate
implementation and it is hoped that this mechanism will eventually
replace the gc_root mechanism.

Overview
========

To collect dead objects, garbage collectors must be able to identify
any references to objects contained within executing code, and,
depending on the collector, potentially update them.  The collector
does not need this information at all points in code - that would make
the problem much harder - but only at well-defined points in the
execution known as 'safepoints' For most collectors, it is sufficient
to track at least one copy of each unique pointer value.  However, for
a collector which wishes to relocate objects directly reachable from
running code, a higher standard is required.

One additional challenge is that the compiler may compute intermediate
results ("derived pointers") which point outside of the allocation or
even into the middle of another allocation.  The eventual use of this
intermediate value must yield an address within the bounds of the
allocation, but such "exterior derived pointers" may be visible to the
collector.  Given this, a garbage collector can not safely rely on the
runtime value of an address to indicate the object it is associated
with.  If the garbage collector wishes to move any object, the
compiler must provide a mapping, for each pointer, to an indication of
its allocation.

To simplify the interaction between a collector and the compiled code,
most garbage collectors are organized in terms of three abstractions:
load barriers, store barriers, and safepoints.

#. A load barrier is a bit of code executed immediately after the
   machine load instruction, but before any use of the value loaded.
   Depending on the collector, such a barrier may be needed for all
   loads, merely loads of a particular type (in the original source
   language), or none at all.

#. Analogously, a store barrier is a code fragement that runs
   immediately before the machine store instruction, but after the
   computation of the value stored.  The most common use of a store
   barrier is to update a 'card table' in a generational garbage
   collector.

#. A safepoint is a location at which pointers visible to the compiled
   code (i.e. currently in registers or on the stack) are allowed to
   change.  After the safepoint completes, the actual pointer value
   may differ, but the 'object' (as seen by the source language)
   pointed to will not.

  Note that the term 'safepoint' is somewhat overloaded.  It refers to
  both the location at which the machine state is parsable and the
  coordination protocol involved in bring application threads to a
  point at which the collector can safely use that information.  The
  term "statepoint" as used in this document refers exclusively to the
  former.

This document focuses on the last item - compiler support for
safepoints in generated code.  We will assume that an outside
mechanism has decided where to place safepoints.  From our
perspective, all safepoints will be function calls.  To support
relocation of objects directly reachable from values in compiled code,
the collector must be able to:

#. identify every copy of a pointer (including copies introduced by
   the compiler itself) at the safepoint,
#. identify which object each pointer relates to, and
#. potentially update each of those copies.

This document describes the mechanism by which an LLVM based compiler
can provide this information to a language runtime/collector, and
ensure that all pointers can be read and updated if desired.  The
heart of the approach is to construct (or rewrite) the IR in a manner
where the possible updates performed by the garbage collector are
explicitly visible in the IR.  Doing so requires that we:

#. create a new SSA value for each potentially relocated pointer, and
   ensure that no uses of the original (non relocated) value is
   reachable after the safepoint,
#. specify the relocation in a way which is opaque to the compiler to
   ensure that the optimizer can not introduce new uses of an
   unrelocated value after a statepoint. This prevents the optimizer
   from performing unsound optimizations.
#. recording a mapping of live pointers (and the allocation they're
   associated with) for each statepoint.

At the most abstract level, inserting a safepoint can be thought of as
replacing a call instruction with a call to a multiple return value
function which both calls the original target of the call, returns
it's result, and returns updated values for any live pointers to
garbage collected objects.

  Note that the task of identifying all live pointers to garbage
  collected values, transforming the IR to expose a pointer giving the
  base object for every such live pointer, and inserting all the
  intrinsics correctly is explicitly out of scope for this document.
  The recommended approach is described in the section of Late
  Safepoint Placement below.

This abstract function call is concretely represented by a sequence of
intrinsic calls known as a 'statepoint sequence'.


Let's consider a simple call in LLVM IR:
  todo

Depending on our language we may need to allow a safepoint during the
execution of the function called from this site.  If so, we need to
let the collector update local values in the current frame.

Let's say we need to relocate SSA values 'a', 'b', and 'c' at this
safepoint.  To represent this, we would generate the statepoint
sequence:

  todo

Ideally, this sequence would have been represented as a M argument, N
return value function (where M is the number of values being
relocated + the original call arguments and N is the original return
value + each relocated value), but LLVM does not easily support such a
representation.

Instead, the statepoint intrinsic marks the actual site of the
safepoint or statepoint.  The statepoint returns a token value (which
exists only at compile time).  To get back the original return value
of the call, we use the 'gc.result' intrinsic.  To get the relocation
of each pointer in turn, we use the 'gc.relocate' intrinsic with the
appropriate index.  Note that both the gc.relocate and gc.result are
tied to the statepoint.  The combination forms a "statepoint sequence"
and represents the entitety of a parseable call or 'statepoint'.

When lowered, this example would generate the following x86 assembly::
  put assembly here

Each of the potentially relocated values has been spilled to the
stack, and a record of that location has been recorded to the
:ref:`Stack Map section <stackmap-section>`.  If the garbage collector
needs to update any of these pointers during the call, it knows
exactly what to change.

Intrinsics
===========

'''gc.statepoint''' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

::

      declare i32
        @gc.statepoint(func_type <target>, i64 <#call args>. 
                       i64 <unused>, ... (call parameters),
                       i64 <# deopt args>, ... (deopt parameters),
                       ... (gc parameters))

Overview:
"""""""""

The statepoint intrinsic represents a call which is parse-able by the
runtime.

Operands:
"""""""""

The 'target' operand is the function actually being called.  The
target can be specified as either a symbolic LLVM function, or as an
arbitrary Value of appropriate function type.  Note that the function
type must match the signature of the callee and the types of the 'call
parameters' arguments.

The '#call args' operand is the number of arguments to the actual
call.  It must exactly match the number of arguments passed in the
'call parameters' variable length section.

The 'unused' operand is unused and likely to be removed.  Please do
not use.

The 'call parameters' arguments are simply the arguments which need to
be passed to the call target.  They will be lowered according to the
specified calling convention and otherwise handled like a normal call
instruction.  The number of arguments must exactly match what is
specified in '# call args'.  The types must match the signature of
'target'.

The 'deopt parameters' arguments contain an arbitrary list of Values
which is meaningful to the runtime.  The runtime may read any of these
values, but is assumed not to modify them.  If the garbage collector
might need to modify one of these values, it must also be listed in
the 'gc pointer' argument list.  The '# deopt args' field indicates
how many operands are to be interpreted as 'deopt parameters'.

The 'gc parameters' arguments contain every pointer to a garbage
collector object which potentially needs to be updated by the garbage
collector.  Note that the argument list must explicitly contain a base
pointer for every derived pointer listed.  The order of arguments is
unimportant.  Unlike the other variable length parameter sets, this
list is not length prefixed.

Semantics:
""""""""""

A statepoint is assumed to read and write all memory.  As a result,
memory operations can not be reordered past a statepoint.  It is
illegal to mark a statepoint as being either 'readonly' or 'readnone'.

Note that legal IR can not perform any memory operation on a 'gc
pointer' argument of the statepoint in a location statically reachable
from the statepoint.  Instead, the explicitly relocated value (from a
''gc.relocate'') must be used.

'''gc.result''' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

::

      declare type*
        @gc.result(i32 %statepoint_token)

Overview:
"""""""""

'''gc.result''' extracts the result of the original call instruction
which was replaced by the '''gc.statepoint'''.  The '''gc.result'''
intrinsic is actually a family of three intrinsics due to an
implementation limitation.  Other than the type of the return value,
the semantics are the same.

Operands:
"""""""""

The first and only argument is the '''gc.statepoint''' which starts
the safepoint sequence of which this '''gc.result'' is a part.
Despite the typing of this as a generic i32, *only* the value defined
by a '''gc.statepoint''' is legal here.

Semantics:
""""""""""

The ''gc.result'' represents the return value of the call target of
the ''statepoint''.  The type of the ''gc.result'' must exactly match
the type of the target.  If the call target returns void, there will
be no ''gc.result''.

A ''gc.result'' is modeled as a 'readnone' pure function.  It has no
side effects since it is just a projection of the return value of the
previous call represented by the ''gc.statepoint''.

'''gc.relocate''' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

::

      declare <type> addrspace(1)*
        @gc.relocate(i32 %statepoint_token, i32 %base_offset, i32 %pointer_offset)

Overview:
"""""""""

A ''gc.relocate'' returns the potentially relocated value of a pointer
at the safepoint.

Operands:
"""""""""

The first argument is the '''gc.statepoint''' which starts the
safepoint sequence of which this '''gc.relocation'' is a part.
Despite the typing of this as a generic i32, *only* the value defined
by a '''gc.statepoint''' is legal here.

The second argument is an index into the statepoints list of arguments
which specifies the base pointer for the pointer being relocated.
This index must land within the 'gc parameter' section of the
statepoint's argument list.

The third argument is an index into the statepoint's list of arguments
which specify the (potentially) derived pointer being relocated.  It
is legal for this index to be the same as the second argument
if-and-only-if a base pointer is being relocated. This index must land
within the 'gc parameter' section of the statepoint's argument list.

Semantics:
""""""""""

The return value of ''gc.relocate'' is the potentially relocated value
of the pointer specified by it's arguments.  It is unspecified how the
value of the returned pointer relates to the argument to the
''gc.statepoint'' other than that a) it points to the same source
language object with the same offset, and b) the 'based-on'
relationship of the newly relocated pointers is a projection of the
unrelocated pointers.  In particular, the integer value of the pointer
returned is unspecified.

A ''gc.relocate'' is modeled as a 'readnone' pure function.  It has no
side effects since it is just a way to extract information about work
done during the actual call modeled by the ''gc.statepoint''.


Stack Map Format
================

Locations for each pointer value which may need read and/or updated by
the runtime or collector are provided via the :ref:`Stack Map format
<stackmap-format>` specified in the PatchPoint documentation.

Each statepoint generates the following Locations:

* Constant which describes number of following deopt *Locations* (not
  operands)
* Variable number of Locations, one for each deopt parameter listed in
  the IR statepoint (same number as described by previous Constant)
* Variable number of Locations pairs, one pair for each unique pointer
  which needs relocated.  The first Location in each pair describes
  the base pointer for the object.  The second is the derived pointer
  actually being relocated.  It is guaranteed that the base pointer
  must also appear explicitly as a relocation pair if used after the
  statepoint. There may be fewer pairs then gc parameters in the IR
  statepoint. Each *unique* pair will occur at least once; duplicates
  are possible.

Note that the Locations used in each section may describe the same
physical location.  e.g. A stack slot may appear as a deopt location,
a gc base pointer, and a gc derived pointer.

The ID field of the 'StkMapRecord' for a statepoint is meaningless and
it's value is explicitly unspecified.

The LiveOut section of the StkMapRecord will be empty for a statepoint
record.

Safepoint Semantics & Verification
==================================

The fundamental correctness property for the compiled code's
correctness w.r.t. the garbage collector is a dynamic one.  It must be
the case that there is no dynamic trace such that a operation
involving a potentially relocated pointer is observably-after a
safepoint which could relocate it.  'observably-after' is this usage
means that an outside observer could observe this sequence of events
in a way which precludes the operation being performed before the
safepoint.

To understand why this 'observable-after' property is required,
consider a null comparison performed on the original copy of a
relocated pointer.  Assuming that control flow follows the safepoint,
there is no way to observe externally whether the null comparison is
performed before or after the safepoint.  (Remember, the original
Value is unmodified by the safepoint.)  The compiler is free to make
either scheduling choice.

The actual correctness property implemented is slightly stronger than
this.  We require that there be no *static path* on which a
potentially relocated pointer is 'observably-after' it may have been
relocated.  This is slightly stronger than is strictly necessary (and
thus may disallow some otherwise valid programs), but greatly
simplifies reasoning about correctness of the compiled code.

By construction, this property will be upheld by the optimizer if
correctly established in the source IR.  This is a key invariant of
the design.

The existing IR Verifier pass has been extended to check most of the
local restrictions on the intrinsics mentioned in their respective
documentation.  The current implementation in LLVM does not check the
key relocation invariant, but this is ongoing work on developing such
a verifier.  Please ask on llvmdev if you're interested in
experimenting with the current version.

Bugs and Enhancements
=====================

Currently known bugs and enhancements under consideration can be
tracked by performing a `bugzilla search
<http://llvm.org/bugs/buglist.cgi?cmdtype=runnamed&namedcmd=Statepoint%20Bugs&list_id=64342>`_
for [Statepoint] in the summary field. When filing new bugs, please
use this tag so that interested parties see the newly filed bug.  As
with most LLVM features, design discussions take place on `llvmdev
<http://lists.cs.uiuc.edu/mailman/listinfo/llvmdev>`_, and patches
should be sent to `llvm-commits
<http://lists.cs.uiuc.edu/mailman/listinfo/llvm-commits>`_ for review.

