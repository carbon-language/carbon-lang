<!--
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
-->

## Procedure reference implementation protocol

Fortran function and subroutine references are complicated.
This document attempts to collect the requirements imposed by the 2018
standard (and legacy extensions) on programs and implementations, work
through the implications of the various features, and propose both a
runtime model and compiler design.

All section, requirement, and constraint numbers herein pertain to
the Fortran 2018 standard.

This note does not consider calls to intrinsic procedures, statement
functions, or calls to internal runtime support library routines.

## Interfaces

Referenced procedures may or may not have declared interfaces
available to their call sites.

Calls to procedures with some post-'77 features require an explicit interface
 (15.4.2.2):
* keyword arguments
* procedures that are `ELEMENTAL` or `BIND(C)`
* procedures that are required to be `PURE` due to the context of the call
  (specification expression, `DO CONCURRENT`, `FORALL`)
* dummy arguments with these attributes: `ALLOCATABLE`, `POINTER`,
  `VALUE`, `TARGET`, `OPTIONAL`, `ASYNCHRONOUS`, `VOLATILE`
* dummy arguments that are coarrays, have assumed-shape/-rank,
  have parameterized derived types, &/or are polymorphic
* function results that are arrays, `ALLOCATABLE`, `POINTER`,
  or have derived types with any length parameters that are
  neither constant nor assumed

Module procedures, internal procedures, procedure pointers,
type-bound procedures, and recursive references by a procedure to itself
always have explicit interfaces.

Other uses of procedures besides calls may also require explicit interfaces,
such as procedure pointer assignment, type-bound procedure bindings, &c.

### Implicit interfaces

In the absence of any characteristic or context that requires an
explicit interface (see above), a top-level function or subroutine
can be called via its implicit interface.
Each of the arguments can be passed as a simple address, including
dummy procedures.
Procedures that *can* be called via an implicit interface can
enjoy more thorough checking
by semantics when they do have a visible external interface, but must be
compiled as if all calls to them were through the implicit interface.

Procedures that can be called via an implicit interface should respect
the naming conventions and ABI, if any, used for Fortran '77 programs
on the target architecture, so that portable libraries can be compiled
and used by distinct implementations (and versions of implementations)
of Fortran.

Note that functions with implicit interfaces still have known result
types, possibly by means of implicit typing of their names.
They can also be `CHARACTER(*)` assumed-length character functions.

In other words: procedures that can be referenced with implicit interfaces
have argument lists that comprise only addresses of actual arguments,
the length of an assumed-length `CHARACTER(*)` result, and links to
of host variable blocks for dummy procedures (see below), and they can
return only scalar values of intrinsic types.
None of their arguments or results are implemented with descriptors.

## Protocol overview

Here is a summary script of all of the actions that may need to be taken
by the calling procedure and its referenced procedure to effect
the call, entry, exit, and return steps of the procedure reference
protocol.
The order of these steps is not particularly strict, and we have
some design alternatives that are explored further below.

### Before the call:
1. Compute &/or copy into temporary storage the values of
   some actual argument expressions and designators (see below).
1. Finalize &/or re-initialize `INTENT(OUT)` non-pointer non-allocatable
   actual arguments (see below).
1. Create and populate descriptors for assumed-shape/-rank arrays,
   parameterized derived types with length, polymorphic types,
   & coarrays.
1. Possibly allocating function result storage,
   if its size can be known by the caller; function results that are
   neither `POINTER` nor `ALLOCATABLE` must have explicit shapes (C816).
1. Create and populate a descriptor for the function result, if it
   needs one (deferred-shape/-length `POINTER`, any `ALLOCATABLE`,
   parameterized derived type with non-constant length parameters, &c.).
1. Capture the values of host-escaping local objects in memory;
   package them into single address (for calls to internal procedures &
   for calls that pass internal procedures as arguments).
1. Resolve the target procedure's polymorphic binding, if any.
1. Marshal actual argument addresses/values into registers.
1. Marshal extra arguments for assumed-length `CHARACTER` result length,
   function result descriptor, target host variable link, &/or dummy
   procedure host variable links
1. Jump.

### On entry:
1. Shuffle `ENTRY` dummy arguments & jump to common entry point.
1. Complete `VALUE` copying and `INTENT(OUT)` finalization and
   reinitialization if these steps are not always done by the caller
   (as I think they will be).
1. Optionally compact assumed-shape arguments for contiguity to enable
   better SIMD vectorization, if not `TARGET` and not already contiguous.
1. Complete allocation of function result storage, if that has
   not been done by the caller.
1. Initialize components of derived type local variables,
   including the function result.

Execute the callee, populating the function result or selecting
the subroutine's alternate return.

### On exit:
1. Clean up local scope (finalization, deallocation)
1. Deallocate `VALUE` argument temporaries.
   (But don't finalize them; see 7.5.6.3(3)).
1. Replace any assumed-shape argument data that were compacted on
   entry for contiguity for SIMD vectorization, if possibly modified
   (and never when `INTENT(IN)` or `VALUE`).
1. Identify alternate `RETURN` to caller.
1. Marshal results.
1. Jump

### On return to the caller:
1. Copy actual argument array designator data that was copied into
   a temporary back into its original storage (see below).
1. Complete deallocation of actual argument temporaries (not `VALUE`).
1. Reload host-escaping local objects from memory.
1. `GO TO` alternate return, if any.
1. Use the function result in an expression.
1. Eventually, finalize &/or deallocate the function result.

## The messy details

### Copying actual argument values into temporary storage

There are several conditions that require the compiler to generate
code that allocates and populates temporary storage for an actual
argument.

First, actual arguments that are expressions, not designators, obviously
need to be computed and captured into memory in order to be passed
by reference.
This includes parenthesized designators like `(X)` as an important
special case, which are expressions in Fortran.
This case should also include constants.
The dummy argument cannot have `INTENT(OUT)` or `INTENT(IN OUT)`.

Small scalar or elemental `VALUE` arguments may be passed in registers.
Multiple elemental `VALUE` arguments might be packed into SIMD registers.

Actual arguments that are designators, not expressions, must
be copied into temporaries in many situations.

1. Coindexed objects need to be copied into the local image.
   (This can get very involved if they contain `ALLOCATABLE`
   components, which also need to be copied, along with their
   `ALLOCATABLE` components, and may be best implemented with a runtime
   library routine working off a description of the type.)
1. Actual arguments associated with dummies with the `VALUE`
   attribute need to copied; this could be done on either
   side of the call, but there are optimization opportunities
   on the caller's side.
1. In non-elemental calls, the values of array sections with
   vector-valued subscripts need to be compacted into temporaries.
   These actual arguments are not definable, and they are not allowed to
   be associated with non-`VALUE` dummy arguments with the attributes
   `INTENT(IN)`, `INTENT(IN OUT)`, `ASYNCHRONOUS`, or `VOLATILE`
   (15.4.2.4(21)).
1. Non-contiguous arrays being passed to dummy arguments that
   must be contiguous due to a `CONTIGUOUS` attribute or not
   being assumed-shape/-rank.
   This should be a runtime decision, so that actual arguments
   that turn out to be contiguous can be passed cheaply.

Actual arguments associated with `INTENT(OUT)` dummies that require
allocation of a temporary don't have to populate it, but they
do have to initialize the storage when a derived type has
component initializations.

Except for `VALUE` and `INTENT(IN)` dummy arguments, the original
contents of local designators that have been compacted into temporaries
could optionally have their `ALLOCATABLE` components invalidated
across the call as an aid to debugging.

Except for `VALUE` and `INTENT(IN)` dummy arguments, the contents of
the temporary storage will be copied back into the actual argument
designator after control returns from the procedure, and it may be necessary
to preserve addresses (or the values of subscripts and cosubscripts
needed to recalculate them) of the actual argument designator, or its
elements, in additional temporary storage if they can't be safely or
quickly recomputed after the call.

### `INTENT(OUT)` preparation
Actual arguments that are associated with `INTENT(OUT)`
dummy arguments are required to be definable.

Such arguments are finalized (as if) on entry to the called
procedure.  In particular, in calls to elemental procedures,
the elements of an array are finalized by a scalar or elemental
`FINAL` procedure (7.5.6.3(7)).

Derived type components with initializers are (re)initialized.

The preparation of actual arguments for `INTENT(OUT)` could be
done on either side of the call.  If the preparation is
done by the caller, there is an optimization opportunity
in situations where unmodified incoming `INTENT(OUT)` dummy
arguments are being passed onward as outgoing `INTENT(OUT)`
arguments.

### Copying temporary storage back into actual argument designators

Except for `VALUE` and `INTENT(IN)` dummy arguments and array sections
with vector-valued subscripts (15.5.2.4(21)), temporary storage into
which actual argument data were compacted for contiguity before the call
must be redistributed back to its original storage by the caller after
the return.

In conjunction with saved cosubscript values, a standard descriptor
suffices to represent a pointer to the original storage into which the
temporary data should be redistributed.

Note that coindexed objects with `ALLOCATABLE` ultimate components
are required to be associated only with dummy arguments with the
`VALUE` &/or `INTENT(IN)` attributes (15.6.2.4(6)), so there is no
requirement that the local image somehow reallocate remote storage
when copying the data back.

### Host association linkage

Calls to dummy procedures and procedure pointers that resolve to
internal procedures need to pass an additional argument that
addresses on block of storage in the stack frame of the their
host subprogram that was active at the time they were passed as an
actual argument or associated with a procedure pointer.
This is similar to a static link in implementations of programming
languages with nested subprograms, although Fortran only allows
one level of nesting.

The host subprogram objects that are visible to any of their internal
subprograms need to be resident in memory across any calls to them
(direct or not).  Any host subprogram object that might be defined
during a call to an internal subprogram needs to be reloaded after
a call or reside permanently in memory.
A simple conservative analysis of the internal subprograms can
identify all of these escaping objects and their definable subset.

The address of the host subprogram storage used to hold the escaping
objects needs to be saved alongside the code address(es) that
represent a procedure pointer.
It also needs to be conveyed alongside the actual argument for a
dummy procedure.

For subprograms that can be called with an implicit interface,
we cannot use a "procedure pointer descriptor" to represent an
actual argument for a dummy procedure -- a Fortran '77 routine
with an `EXTERNAL` dummy argument expects to receive a single
address.  Instead, when passing an actual procedure on a call
to a procedure that can be called with an implicit interface,
we will need to use additional arguments to convey the host
storage link addresses.

## Further topics to document

### Target resolution
* polymorphic bindings
* procedure pointers
* dummy procedures
* generic resolution

### Arguments
* Alternate return specifiers
* `%VAL()` and `%REF()`
* Unrestricted specific intrinsic functions as actual arguments
* Check definability of `INTENT(OUT)` and `INTENT(IN OUT)` actuals.

### Naming
* Modules
* Submodules
* Subprograms
* SIMD vs. scalar versions of `ELEMENTAL` procedures
* Mangling explicit interfaces, possibly with versioning
* Unrestricted specific intrinsic functions (and perhaps SIMD variants)

### Other
* SIMD variants of `ELEMENTAL` procedures (& unrestricted specific intrinsics)
* Interoperable procedures
* Multiple code addresses for dummy procedures
* Elemental calls with array arguments
