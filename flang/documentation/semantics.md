The semantic pass will determine whether the input program is a legal Fortran program.

If the program is not legal, the results of the semantic pass will be a list of errors associated with the program.

If the program is legal, the semantic pass will produce an unambiguous parse tree with additional information that is useful for the tools API and creation of the DST.

What is required of semantics?
* Error checking
* A non-ambiguous parse tree
* Symbol tables with scope information
* Name & operator resolution

What do we want from semantics?
* Cache information about labels and references to labels
* Cache information derived from static expression evaluation

What don’t we want from semantics?
* Semantics will not display error messages directly.  Instead, error messages and their associated source locations will be saved and returned to the caller.
* The parse tree will not be modified except to resolve ambiguity and resolve names, operators, and labels.

Semantic checking does not need to preserve information that is easily recomputed, such as pointers to enclosing structures.

The parse tree shall be  immutable after resolution of names, operators, labels and ambiguous sub-trees.   This means that the parse tree does not have direct references error messages, etc.

Much of the work that is to be performed by semantic analysis has been specified in the Fortran standard with numbered constraints.  The structure of the code in the semantic analyzer should correspond to the structure of the Fortran standard as closely as possible so that one can refer to the Standard easily from the code, and so that we can audit the code for missing checks.

The code that generates LLVM will be able to be implemented with assertions rather than with user error message generation; in other words, semantic analysis will detect and report all errors. Note that informational and warning messages may be generated after semantic analysis.

Analyses and data structures that can be deferred to the deep structure should be so, with exceptions for cases where completing an analysis is just a little more complex than completing a correctness check (e.g. EQUIVALENCE overlays).


## Symbol resolution and scope assignment
The section describes the when scopes are created and how symbols are resolved.  It is a step-by-step process.  Each step is envisioned as a separate pass over the tree.  The sub-bullets under each step will happen roughly in the order specified.

There is a special predefined scope for intrinsics.  This scope is an ancestor of all other scopes.

The following steps will be followed each program unit:

_N.B. Modules are not yet covered_

_N.B. We need to define the semantics of the LOC intrinsic_

#### Step 1. Process the top-level declaration, e.g. a subroutine
1. Create a new scope
1. Add the name of the program unit to the scope
  - Except for functions without a result clause
1. Add the result variable to the scope
1. Add the names of the dummy arguments to the scope

Implementation note:  When a program make an illegal forward reference, we should emit at least a warning so that programs that are illegally assuming host association for a name won’t be silently invalidated; preferably with a message that references both instances.

#### Step 2.  Process the specification part
1. Setup implicit rules
1. Process imports, uses, and host association
  - Host association logically happens here; can be deferred until referenced?
1. Add the names of the internal and module procedures
1. Process declaration constructs in a single pass
  - We think we can process declaration constructs in a single pass because:
    - Is it not legal to reference an internal procedure.
    - Is it not  legal to reference not-yet-defined parameters, constants, etc.
    - Is it not possible to inquire about a type parameter or array bound for an object that is not yet defined
    - So, no other forward definitions, so yes, we can do in a single pass
1. Do we ever need to apply implicit rules in the specification section?  TBD:
  - `integer(kind = kind(x)) :: y ! does implicit rule apply to ‘x’`?
  - `integer, parameter :: z = rank(x) ! use implicit rule to get ‘0’`?
  - What if (i) and (ii) are legal & x’s type is subsequently declared?
1. Apply implicit rules to undefined locals, dummy arguments and the function result
1. Create new scopes for derived type, structure, union

At this point, All names in the specification part of the parse tree reference a symbol

#### Step 3. Resolve statement functions vs array assignments
1. Rewrite and move array assignments to execution part
1. Why rewrite?  Because array assignment needs processing in Step 4
1. Statement functions need scopes for the dummy arguments

N.B. As soon as a statement function definition is determined to actually be a misrecognized assignment to an array element, all of the statement definitions that follow it in the same specification-part must also be converted into array element assignments, even if that would lead to an error.

#### Step 4. Resolve symbols in the execution part
1. Lookup the name
  - If it exists in a scope, update the name to reference the symbol
  - If it does not exist,
    * Apply the implicit rules
    * Add the name to the scope
    * Update the name to reference the new symbol
  - Introduce new scopes for
    * Select Type type guard statements
    * Select Rank case statements
    * Associate construct
    * Block construct
      - Block has a specification part
      - Blocks start Step 1..4 again
      - N.B. Implicits are applied to the host scope
    * Implied Do
    * Index names in Forall and Do Concurrent
    * Change Team
    * OpenMP and OpenACC constructs
    * ENTRY

References to derived types members are not resolved until semantics

No semantic checking or resolving of types (except for implicit declarations) has happened yet.

#### Step 5. Perform Step 1..4 on each internal procedure
1. Side effect is that they get a proper interface in the parent scope
1. Why now?  Return types for functions, e.g
  - `a = f(a, b, c) % x + 1`
  - Need to know the return type and types of arguments

#### Step 6. Tree Disambiguation

At this point, or during Step 3 (TBD), the tree can be rewritten to be unambiguous.
1. Structure vs operator a.b.c.d
1. Array references vs function calls
1. Statement functions vs array assignment (In Step 3)
1. READ/WRITE stmts where the arguments do not have keywords
  - WRITE (6, X)  ….
  - That X might be a namelist group or an internal variable
  - Need to know the names of the namelist groups to disambiguate it
1. Others….? TBD

Resolution of parse tree ambiguity (statement function definition, function vs. array)

#### Step 7. Do enough semantic processing to generate .mod files
1. Fully resolve derived types
1. Combine and check declarations of all entities within a given scope; resolve their type, rank, shape, and other attributes.
1. Constant evaluation is required at this point.

Why do Step 7 before the rest of semantic checking? The sooner we can generate mod file the sooner we can read ‘em; you can test a lot of Fortran programs as soon as you can read mod files.

#### Step 8. Semantic Rule Checking

An incomplete and unordered list of requirements for semantic analysis:

* EQUIVALENCE overlaying (checking at least)
* Intrinsic function generic->specific resolution, constraint checking, T/R/S.
* Compile-time evaluation of constant expressions, including intrinsic functions.
* Resolution of generics and type-bound procedures.
* Identifying and recording uplevel references.
* Control flow constraint checking
* Labeled DO loop terminal statement expansion? (maybe not, can defer to CFG in DST).
* Construct association: distinguish pointer-like from allocatable-like
* OMP and OACC checking
* CUF constraint checking

## Utility Routines

### Diagnostic Output
TBD

### Constant Expression Evaluation
1. Scalars
1. Array intrinsics
