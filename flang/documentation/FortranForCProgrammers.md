<!--
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
-->

Fortran For C Programmers
=========================

This note is limited to essential information about Fortran so that
a C or C++ programmer can get started quickly with the language,
at least as a reader, and avoid some common pitfalls when starting
to write or modify Fortran code.
Please see other sources to learn about Fortran's rich history,
current applications, and modern best practices in new code.

Know This At Least
------------------
* There have been many implementations of Fortran, often from competing
  vendors, and the standard language has been defined by U.S. and
  international standards organizations.  The various editions of
  the standard are known as the '66, '77, '90, '95, 2003, 2008, and
  (real soon now) 2018 standards.
* Forward compatibility is important.  Fortran has outlasted many
  generations of computer systems hardware and software.  Standard
  compliance notwithstanding, Fortran programmers generally expect that
  code that has compiled successfully in the past will continue to
  compile and work indefinitely.  The standards sometimes designate
  features as being deprecated, obsolescent, or even deleted, but that
  can be read only as discouraging their use in new code -- they'll
  probably always work in any serious implementation.
* Fortran has two source forms, which are typically distinguished by
  filename suffixes.  `foo.f` is old-style "fixed-form" source, and
  `foo.f90` is new-style "free-form" source.  All language features
  are available in both source forms.  Neither form has reserved words
  in the sense that C does.  Spaces are not required between tokens
  in fixed form, and case is not significant in either form.
* Variable declarations are optional by default.  Variables whose
  names begin with the letters `I` through `N` are implicitly
  `INTEGER`, and others are implicitly `REAL`.  These implicit typing
  rules can be changed in the source.
* Fortran uses parentheses for both array references and for function calls.
  All arrays must be declared as such; other names followed by parenthesized
  expressions are assumed to be function calls.
* Fortran has a _lot_ of built-in "intrinsic" functions.  They are always
  available without a need to declare or import them.  Their names reflect
  the implicit typing rules, so you will encounter names that have been
  modified so that they have the right type (e.g., `AIMAG` has a leading `A`
  so that it's `REAL` rather than `INTEGER`).
* The modern language has means for declaring types, data, and subprogram
  interfaces in compiled "modules", as well as legacy mechanisms for
  sharing data and interconnecting subprograms.

Data Types
----------
There are five built-in ("intrinsic") types: `INTEGER`, `REAL`, `COMPLEX`,
`LOGICAL`, and `CHARACTER`.
They are parameterized with "kind" values, which should be treated as
non-portable integer codes but in practice today are the byte sizes of
the data.
The legacy `DOUBLE PRECISION` intrinsic type is an alias for a kind of `REAL`
that should be bigger than the default `REAL`.

`COMPLEX` is a simple structure that comprises two `REAL` components.

`CHARACTER` data also have length, which may or may not be known at compilation
time.
`CHARACTER` variables are fixed-length strings and they get padded out
with space characters when not completely assigned.

User-defined ("derived") data types can be synthesized from the intrinsic
types and from previously-defined user types, much like a C `struct`.
Derived types can be parameterized with integer values that either have
to be constant at compilation time ("kind" parameters) or deferred to
execution ("len" parameters).

Derived types can inherit ("extend") from one other derived type, no more.
They can have user-defined destructors (`FINAL` procedures).
They can specify default initial values for their components.
With some work, one can also specify a general constructor function,
since Fortran allows a generic interface to have the same name as that
of a derived type.

Last, there are "typeless" binary constants that can be used in a few
situations, like static data initialization or immediate conversion,
where type is not necessary.

Arrays
------
Arrays are not a type in Fortran.
Being an array is a property of an object, not of a type.
Unlike C, one cannot have an array of arrays or an array of pointers,
although can can have an array of a derived type that has arrays or
pointers as components.
Arrays are multidimensional, and the number of dimensions is called
the _rank_ of the array.
In storage, arrays are stored such that the last subscript has the
largest stride in memory, e.g. A(1,1) is followed by A(2,1), not A(1,2).
And yes, the default lower bound on each dimension is 1, not 0.

Expressions can manipulate arrays as multidimensional values, and
the compiler will create the necessary loops.

Allocatables
------------
Modern Fortran programs use `ALLOCATABLE` data extensively.
Such variables and derived type components are allocated dynamically.
They are automatically deallocated when they go out of scope, much
like C++'s `std::vector<>` class template instances are.
The array bounds, derived type `LEN` parameters, and even the
type of an allocatable can be deferred to run time.

I/O
---
Fortran's input/output features are built into the syntax of the language,
not defined by library interfaces as in C and C++.
There are means for raw binary I/O and for "formatted" transfers to
character representations.
There are means for random-access I/O using fixed-size records as well as for
sequential I/O.
One can scan data from or format data into `CHARACTER` variables via
"internal" formatted I/O.
I/O from and to files uses a scheme of integer "unit" numbers that is
similar to the open file descriptors of UNIX; i.e., one opens a file
and assigns it a unit number, then uses that unit number in subsequent
`READ` and `WRITE` statements.

Formatted I/O relies on format specifications to map values to fields of
characters, similar to the format strings used with C's `printf` family
of standard library functions.
These format specifications can appear in `FORMAT` statements and
be referenced by their labels, in character literals directly in I/O
statements, or in character variables.

One can also use compiler-generated formatting in "list-directed" I/O,
in which the compiler derives reasonable default formats based on
data types.

Subprograms
-----------
Fortran has both `FUNCTION` and `SUBROUTINE` subprograms.
They share the same name space.
Subroutines are called with the `CALL` statement, while functions are
invoked with function references in expressions.

There is one level of subprogram nesting.
A function, subroutine, or main program can have functions and subroutines
nested within it, but these "internal" procedures cannot themselves have
their own internal procedures.
As is the case with C++ lambda expressions, internal procedures can
reference names from their host subprograms.

Arguments
---------
Functions and subroutines have "dummy" arguments that are dynamically
associated with actual arguments during calls.
Essentially, all argument passing in Fortran is by reference, not value.
One may restrict access to argument data by declaring that dummy
arguments have `INTENT(IN)`, but that corresponds to the use of
a `const` reference in C++ and does not imply that the data are
copied; use `VALUE` for that.

When it is not possible to pass a reference to an object, or a sparse
regular array section of an object, as an actual argument, Fortran
compilers must allocate temporary space to hold the actual argument
across the call.
This is always guaranteed to happen when an actual argument is enclosed
in parentheses.

The compiler is free to assume that any aliasing between dummy arguments
and other data is safe.
In other words, if some object can be written to under one name, it's
never going to be read or written using some other name in that same
scope.
```
  SUBROUTINE FOO(X,Y,Z)
  X = 3.14159
  Y = 2.1828
  Z = 2 * X ! CAN BE FOLDED AT COMPILE TIME
  END
```
This is the opposite of the assumptions under which a C or C++ compiler must
labor when trying to optimize code with pointers.

Polymorphism
------------
Fortran code can be written to accept data of some derived type or
any extension thereof using `CLASS`, deferring the actual type to
execution, rather than the usual `TYPE` syntax.
This is somewhat similar to the use of `virtual` functions in c++.

Fortran's `SELECT TYPE` construct is used to distinguish between
possible specific types dynamically.

Pointers
--------
Pointers are objects in Fortran, not a data type.
Pointers can point to data, arrays, and subprograms, but not to other pointers
or to an allocatable.
A pointer can only point to data that has the `TARGET` attribute.
Outside of the pointer assignment statement (`P=>X`) and some intrinsic
functions and cases with pointer dummy arguments, pointers are implicitly
dereferenced, and the use of their name is a reference to the data to which
they point instead.

Unlike allocatables, pointers do not deallocate their data when they go
out of scope.

A legacy feature, "Cray pointers", implements dynamic base addressing of
one variable using an address stored in another.

Preprocessing
-------------
There is no standard preprocessing feature, but every real Fortran implementation
has some support for passing Fortran source code through a variant of
the standard C source preprocessor.
Behavior varies across implementations and one should not depend on
much portability.
Preprocessing is typically requested by the use of a capitalized filename
suffix (e.g., "foo.F90") or a compiler command line option.

Pitfalls
--------
Variable initializers, e.g. `INTEGER :: J=123`, are _static_ initializers!
They imply that the variable is stored in static storage, not on the stack,
and the initialized value lasts only until the variable is assigned.
One must use an assignment statement to implement a dynamic initializer
that will apply to every fresh instance of the variable.
(Derived type component initializers, however, do work as expected.)

If you see an assignment to an array that's never been declared as such,
it's probably a definition of a "statement function", which is like
a parameterized macro definition, e.g. "A(X)=SQRT(X)**3".
In the original Fortran language, this was the only means for user
function definitions.
Today, of course, one should use an external or internal function instead.

Fortran expressions don't bind exactly like C's do.
Watch out for exponentiation with `**`, which of course C lacks; it
binds more tightly than negation does (e.g., `-2**2` is -4),
and it binds to the right, unlike any other Fortran or C operator
(e.g., `2**2**3` is 256, not 64).
Also dangerous are logical expressions, in which the unary negation
operator `.NOT.` binds less tightly than the binary `.AND.` and `.OR.`
operators do.

A Fortran compiler is allowed to short-circuit expression evaluation,
but not required to do so.
If one needs to protect a use of an `OPTIONAL` argument or possibly
disassociated pointer, use an `IF` statement, not a logical `.AND.`
operation.
In fact, Fortran can remove function calls from expressions if their
values are not required to determine the value of the expression's
result; e.g., if there is a `PRINT` statement in function `F`, it
may or may not be executed by the assignment statement `X=0*F()`.

