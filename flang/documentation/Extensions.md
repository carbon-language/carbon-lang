<!--
Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
-->

As a general principle, this compiler will accept by default and
without complaint many legacy features, extensions to the standard
language, and features that have been deleted from the standard,
so long as the recognition of those features would not cause a
standard-conforming program to be rejected or misinterpreted.

Other non-standard features, which do conflict with the current
standard specification of the Fortran programming language, are
accepted if enabled by command-line options.

Intentional violations of the standard
======================================
* The default `INTEGER` type is required by the standard to occupy
  the same amount of storage as the default `REAL` type.  Default
  `REAL` is of course 32-bit IEEE-754 floating-point today.  This legacy
  rule imposes an artificially small constraint in some cases
  where Fortran mandates that something have the default `INTEGER`
  type: specifically, the results of references to the intrinsic functions
  `LEN`, `SIZE`, `LBOUND`, `UBOUND`, and `SHAPE`.  We return
  `INTEGER(KIND=8)` in these cases.

Extensions, deletions, and legacy features supported by default
===============================================================
* Tabs in source
* `<>` as synonym for `.NE.` and `/=`
* `$` and `@` as legal characters in names
* Initialization in type declaration statements using `/values/`
* Kind specification with `*`, e.g. `REAL*4`
* `DOUBLE COMPLEX`
* Signed complex literal constants
* DEC `STRUCTURE`, `RECORD`, `UNION`, and `MAP`
* Structure field access with `.field`
* `BYTE` as synonym for `INTEGER(KIND=1)`
* Quad precision REAL literals with `Q`
* `X` prefix/suffix as synonym for `Z` on hexadecimal literals
* `B`, `O`, `Z`, and `X` accepted as suffixes as well as prefixes
* Triplets allowed in array constructors
* Old-style `PARAMETER pi=3.14` statement without parentheses
* `%LOC`, `%VAL`, and `%REF`
* Leading comma allowed before I/O item list
* Empty parentheses allowed in `PROGRAM P()`
* Missing parentheses allowed in `FUNCTION F`
* Cray based `POINTER(p,x)` and `LOC()` intrinsic (with `%LOC()` as
  an alias)
* Arithmetic `IF`.  (Which branch should NaN take? Fall through?)
* `ASSIGN` statement, assigned `GO TO`, and assigned format
* `PAUSE` statement
* Hollerith literals and edit descriptors
* `NAMELIST` allowed in the execution part
* Omitted colons on type declaration statements with attributes
* COMPLEX constructor expression, e.g. `(x+y,z)`
* `+` and `-` before all primary expressions, e.g. `x*-y`
* `.NOT. .NOT.` accepted
* `NAME=` as synonym for `FILE=`
* Data edit descriptors without width or other details
* `D` lines in fixed form as comments or debug code
* `CONVERT=` on the OPEN and INQUIRE statements
* `DISPOSE=` on the OPEN and INQUIRE statements
* Leading semicolons are ignored before any statement that
  could have a label
* The character `&` in column 1 in fixed form source is a variant form
  of continuation line.
* Character literals as elements of an array constructor without an explicit
  type specifier need not have the same length; the longest literal determines
  the length parameter of the implicit type, not the first.
* Outside a character literal, a comment after a continuation marker (&)
  need not begin with a comment marker (!).
* Classic C-style /*comments*/ are skipped, so multi-language header
  files are easier to write and use.
* $ and \ edit descriptors are supported in FORMAT to suppress newline
  output on user prompts.
* REAL and DOUBLE PRECISION variable and bounds in DO loops
* Integer literals without explicit kind specifiers that are out of range
  for the default kind of INTEGER are assumed to have the least larger kind
  that can hold them, if one exists.
* BOZ literals can be used as INTEGER values in contexts where the type is
  unambiguous: the right hand sides of assigments and initializations
  of INTEGER entities, and as actual arguments to a few intrinsic functions
  (ACHAR, BTEST, CHAR).  But they cannot be used if the type would not
  be known (e.g., `IAND(X'1',X'2')`).
* BOZ literals can also be used as REAL values in some contexts where the
  type is unambiguous, such as initializations of REAL parameters.
* EQUIVALENCE of numeric and character sequences (a ubiquitous extension)
* Values for whole anonymous parent components in structure constructors
  (e.g., `EXTENDEDTYPE(PARENTTYPE(1,2,3))` rather than `EXTENDEDTYPE(1,2,3)`
   or `EXTENDEDTYPE(PARENTTYPE=PARENTTYPE(1,2,3))`).
* Some intrinsic functions are specified in the standard as requiring the
  same type and kind for their arguments (viz., ATAN with two arguments,
  ATAN2, DIM, HYPOT, MAX, MIN, MOD, and MODULO);
  we allow distinct types to be used, promoting
  the arguments as if they were operands to an intrinsic `+` operator,
  and defining the result type accordingly.
* DOUBLE COMPLEX intrinsics DCONJG and DIMAG.

Extensions supported when enabled by options
--------------------------------------------
* C-style backslash escape sequences in quoted CHARACTER literals
  (but not Hollerith) [-fbackslash]
* Logical abbreviations `.T.`, `.F.`, `.N.`, `.A.`, `.O.`, and `.X.`
  [-flogical-abbreviations]
* `.XOR.` as a synonym for `.NEQV.` [-fxor-operator]

Extensions and legacy features deliberately not supported
---------------------------------------------------------
* `.LG.` as synonym for `.NE.`
* `REDIMENSION`
* Allocatable `COMMON`
* Expressions in formats
* `ACCEPT` as synonym for `READ *`
* `TYPE` as synonym for `PRINT`
* `ARRAY` as synonym for `DIMENSION`
* `VIRTUAL` as synonym for `DIMENSION`
* `ENCODE` and `DECODE` as synonyms for internal I/O
* `IMPLICIT AUTOMATIC`, `IMPLICIT STATIC`
* Default exponent of zero, e.g. `3.14159E`
* Characters in defined operators that are neither letters nor digits
* `B` suffix on unquoted octal constants
* `Z` prefix on unquoted hexadecimal constants (dangerous)
* `T` and `F` as abbreviations for `.TRUE.` and `.FALSE.` in DATA (PGI/XLF)
* Use of host FORMAT labels in internal subprograms (PGI-only feature)
* ALLOCATE(TYPE(derived)::...) as variant of correct ALLOCATE(derived::...) (PGI only)
* Defining an explicit interface for a subprogram within itself (PGI only)
* USE association of a procedure interface within that same procedure's definition
* NULL() as a structure constructor expression for an ALLOCATABLE component (PGI).
* Conversion of LOGICAL to INTEGER.
* IF (integer expression) THEN ... END IF  (PGI/Intel)
* Comparsion of LOGICAL with ==/.EQ. rather than .EQV. (also .NEQV.) (PGI/Intel)
* Procedure pointers in COMMON blocks (PGI/Intel)
* Underindexing multi-dimensional arrays (e.g., A(1) rather than A(1,1)) (PGI only)
* Legacy PGI `NCHARACTER` type and `NC` Kanji character literals
* Using non-integer expressions for array bounds (e.g., REAL A(3.14159)) (PGI/Intel)
* Mixing INTEGER types as operands to bit intrinsics (e.g., IAND); only two
  compilers support it, and they disagree on sign extension.
* Module & program names that conflict with an object inside the unit (PGI only).
* When the same name is brought into scope via USE association from
  multiple modules, the name must refer to a generic interface; PGI
  allows a name to be a procedure from one module and a generic interface
  from another.
* Type parameter declarations must come first in a derived type definition;
  some compilers allow them to follow `PRIVATE`, or be intermixed with the
  component declarations.

Preprocessing behavior
======================
* The preprocessor is always run, whatever the filename extension may be.
* We respect Fortran comments in macro actual arguments (like GNU, Intel, NAG;
  unlike PGI and XLF) on the principle that macro calls should be treated
  like function references.  Fortran's line continuation methods also work.
