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
* `NCHARACTER` type and `NC` Kanji character literals
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
* Cray based `POINTER(p,x)`
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
* REAL variable and bounds in DO loops

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
* `T` and `F` as abbreviations for `.TRUE.` and `.FALSE.` (dangerous)
* Use of host FORMAT labels in internal subprograms (PGI-only feature)
* ALLOCATE(TYPE(derived)::...) as variant of correct ALLOCATE(derived::...) (PGI only)
