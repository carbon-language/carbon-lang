Extensions, deletions, and legacy features supported
====================================================

* Tabs in source
* `<>` as synonym for `.NE.` and `/=`
* `$` and `@` as legal characters in names
* `.T.` and `.F.`
* Initialization in type declaration statements using `/values/`
* Kind specification with `*`, e.g. `REAL*4`
* `DOUBLE COMPLEX`
* Signed complex literal constants
* `.XOR.` as predefined operator (can be overridden)
* `.N.`, `.A.`, `.O.`, `.X.` predefined operator synonyms
* `STRUCTURE`, `RECORD`, `UNION`, and `MAP`
* Structure field access with `.field`
* `NCHARACTER` type and `NC` Kanji character literals
* `BYTE` as synonym for `INTEGER(KIND=1)`
* Quad precision REAL literals with `Q`
* `X` prefix/suffix as synonym for `Z` on hexadecimal literals
* `B`, `O`, `Z`, and `X` accepted as suffixes as well as prefixes
* Triplets allowed in array constructors
* Old-style `PARAMETER pi=3.14` statement (no parentheses)
* `%LOC`, `%VAL`, and `%REF`
* Leading comma allowed before I/O item list
* Empty parentheses allowed in `PROGRAM P()`
* Missing parentheses allowed in `FUNCTION F`
* Cray based `POINTER(p,x)`
* Arithmetic `IF`.  (Which branch with NaN take?)
* `ASSIGN` statement, assigned `GO TO`, and assigned format
* `PAUSE` statement
* Hollerith literals and edit descriptors
* `NAMELIST` allowed in the execution part
* Omitted colons on type declaration statements with attributes
* COMPLEX constructor expression, e.g. `(x+y,z)`
* `+` and `-` before all primary expressions, e.g. `x*-y`
* `.NOT. .NOT.` accepted
* `NAME=` as synonym for `FILE=`
* `DISPOSE=`
* Data edit descriptors without width or other details
* Backslash escape character sequences in quoted character literals
* `D` lines in fixed form as comments or debug code

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
* `Z` prefix on unquoted hexadecimal constants
