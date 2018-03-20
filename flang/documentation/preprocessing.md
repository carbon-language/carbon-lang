Fortran Preprocessing
=====================

Behavior common to all compilers (or nearly so):

* Macro and argument names are case-sensitive
* Fixed form right margin clipping has precedence over macro
  name recognition, and over recognition of function-like parentheses
  and arguments
* Fixed form right margin clipping does not apply to directives
* Keyword-like macro names are not recognized if spaces are inserted
  into their invocations, even in fixed form
* Text is rescanned after expansion of macros and arguments
* Macros are not expanded within quoted character literals or
  quoted FORMAT edit descriptors
* Keyword-like macro expansion is before token pasting via fixed form
  space removal
* C-like line continuations with backslash-newline are allowed in
  definitions of macro bodies
* /* Old style C comments */ are removed from macro definitions
* // New style C comments are not removed, since Fortran has OPERATOR(//)
* Backslash-newline can appear in old-style C comments
* #define FALSE TRUE ...    .FALSE.  yields .TRUE.  (i.e., tokenization
  does not hide the names of operators or logical constants)
* #define KWM c   allows the use of column-1 KWM as a fixed form comment
  indicator
* A #define in continuation lines can't define a macro that's invoked
  earlier in the same continued statement

Inconsistent behavior that can be defined in obvious ways, despite some
compilers' behavior otherwise:

* Invoked macro names can straddle a Fortran line continuation
* ... unless implicit fixed form card padding intervenes
* Comment lines may appear with continuations in a split macro names
* Function-like macro invocations can straddle a Fortran fixed form line
  continuation between the name and the left parenthesis, and comment lines
  can be there too
* Function-like macro invocations can straddle a Fortran fixed form line
  continuation between the parentheses, and comment lines can be there
* Macros are not to be expanded within Hollerith constants or Hollerith
  FORMAT edit descriptors
* Token pasting with ## works in function-like macros
* Directives can be #CAPITALIZED in fixed form
* Fixed form clipping after column 72 is done before macro expansion,
  not after
* C-like backslash-newline can appear in the name of a keyword-like macro
  definition
* If # is in column 6 in fixed form, it's a continuation marker, not a
  directive
* #define KWM !   allows KWM to begin a comment

Judgement calls, where precedents are unclear:
* #if 2 .LT. 3 should work
* If a function-like macro does not close its parentheses, additional text
  is read until it does, even if proper Fortran line continuation does
  not appear
* ... However, the leading parenthesis has to be on the same line or a
  continuation thereof
* If macros expand to text containing &, it doesn't work as a free form
  continuation marker
* #define c 1   does not allow a column-1 c to be used as a label in fixed form
* #define KWM &   does not allow KWM to be use as a continuation indicator

Bad behavior that needs fixing:
* A macro invocation can straddle free form continuation lines in all of their
  forms, with continuation allowed in the name, before the arguments, and
  within the arguments
* Directives can be #CAPITALIZED in free form


In short, things should work as if...
1. Fixed form lines are clipped after column 72, and padded up to 72
2. Fortran comments are removed
3. Continuation lines are joined up
4. Directives are processed and macros expanded
5. Newly visible comments are removed
