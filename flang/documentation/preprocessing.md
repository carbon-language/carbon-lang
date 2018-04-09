Fortran Preprocessing
=====================

Behavior common to (nearly) all compilers:
------------------------------------------
* Macro and argument names are sensitive to case.
* Fixed form right margin clipping after column 72 (or 132)
  has precedence over macro name recognition, and also over
  recognition of function-like parentheses and arguments.
* Fixed form right margin clipping does not apply to directive lines.
* Macro names are not recognized as such when spaces are inserted
  into their invocations in fixed form.
* Text is rescanned after expansion of macros and arguments.
* Macros are not expanded within quoted character literals or
  quoted FORMAT edit descriptors.
* Macro expansion occurs before any effective token pasting via fixed form
  space removal.
* C-like line continuations with backslash-newline are allowed in
  directives, including the definitions of macro bodies.
* `/* Old style C comments */` are ignored in directives and
  removed from the bodies of macro definitions.
* `// New style C comments` are not removed, since Fortran has OPERATOR(//).
* C-like line continuations with backslash-newline can appear in
  old-style C comments in directives.
* After `#define FALSE TRUE`, `.FALSE.` is replaced by `.TRUE.`;
  i.e., tokenization does not hide the names of operators or logical constants.
* `#define KWM c` allows the use of `KWM` in column as a fixed form comment
  line indicator.
* A `#define` directive intermixed with continuation lines can't
  define a macro that's invoked earlier in the same continued statement.

Behavior that is not consistent to all extant compilers but which
probably should be uncontroversial:
-----------------------------------
* Invoked macro names can straddle a Fortran line continuation.
* ... unless implicit fixed form card padding intervenes; i.e.,
  in fixed form, a continued macro name has to be split at column
  72 (or 132).
* Comment lines may appear with continuations in a split macro names.
* Function-like macro invocations can straddle a Fortran fixed form line
  continuation between the name and the left parenthesis, and comment and
  directive lines can be there too.
* Function-like macro invocations can straddle a Fortran fixed form line
  continuation between the parentheses, and comment lines can be there too.
* Macros are not expanded within Hollerith constants or Hollerith
  FORMAT edit descriptors.
* Token pasting with `##` works in function-like macros.
* Argument stringization with `#` works in function-like macros.
* Directives can be capitalized (e.g., `#DEFINE`) in fixed form.
* Fixed form clipping after column 72 or 132 is done before macro expansion,
  not after.
* C-like line continuation with backslash-newline can appear in the name of
  a keyword-like macro definition.
* If `#` is in column 6 in fixed form, it's a continuation marker, not a
  directive indicator.
* `#define KWM !` allows KWM to signal a comment.

Judgement calls, where precedents are unclear:
----------------------------------------------
* Expressions in `#if` and `#elif` should support both Fortran and C
  operators; e.g., `#if 2 .LT. 3` should work.
* If a function-like macro does not close its parentheses, line
  continuation should be assumed.
* ... However, the leading parenthesis has to be on the same line as
  the name of the function-like macro, or on a continuation line thereof.
* If macros expand to text containing `&`, it doesn't work as a free form
  line continuation marker.
* `#define c 1` does not allow a `c` in column 1 to be used as a label
  in fixed form, rather than as a comment line indicator.
* IBM claims to be ISO C compliant and therefore recognizes trigraph sequences.

Behavior that few compilers properly support (or none), but should:
-------------------------------------------------------------------
* A macro invocation can straddle free form continuation lines in all of their
  forms, with continuation allowed in the name, before the arguments, and
  within the arguments.
* Directives can be capitalized in free form, too.
* `__VA_ARGS__` and `__VA_OPT__` work in variadic function-like macros.

In short, a Fortran preprocessor should work as if:
---------------------------------------------------
1. Fixed form lines are padded up to column 72 (or 132) and clipped thereafter.
2. Fortran comments are removed.
3. Fortran line continuations are processed (outside directives).
4. C-style line continuations are processed in directives.
5. C old-style comments are removed from directives.
6. Directives are processed and macros expanded.
   Along the way, Fortran `INCLUDE` lines and preprocessor `#include` directives
   are expanded, and all these steps applied recursively to the introduced text.
7. Newly visible Fortran comments are removed.

Last, if the preprocessor is not integrated into the Fortran compiler,
new Fortran continuation line markers should be introduced into the final
text.

OpenMP-style directives that look like comments are not addressed by
this scheme but are obvious extensions.
