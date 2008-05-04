/* RUN: clang -E %s -DNO_ERRORS &&
   RUN: not clang -E %s
 */

#ifdef NO_ERRORS
/* None of these divisions by zero are in live parts of the expression, do not
   emit any diagnostics. */

#define MACRO_0 0
#define MACRO_1 1

#if MACRO_0 && 10 / MACRO_0
foo
#endif

#if MACRO_1 || 10 / MACRO_0
bar
#endif

#if 0 ? 124/0 : 42
#endif

// PR2279
#if 0 ? 1/0: 2
#else
#error
#endif


#else


/* The 1/0 is live, it should error out. */
#if 0 && 1 ? 4 : 1 / 0
baz
#endif


#endif
