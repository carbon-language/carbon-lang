/* RUN: clang-cc %s -Eonly -std=c89 -pedantic -verify
*/
/* PR3937 */
#define zero() 0
#define one(x) 0
#define two(x, y) 0

zero()
zero(1);          /* expected-error {{too many arguments provided to function-like macro invocation}} */
zero(1, 2, 3);    /* expected-error {{too many arguments provided to function-like macro invocation}} */

one()   /* ok */
one(a)
one(a,)           /* expected-error {{too many arguments provided to function-like macro invocation}} */
one(a, b)         /* expected-error {{too many arguments provided to function-like macro invocation}} */

two()       /* expected-error {{too few arguments provided to function-like macro invocation}} */
two(a)      /* expected-error {{too few arguments provided to function-like macro invocation}} */
two(a,b)
two(a, )    /* expected-warning {{empty macro arguments were standardized in C99}} */
two(a,b,c)  /* expected-error {{too many arguments provided to function-like macro invocation}} */
two(
    ,     /* expected-warning {{empty macro arguments were standardized in C99}} */
    ,     /* expected-warning {{empty macro arguments were standardized in C99}}  \
             expected-error {{too many arguments provided to function-like macro invocation}} */
    )     
two(,)      /* expected-warning 2 {{empty macro arguments were standardized in C99}} */



/* PR4006 */
#define e(...) __VA_ARGS__  /* expected-warning {{variadic macros were introduced in C99}} */
e(x)
