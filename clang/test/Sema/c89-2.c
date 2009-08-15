/* RUN: clang-cc %s -std=c89 -pedantic-errors -verify
 */

#if 1LL				/* expected-error {{long long}} */
#endif
