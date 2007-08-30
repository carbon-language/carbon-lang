/* RUN: not clang %s -std=c89 -pedantic-errors
 */

/* We can't put expected-warning lines on #if lines. */

#if 1LL				/* expected-warning {{long long}} */
#endif
