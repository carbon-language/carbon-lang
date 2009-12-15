/* RUN: %clang_cc1 %s -std=c89 -pedantic-errors -verify
 */

#if 1LL        /* expected-error {{long long}} */
#endif
