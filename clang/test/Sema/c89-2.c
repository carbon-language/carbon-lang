/* RUN: %clang_cc1 %s -std=c89 -pedantic-errors -Wno-empty-translation-unit -verify
 */

#if 1LL        /* expected-error {{long long}} */
#endif
