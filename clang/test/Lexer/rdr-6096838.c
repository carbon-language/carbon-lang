/* RUN: clang -fsyntax-only -verify %s &&
 * RUN: clang -std=gnu89 -fsyntax-only -verify %s &&
 * RUN: clang -DPEDANTIC -pedantic -std=gnu89 -fsyntax-only -verify %s
 */

#ifdef PEDANTIC

long double d = 0x0.0000003ffffffff00000p-16357L; /* expected-warning {{ hexadecimal floating constants are a C99 feature }} */

#else

long double d = 0x0.0000003ffffffff00000p-16357L;

#endif
