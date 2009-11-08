/* RUN: clang-cc -fsyntax-only -verify %s
 * RUN: clang-cc -std=gnu89 -fsyntax-only -verify %s
 rdar://6096838
 */

long double d = 0x0.0000003ffffffff00000p-16357L;
