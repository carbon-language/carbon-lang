/* RUN: clang -fsyntax-only -verify %s
 */

int x = 000000080;  /* expected-error {{invalid digit}} */

int y = 0000\
00080;             /* expected-error {{invalid digit}} */

