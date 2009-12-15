/* RUN: %clang_cc1 -std=c90 -fsyntax-only %s -verify -pedantic-errors
 */

enum { cast_hex = (long) (
      0x0p-1   /* expected-error {{hexadecimal floating constants are a C99 feature}} */
     ) };

/* PR2477 */
int test1(int a,int b) {return a//* This is a divide followed by block comment in c89 mode */
b;}

// comment accepted as extension    /* expected-error {{// comments are not allowed in this language}}

