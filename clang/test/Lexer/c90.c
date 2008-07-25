/* RUN: clang -std=c90 -fsyntax-only %s -verify -pedantic-errors
 */

enum { cast_hex = (long) (
      0x0p-1   /* expected-error {{hexadecimal floating constants are a C99 feature}} */
     ) };
