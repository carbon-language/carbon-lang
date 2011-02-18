/* RUN: %clang_cc1 -fsyntax-only -verify %s
*/

void foo() { 
  goto ; /* expected-error {{expected identifier}} */
}
