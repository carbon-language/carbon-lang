/* RUN: clang-cc -fsyntax-only -verify %s
*/

void foo() { 
  goto ; /* expected-error {{expected identifier}} */
}
