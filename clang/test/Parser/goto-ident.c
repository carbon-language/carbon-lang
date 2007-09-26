/* RUN: clang -parse-ast -verify %s
*/

void foo() { 
  goto ; /* expected-error {{expected identifier}} */
}
