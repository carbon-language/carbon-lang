/* RUN: clang -parse-ast-check %s
*/

void foo() { 
  goto ; /* expected-error {{expected identifier}} */
}
