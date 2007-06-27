/* RUN: clang -parse-ast-check %s
*/

int foo(int A) { /* expected-error {{previous definition is here}} */
  int A; /* expected-error {{redefinition of 'A'}} */
}
