/* RUN: clang -parse-ast-check %s
*/
int foo() { 
  break; /* expected-error {{'break' statement not in loop or switch statement}} */
}

int foo2() { 
  continue; /* expected-error {{'continue' statement not in loop statement}} */
}
