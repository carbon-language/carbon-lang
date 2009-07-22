/* RUN: clang-cc -fsyntax-only -verify %s
*/
void foo() { 
  break; /* expected-error {{'break' statement not in loop or switch statement}} */
}

void foo2() { 
  continue; /* expected-error {{'continue' statement not in loop statement}} */
}
