/* RUN: clang-cc -fsyntax-only -verify -std=c89 -pedantic %s
 */

@class NSArray;

void f(NSArray *a) {
    id keys;
    for (int i in a); /* expected-error{{selector element type 'int' is not a valid object}} */
    for ((id)2 in a); /* expected-error{{selector element is not a valid lvalue}} */
    for (2 in a); /* expected-error{{selector element is not a valid lvalue}} */
  
  /* This should be ok, 'thisKey' should be scoped to the loop in question,
   * and no diagnostics even in pedantic mode should happen.
   * rdar://6814674
   */
  for (id thisKey in keys);
  for (id thisKey in keys);
}
