/* RUN: clang -parse-ast-check %s
*/

int foo(int A) { int A; } /* expected-error {{redefinition of 'A'}} \
                             expected-error {{previous definition is here}} */
