/* RUN: clang %s -fsyntax-only -pedantic -verify
 */

extern struct {int a;} x; // expected-error{{previous definition is here}}
extern struct {int a;} x; // expected-error{{redefinition of 'x'}}

struct x;
int a(struct x* b) {
// FIXME: This test currently fails
// Per C99 6.7.2.3, since the outer and inner "struct x"es have different
// scopes, they don't refer to the same type, and are therefore incompatible
struct x {int a;} *c = b;
}

struct x {int a;} r;
int b() {
// FIXME: This test currently also fails
struct x {char x;} s = r;
}
