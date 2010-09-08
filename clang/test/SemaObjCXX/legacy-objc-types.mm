// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar:// 8400356

struct objc_object;

void f(id ptr) { } // expected-note {{previous definition is here}}
void f(objc_object* ptr) { }	// expected-error {{redefinition of 'f'}}

struct objc_class;

void g(Class ptr) {} // expected-note {{previous definition is here}}
void g(objc_class* ptr) { }	// expected-error {{redefinition of 'g'}}

void h(Class ptr, objc_object* ptr1) {} // expected-note {{previous definition is here}}
void h(objc_class* ptr, id ptr1) {}	// expected-error {{redefinition of 'h'}}

void i(Class ptr, objc_object* ptr1);
void i(objc_class* ptr, id ptr1) {}
void i(objc_class* ptr, objc_object* ptr1);

