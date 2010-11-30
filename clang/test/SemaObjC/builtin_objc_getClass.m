// RUN: %clang_cc1 %s -fsyntax-only -std=gnu99 -verify
// rdar://8592641
Class f0() { return objc_getClass("a"); } // expected-warning {{implicitly declaring C library function 'objc_getClass' with type 'id (const char *)'}} \
					  // expected-note {{please include the header <objc/runtime.h> or explicitly provide a declaration for 'objc_getClass'}}
