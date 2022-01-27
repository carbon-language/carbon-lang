// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://8592156

typedef struct objc_class *Class;
@interface A
-(Class) foo;
@end

void f0(A *a) { int x = [[a foo] baz]; } // expected-warning {{method '+baz' not found (return type defaults to 'id')}} \
					 // expected-warning {{ncompatible pointer to integer conversion initializing 'int' with an expression of type 'id'}}
