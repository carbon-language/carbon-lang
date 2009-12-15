// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef struct NSSize {
     		int width;
		struct {
		  int dim;
		} inner;
} NSSize;

@interface Foo  {
        NSSize _size;
}
@property NSSize size;
@end

void foo() { 
        Foo *f;
        f.size.width = 2.2; // expected-error {{cannot assign to a sub-structure of an ivar using property assignment syntax}}
	f.size.inner.dim = 200; // expected-error {{cannot assign to a sub-structure of an ivar using property assignment syntax}}
}
