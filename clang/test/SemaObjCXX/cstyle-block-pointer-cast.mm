// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s
// radar 7562285

typedef int (^blocktype)(int a, int b);

@interface A {
    A* a;
    id b;
    Class c;
}
- (blocktype)Meth;
@end

@implementation A
- (blocktype)Meth {
        if (b)
	  return (blocktype)b;
        else if (a)
          return (blocktype)a; // expected-error {{C-style cast from 'A *' to 'blocktype' (aka 'int (^)(int, int)') is not allowed}}
        else
          return (blocktype)c;
}
@end
