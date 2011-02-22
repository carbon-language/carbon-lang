// RUN: %clang_cc1 -x objective-c -fblocks -fsyntax-only -verify %s

@interface I
{
  id pi;
}
- (id) Meth;
@end

// Objective-C does not support trailing return types, so check we don't get
// the C++ diagnostic suggesting we forgot one.
auto noTrailingReturnType(); // expected-error {{'auto' not allowed in function return type}}

typedef int (^P) (int x);

@implementation I
- (id) Meth {
  auto p = [pi Meth];
  return p;
}

- (P) bfunc {
  auto my_block = ^int (int x) {return x; };
  my_block(1);
  return my_block;
}
@end
