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


// rdar://9036633
int main() {
  auto int auto_i = 7; // expected-warning {{'auto' storage class specifier is redundant and will be removed in future releases}}
}
