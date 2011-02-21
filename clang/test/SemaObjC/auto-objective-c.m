// RUN: %clang_cc1 -x objective-c -fblocks -fsyntax-only -verify %s

@interface I
{
  id pi;
}
- (id) Meth;
@end


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
