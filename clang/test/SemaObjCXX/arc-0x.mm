// RUN: %clang_cc1 -std=c++0x -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -verify -fblocks %s

// "Move" semantics, trivial version.
void move_it(__strong id &&from) {
  id to = static_cast<__strong id&&>(from);
}

// Deduction with 'auto'.
@interface A
+ alloc;
- init;
@end

// Ensure that deduction works with lifetime qualifiers.
void deduction(id obj) {
  auto a = [[A alloc] init];
  __strong A** aPtr = &a;

  auto a2([[A alloc] init]);
  __strong A** aPtr2 = &a2;

  __strong id *idp = new auto(obj);

  __strong id array[17];
  for (auto x : array) {
    __strong id *xPtr = &x;
  }
}
