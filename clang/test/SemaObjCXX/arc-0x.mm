// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fobjc-arc -verify -fblocks -fobjc-exceptions %s

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

  @try {
  } @catch (auto e) { // expected-error {{'auto' not allowed in exception declaration}}
  }
}

// rdar://problem/11068137
void test1a() {
  __autoreleasing id p; // expected-note 2 {{'p' declared here}}
  (void) [&p] {};
  (void) [p] {}; // expected-error {{cannot capture __autoreleasing variable in a lambda by copy}}
  (void) [=] { (void) p; }; // expected-error {{cannot capture __autoreleasing variable in a lambda by copy}}
}
void test1b() {
  __autoreleasing id v;
  __autoreleasing id &p = v; // expected-note 2 {{'p' declared here}}
  (void) [&p] {};
  (void) [p] {}; // expected-error {{cannot capture __autoreleasing variable in a lambda by copy}}
  (void) [=] { (void) p; }; // expected-error {{cannot capture __autoreleasing variable in a lambda by copy}}
}
void test1c() {
  __autoreleasing id v; // expected-note {{'v' declared here}}
  __autoreleasing id &p = v;
  (void) ^{ (void) p; };
  (void) ^{ (void) v; }; // expected-error {{cannot capture __autoreleasing variable in a block}}
}
