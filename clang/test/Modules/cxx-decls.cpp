// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11

// expected-no-diagnostics

void use_implicit_new() { operator new[](3); }

@import dummy;
@import cxx_decls.imported;

void test_delete(int *p) {
  // We can call the normal global deallocation function even though it has only
  // ever been explicitly declared in an unimported submodule.
  delete p;
}

void friend_1(HasFriends s) {
  s.private_thing();
}
void test_friends(HasFriends s) {
  friend_1(s);
  friend_2(s);
}

static_assert(!__is_trivial(HasNontrivialDefaultConstructor), "");
static_assert(!__has_trivial_constructor(HasNontrivialDefaultConstructor), "");

void use_implicit_new_again() { operator new[](3); }
