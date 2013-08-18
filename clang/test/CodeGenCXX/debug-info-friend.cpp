// RUN: %clang -emit-llvm -S -g %s -o - | FileCheck %s

class MyFriend;

class SomeClass {
  friend class MyFriend;
  typedef int SomeType;
};

SomeClass *x;

struct MyFriend {
  static void func(SomeClass::SomeType) {
  }
};

// CHECK: DW_TAG_friend
