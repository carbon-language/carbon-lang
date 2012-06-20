// RUN: %clang_cc1 -x objective-c++ -fblocks -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 %s -verify -emit-llvm -o %t
// rdar://8979379

@interface A
@end

@interface B : A
@end

void f(int (^bl)(B* b));

// Test1
void g() {
  f(^(A* a) { return 0; });
}

// Test2
void g1() {
  int (^bl)(B* b) = ^(A* a) { return 0; };
}

// Test3
@protocol NSObject;

void bar(id(^)(void));

void foo(id <NSObject>(^objectCreationBlock)(void)) {
    return bar(objectCreationBlock);
}

// Test4
struct S {
  S *(^a)() = ^{ // expected-warning {{C++11}}
    return this;
  };
};
S s;

// Test5
struct X {
  void f() {
    ^ {
      struct Nested { Nested *ptr = this; }; // expected-warning {{C++11}}
    } ();
  };
};
