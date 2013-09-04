// Without inline namespace:
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -replace-auto_ptr %t.cpp -- -I %S/Inputs std=c++11
// RUN: FileCheck -input-file=%t.cpp %s
//
// With inline namespace:
//
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -replace-auto_ptr %t.cpp -- -I %S/Inputs std=c++11 \
// RUN:                                           -DUSE_INLINE_NAMESPACE=1
// RUN: FileCheck -input-file=%t.cpp %s

#include "memory_stub.h"

void takes_ownership_fn(std::auto_ptr<int> x);
// CHECK: void takes_ownership_fn(std::unique_ptr<int> x);

std::auto_ptr<int> get_by_value();
// CHECK: std::unique_ptr<int> get_by_value();

class Wrapper {
public:
  std::auto_ptr<int> &get_wrapped();

private:
  std::auto_ptr<int> wrapped;
};

void f() {
  std::auto_ptr<int> a, b, c;
  // CHECK: std::unique_ptr<int> a, b, c;
  Wrapper wrapper_a, wrapper_b;

  a = b;
  // CHECK: a = std::move(b);

  wrapper_a.get_wrapped() = wrapper_b.get_wrapped();
  // CHECK: wrapper_a.get_wrapped() = std::move(wrapper_b.get_wrapped());

  // Test that 'std::move()' is inserted when call to the
  // copy-constructor are made.
  takes_ownership_fn(c);
  // CHECK: takes_ownership_fn(std::move(c));
  takes_ownership_fn(wrapper_a.get_wrapped());
  // CHECK: takes_ownership_fn(std::move(wrapper_a.get_wrapped()));

  std::auto_ptr<int> d[] = { std::auto_ptr<int>(new int(1)),
                             std::auto_ptr<int>(new int(2)) };
  std::auto_ptr<int> e = d[0];
  // CHECK: std::unique_ptr<int> d[] = { std::unique_ptr<int>(new int(1)),
  // CHECK-NEXT:                         std::unique_ptr<int>(new int(2)) };
  // CHECK-NEXT: std::unique_ptr<int> e = std::move(d[0]);

  // Test that std::move() is not used when assigning an rvalue
  std::auto_ptr<int> f;
  f = std::auto_ptr<int>(new int(0));
  // CHECK: std::unique_ptr<int> f;
  // CHECK-NEXT: f = std::unique_ptr<int>(new int(0));

  std::auto_ptr<int> g = get_by_value();
  // CHECK: std::unique_ptr<int> g = get_by_value();
}
