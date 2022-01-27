// RUN: %clang_cc1 -emit-llvm -std=c++11 -triple x86_64-pc-linux-gnu -o- %s | FileCheck %s

// Global @x:
// CHECK: [[X_GLOBAL:@[^ ]+]]{{.*}}thread_local global

// returned somewhere in TLS wrapper:
// CHECK: ret{{.*}}[[X_GLOBAL]]

template <typename T> class unique_ptr {
  template <typename F, typename S> struct pair {
    F first;
    S second;
  };
  pair<T *, int> data;
public:
  constexpr unique_ptr() noexcept : data() {}
  explicit unique_ptr(T *p) noexcept : data() {}
};

thread_local unique_ptr<int> x;
int main() { x = unique_ptr<int>(new int(5)); }

