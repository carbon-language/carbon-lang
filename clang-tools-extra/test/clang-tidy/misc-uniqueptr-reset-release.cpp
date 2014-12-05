// RUN: $(dirname %s)/check_clang_tidy.sh %s misc-uniqueptr-reset-release %t
// REQUIRES: shell

namespace std {
template <typename T>
struct unique_ptr {
  unique_ptr<T>();
  explicit unique_ptr<T>(T *);
  template <typename U>
  unique_ptr<T>(unique_ptr<U> &&);
  void reset(T *);
  T *release();
};
} // namespace std

struct Foo {};
struct Bar : Foo {};

std::unique_ptr<Foo> Create();
std::unique_ptr<Foo> &Look();
std::unique_ptr<Foo> *Get();

void f() {
  std::unique_ptr<Foo> a, b;
  std::unique_ptr<Bar> c;
  std::unique_ptr<Foo> *x = &a;
  std::unique_ptr<Foo> *y = &b;

  a.reset(b.release());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer ptr1 = std::move(ptr2) over ptr1.reset(ptr2.release()) [misc-uniqueptr-reset-release]
  // CHECK-FIXES: a = std::move(b);
  a.reset(c.release());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer ptr1 = std::move(ptr2)
  // CHECK-FIXES: a = std::move(c);
  a.reset(Create().release());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: prefer ptr1 = std::move(ptr2)
  // CHECK-FIXES: a = Create();
  x->reset(y->release());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: prefer ptr1 = std::move(ptr2)
  // CHECK-FIXES: *x = std::move(*y);
  Look().reset(Look().release());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer ptr1 = std::move(ptr2)
  // CHECK-FIXES: Look() = std::move(Look());
  Get()->reset(Get()->release());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: prefer ptr1 = std::move(ptr2)
  // CHECK-FIXES: *Get() = std::move(*Get());
}

