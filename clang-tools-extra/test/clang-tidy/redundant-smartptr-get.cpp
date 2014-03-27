// RUN: clang-tidy --checks=misc-redundant-smartptr-get %s -- | FileCheck %s

// CHECK-NOT: warning

namespace std {

template <typename T>
struct MakeRef {
  typedef T& type;
};

template <typename T>
struct unique_ptr {
  T* get();
  T* operator->();
  // This simulates libstdc++'s implementation of unique_ptr.
  typename MakeRef<T>::type operator*();
};
}  // namespace std

struct int_ptr {
  int* get();
  int* operator->();
  int& operator*();
};

struct Bar {
  void Do();
  void ConstDo() const;
};

struct Fail1 {
  Bar* get();
};
struct Fail2 {
  Bar* get();
  int* operator->();
  int& operator*();
};

void Positive() {
  std::unique_ptr<Bar> u;
  std::unique_ptr<Bar>().get()->Do();
  // CHECK: :[[@LINE-1]]:3: warning: Redundant get() call on smart pointer. [misc-redundant-smartptr-get]
  // CHECK: std::unique_ptr<Bar>().get()->Do();

  u.get()->ConstDo();
  // CHECK: :[[@LINE-1]]:3: warning: Redundant get() call on smart pointer.
  // CHECK: u.get()->ConstDo();

  Bar& b = *std::unique_ptr<Bar>().get();
  // CHECK: :[[@LINE-1]]:13: warning: Redundant get() call on smart pointer.
  // CHECK: Bar& b = *std::unique_ptr<Bar>().get();

  (*std::unique_ptr<Bar>().get()).ConstDo();
  // CHECK: :[[@LINE-1]]:5: warning: Redundant get() call on smart pointer.
  // CHECK: (*std::unique_ptr<Bar>().get()).ConstDo();

  std::unique_ptr<Bar>* up;
  (*up->get()).Do();
  // CHECK: :[[@LINE-1]]:5: warning: Redundant get() call on smart pointer.
  // CHECK: (*up->get()).Do();

  int_ptr ip;
  int i = *ip.get();
  // CHECK: :[[@LINE-1]]:12: warning: Redundant get() call on smart pointer.
  // CHECK: int i = *ip.get();
}

// CHECK-NOT: warning

void Negative() {
  std::unique_ptr<Bar>* u;
  u->get()->Do();

  Fail1().get()->Do();
  Fail2().get()->Do();
  const Bar& b = *Fail1().get();
  (*Fail2().get()).Do();
}
