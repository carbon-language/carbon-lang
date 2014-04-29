// RUN: $(dirname %s)/check_clang_tidy_output.sh %s misc-redundant-smartptr-get
// REQUIRES: shell

// CHECK-NOT: warning

namespace std {

template <typename T>
class unique_ptr {
  T& operator*() const;
  T* operator->() const;
  T* get() const;
};

template <typename T>
class shared_ptr {
  T& operator*() const;
  T* operator->() const;
  T* get() const;
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

  bool bb = u.get() == nullptr;
  // CHECK: :[[@LINE-1]]:13: warning: Redundant get() call on smart pointer.
  // CHECK: u.get() == nullptr;
  std::shared_ptr<double> *sp;
  bb = nullptr != sp->get();
  // CHECK: :[[@LINE-1]]:19: warning: Redundant get() call on smart pointer.
  // CHECK: nullptr != sp->get();
}

// CHECK-NOT: warning

void Negative() {
  struct NegPtr {
    int* get();
    int* operator->() {
      return &*this->get();
    }
    int& operator*() {
      return *get();
    }
  };

  std::unique_ptr<Bar>* u;
  u->get()->Do();

  Fail1().get()->Do();
  Fail2().get()->Do();
  const Bar& b = *Fail1().get();
  (*Fail2().get()).Do();

  int_ptr ip;
  bool bb = std::unique_ptr<int>().get() == NULL;
  bb = ip.get() == nullptr;
  bb = u->get() == NULL;
}
