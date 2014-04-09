// RUN: $(dirname %s)/check_clang_tidy_fix.sh %s misc-redundant-smartptr-get %t

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

struct Bar {
  void Do();
  void ConstDo() const;
};
struct BarPtr {
  Bar* operator->();
  Bar& operator*();
  Bar* get();
};
struct int_ptr {
  int* get();
  int* operator->();
  int& operator*();
};

void Positive() {
  BarPtr u;
  // CHECK: BarPtr u;
  BarPtr().get()->Do();
  // CHECK: BarPtr()->Do();

  u.get()->ConstDo();
  // CHECK: u->ConstDo();

  Bar& b = *BarPtr().get();
  // CHECK: Bar& b = *BarPtr();

  (*BarPtr().get()).ConstDo();
  // CHECK: (*BarPtr()).ConstDo();

  BarPtr* up;
  (*up->get()).Do();
  // CHECK: (**up).Do();

  int_ptr ip;
  int i = *ip.get();
  // CHECK: int i = *ip;

  std::unique_ptr<int> uu;
  std::shared_ptr<double> *ss;
  bool bb = uu.get() == nullptr;
  // CHECK: bool bb = uu == nullptr;
  bb = nullptr != ss->get();
  // CHECK: bb = nullptr != *ss;
}
