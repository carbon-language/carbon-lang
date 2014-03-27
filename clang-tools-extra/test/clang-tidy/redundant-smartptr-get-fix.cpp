// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -fix -checks=misc-redundant-smartptr-get --
// RUN: FileCheck -input-file=%t.cpp %s

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
}
