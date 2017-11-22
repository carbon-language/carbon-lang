// RUN: %clang -S -emit-llvm -o %t %s
// RUN: FileCheck --input-file=%t %s

#pragma GCC visibility push(hidden)

struct Base {
  virtual ~Base() = default;
  virtual void* Alloc() = 0;
};

class Child : public Base {
public:
  Child() = default;
  void* Alloc();
};

void test() {
  Child x;
}

// CHECK: @_ZTV5Child = external hidden unnamed_addr constant
