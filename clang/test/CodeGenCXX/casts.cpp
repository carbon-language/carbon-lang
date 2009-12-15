// RUN: %clang_cc1 %s -emit-llvm -o %t

// PR5248
namespace PR5248 {
struct A {
  void copyFrom(const A &src);
  void addRef(void);
};

void A::copyFrom(const A &src) {
  ((A &)src).addRef();
}
}

