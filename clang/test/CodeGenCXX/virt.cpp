// RUN: clang-cc %s -emit-llvm -o - -std=c++0x

class A {
public:
  virtual void foo();
};

static_assert (sizeof (A) == (sizeof(void *)), "vtable pointer layout");
