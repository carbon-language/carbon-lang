// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -O3 -o - | FileCheck %s
#include <typeinfo>

class __pbase_type_info : public std::type_info {
public:
  unsigned int __flags;
  const std::type_info *__pointee;

  enum __masks {
    __const_mask = 0x1,
    __volatile_mask = 0x2,
    __restrict_mask = 0x4,
    __incomplete_mask = 0x8,
    __incomplete_class_mask = 0x10
  };
};

template<typename T> const T& to(const std::type_info &info) {
return static_cast<const T&>(info);
}
struct Incomplete;

struct A { };

#define CHECK(x) if (!(x)) return __LINE__;

// CHECK: define i32 @_Z1fv()
int f() {
  // Pointers to incomplete classes.
  CHECK(to<__pbase_type_info>(typeid(Incomplete *)).__flags == __pbase_type_info::__incomplete_mask);
  CHECK(to<__pbase_type_info>(typeid(Incomplete **)).__flags == __pbase_type_info::__incomplete_mask);
  CHECK(to<__pbase_type_info>(typeid(Incomplete ***)).__flags == __pbase_type_info::__incomplete_mask);

  // Member pointers.
  CHECK(to<__pbase_type_info>(typeid(int Incomplete::*)).__flags == __pbase_type_info::__incomplete_class_mask);
  CHECK(to<__pbase_type_info>(typeid(Incomplete Incomplete::*)).__flags == (__pbase_type_info::__incomplete_class_mask | __pbase_type_info::__incomplete_mask));
  CHECK(to<__pbase_type_info>(typeid(Incomplete A::*)).__flags == (__pbase_type_info::__incomplete_mask));

  // Success!
  // CHECK: ret i32 0
  return 0;
}

#ifdef HARNESS
extern "C" void printf(const char *, ...);

int main() {
  int result = f();
  
  if (result == 0)
    printf("success!\n");
  else
    printf("test on line %d failed!\n", result);

  return result;
}
#endif


