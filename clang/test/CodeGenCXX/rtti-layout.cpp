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

// CHECK: define i32 @_Z1fv()
int f() {
  if (to<__pbase_type_info>(typeid(Incomplete *)).__flags != __pbase_type_info::__incomplete_mask)
    return 1;
    
  // Success!
  return 0;
}

#ifdef HARNESS
extern "C" void printf(const char *, ...);

int main() {
  int result = f();
  
  if (result == 0)
    printf("success!\n");
  else
    printf("test %d failed!\n", result);

  return result;
}
#endif


