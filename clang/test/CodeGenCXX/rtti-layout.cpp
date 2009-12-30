// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -O3 -o - | FileCheck %s
#include <typeinfo>

// vtables.
extern "C" {
  const void *_ZTVN10__cxxabiv123__fundamental_type_infoE;
  const void *_ZTVN10__cxxabiv117__class_type_infoE;
  const void *_ZTVN10__cxxabiv120__si_class_type_infoE;
  const void *_ZTVN10__cxxabiv119__pointer_type_infoE;
  const void *_ZTVN10__cxxabiv129__pointer_to_member_type_infoE;
};
#define fundamental_type_info_vtable _ZTVN10__cxxabiv123__fundamental_type_infoE
#define class_type_info_vtable _ZTVN10__cxxabiv117__class_type_infoE
#define si_class_type_info_vtable _ZTVN10__cxxabiv120__si_class_type_infoE
#define pointer_type_info_vtable _ZTVN10__cxxabiv119__pointer_type_infoE
#define pointer_to_member_type_info_vtable _ZTVN10__cxxabiv129__pointer_to_member_type_infoE

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

#define CHECK(x) if (!(x)) return __LINE__
#define CHECK_VTABLE(type, vtable) if (&vtable##_type_info_vtable + 2 != (((void **)&(typeid(type)))[0])) return __LINE__

// CHECK: define i32 @_Z1fv()
int f() {
  // Vectors should be treated as fundamental types.
  typedef short __v4hi __attribute__ ((__vector_size__ (8)));
  CHECK_VTABLE(__v4hi, fundamental);
  
  // Pointers to incomplete classes.
  CHECK_VTABLE(Incomplete *, pointer);
  CHECK(to<__pbase_type_info>(typeid(Incomplete *)).__flags == __pbase_type_info::__incomplete_mask);
  CHECK(to<__pbase_type_info>(typeid(Incomplete **)).__flags == __pbase_type_info::__incomplete_mask);
  CHECK(to<__pbase_type_info>(typeid(Incomplete ***)).__flags == __pbase_type_info::__incomplete_mask);

  // Member pointers.
  CHECK_VTABLE(int Incomplete::*, pointer_to_member);
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


