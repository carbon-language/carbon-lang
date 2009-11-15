// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -O0 -S %s -o %t.s
// RUN: FileCheck --input-file=%t.s %s

// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -emit-llvm %s -o %t.ll
// RUN: FileCheck -check-prefix LL --input-file=%t.ll %s

#include <typeinfo>

class test1_B1 {
  virtual void foo() { }
};
class test1_B2 : public test1_B1 {
  virtual void foo() { }
};
class test1_B3 : public test1_B2, public test1_B1 {
  virtual void foo() { }
};
class test1_B4 : virtual public test1_B3 {
  virtual void foo() { }
};
class test1_B5 : virtual test1_B3, test1_B4 {
  virtual void foo() { }
};
class test1_B6 {
  virtual void foo() { }
};
class test1_B7 : public test1_B6, public test1_B5 {
  virtual void foo() { }
};
class test1_D : public test1_B7 {
  virtual void foo() { }
} d1;

// CHECK:__ZTI7test1_D:
// CHECK-NEXT: .quad (__ZTVN10__cxxabiv120__si_class_type_infoE) + 16
// CHECK-NEXT: .quad __ZTS7test1_D
// CHECK-NEXT: .quad __ZTI8test1_B7

// CHECK:__ZTI8test1_B7:
// CHECK-NEXT: .quad (__ZTVN10__cxxabiv121__vmi_class_type_infoE) + 16
// CHECK-NEXT: .quad __ZTS8test1_B7
// CHECK-NEXT: .long 3
// CHECK-NEXT: .long 2
// CHECK-NEXT: .quad __ZTI8test1_B6
// CHECK-NEXT: .quad 2
// CHECK-NEXT: .quad __ZTI8test1_B5
// CHECK-NEXT: .quad 2050

// CHECK:__ZTI8test1_B5:
// CHECK-NEXT: .quad (__ZTVN10__cxxabiv121__vmi_class_type_infoE) + 16
// CHECK-NEXT: .quad __ZTS8test1_B5
// CHECK-NEXT: .long 3
// CHECK-NEXT: .long 2
// CHECK-NEXT: .quad __ZTI8test1_B3
// CHECK-NEXT: .quad 18446744073709545473
// CHECK-NEXT: .quad __ZTI8test1_B4
// CHECK-NEXT: .space 8

// CHECK:__ZTI8test1_B4:
// CHECK-NEXT: .quad (__ZTVN10__cxxabiv121__vmi_class_type_infoE) + 16
// CHECK-NEXT: .quad __ZTS8test1_B4
// CHECK-NEXT: .long 1
// CHECK-NEXT: .long 1
// CHECK-NEXT: .quad __ZTI8test1_B3
// CHECK-NEXT: .quad 18446744073709545475

// CHECK:__ZTI8test1_B6:
// CHECK-NEXT: .quad (__ZTVN10__cxxabiv117__class_type_infoE) + 16
// CHECK-NEXT: .quad __ZTS8test1_B6

// CHECK:__ZTI8test1_B3:
// CHECK-NEXT: .quad (__ZTVN10__cxxabiv121__vmi_class_type_infoE) + 16
// CHECK-NEXT: .quad __ZTS8test1_B3
// CHECK-NEXT: .long 1
// CHECK-NEXT: .long 2
// CHECK-NEXT: .quad __ZTI8test1_B2
// CHECK-NEXT: .quad 2
// CHECK-NEXT: .quad __ZTI8test1_B1
// CHECK-NEXT: .quad 2050

// CHECK:__ZTS8test1_B1:
// CHECK-NEXT: .asciz "8test1_B1"

// CHECK:__ZTI8test1_B1:
// CHECK-NEXT: .quad (__ZTVN10__cxxabiv117__class_type_infoE) + 16
// CHECK-NEXT:. quad __ZTS8test1_B1

// CHECK:__ZTS8test1_B2:
// CHECK-NEXT: .asciz "8test1_B2"

// CHECK:__ZTI8test1_B2:
// CHECK-NEXT: .quad (__ZTVN10__cxxabiv120__si_class_type_infoE) + 16
// CHECK-NEXT: .quad __ZTS8test1_B2
// CHECK-NEXT: .quad __ZTI8test1_B1


class NP { };
void test2_1();
void test2_2(test1_D *dp) {
  test1_D &d = *dp;
  if (typeid(d) == typeid(test1_D))
    test2_1();
  if (typeid(NP) == typeid(test1_D))
    test2_1();
}

// CHECK-LL:define void @_Z7test2_2P7test1_D(%class.test1_B7* %dp) nounwind {
// CHECK-LL:       %tmp1 = load %class.test1_B7** %d
// CHECK-LL-NEXT:  %0 = bitcast %class.test1_B7* %tmp1 to %"class.std::type_info"***
// CHECK-LL-NEXT:  %vtable = load %"class.std::type_info"*** %0
// CHECK-LL-NEXT:  %1 = getelementptr inbounds %"class.std::type_info"** %vtable, i64 -1
// CHECK-LL-NEXT:  %2 = load %"class.std::type_info"** %1
// CHECK-LL-NEXT:  %call = call zeroext i1 @_ZNK3std9type_infoeqERKS0_(%"class.std::type_info"* %2, %"class.std::type_info"* bitcast (%1* @_ZTI7test1_D to %"class.std::type_info"*))

// CHECK-LL:       %call2 = call zeroext i1 @_ZNK3std9type_infoeqERKS0_(%"class.std::type_info"* bitcast (%0* @_ZTI2NP to %"class.std::type_info"*), %"class.std::type_info"* bitcast (%1* @_ZTI7test1_D to %"class.std::type_info"*))
