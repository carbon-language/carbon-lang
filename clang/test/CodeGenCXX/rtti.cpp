// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -O0 -S %s -o %t.s
// RUN: FileCheck --input-file=%t.s %s

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
