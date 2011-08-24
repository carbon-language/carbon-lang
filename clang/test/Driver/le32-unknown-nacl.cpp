// RUN: %clang -ccc-host-triple le32-unknown-nacl -ccc-echo %s -emit-llvm -c 2>&1 | FileCheck %s -check-prefix=ECHO
// RUN: %clang -ccc-host-triple le32-unknown-nacl %s -emit-llvm -S -c -o - | FileCheck %s

// ECHO: clang{{.*}} -cc1 {{.*}}le32-unknown-nacl.c

// Check platform defines
#include <stddef.h>

extern "C" {

#ifdef __native_client__
void __native_client__defined() {
  // CHECK: __native_client__defined
}
#endif

#ifdef __le32__
void __le32__defined() {
  // CHECK: __le32__defined
}
#endif

#ifdef __pnacl__
void __pnacl__defined() {
  // CHECK: __pnacl__defined
}
#endif

// Check types

// CHECK: signext i8 @check_char()
char check_char() { return 0; }

// CHECK: signext i16 @check_short()
short check_short() { return 0; }

// CHECK: i32 @check_int()
int check_int() { return 0; }

// CHECK: i32 @check_long()
long check_long() { return 0; }

// CHECK: i64 @check_longlong()
long long check_longlong() { return 0; }

// CHECK: zeroext i8 @check_uchar()
unsigned char check_uchar() { return 0; }

// CHECK: zeroext i16 @check_ushort()
unsigned short check_ushort() { return 0; }

// CHECK: i32 @check_uint()
unsigned int check_uint() { return 0; }

// CHECK: i32 @check_ulong()
unsigned long check_ulong() { return 0; }

// CHECK: i64 @check_ulonglong()
unsigned long long check_ulonglong() { return 0; }

// CHECK: i32 @check_size_t()
size_t check_size_t() { return 0; }

// CHECK: float @check_float()
float check_float() { return 0; }

// CHECK: double @check_double()
double check_double() { return 0; }

// CHECK: double @check_longdouble()
long double check_longdouble() { return 0; }

}

#include <stdarg.h>

template<int> void Switch();
template<> void Switch<4>();
template<> void Switch<8>();
template<> void Switch<16>();

void check_pointer_size() {
  // CHECK: SwitchILi4
  Switch<sizeof(void*)>();

  // CHECK: SwitchILi8
  Switch<sizeof(long long)>();

  // CHECK: SwitchILi16
  Switch<sizeof(va_list)>();
}
