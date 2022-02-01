// RUN: %clang_cc1 -triple mips-none-linux-gnu -emit-llvm -w -o - %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -emit-llvm -w -target-abi n32 -o - %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -emit-llvm -w -o - %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s

int check_char() {
  return sizeof(char);
// ALL: ret i32 1
}

int check_short() {
  return sizeof(short);
// ALL: ret i32 2
}

int check_int() {
  return sizeof(int);
// ALL: ret i32 4
}

int check_long() {
  return sizeof(long);
// O32: ret i32 4
// N32: ret i32 4
// N64: ret i32 8
}

int check_longlong() {
  return sizeof(long long);
// ALL: ret i32 8
}

int check_fp16() {
  return sizeof(__fp16);
// ALL: ret i32 2
}

int check_float() {
  return sizeof(float);
// ALL: ret i32 4
}

int check_double() {
  return sizeof(double);
// ALL: ret i32 8
}

int check_longdouble() {
  return sizeof(long double);
// O32: ret i32 8
// N32: ret i32 16
// N64: ret i32 16
}

int check_floatComplex() {
  return sizeof(float _Complex);
// ALL: ret i32 8
}

int check_doubleComplex() {
  return sizeof(double _Complex);
// ALL: ret i32 16
}

int check_longdoubleComplex() {
  return sizeof(long double _Complex);
// O32: ret i32 16
// N32: ret i32 32
// N64: ret i32 32
}

int check_bool() {
  return sizeof(_Bool);
// ALL: ret i32 1
}

int check_wchar() {
  return sizeof(__WCHAR_TYPE__);
// ALL: ret i32 4
}

int check_wchar_is_unsigned() {
  return (__WCHAR_TYPE__)-1 > (__WCHAR_TYPE__)0;
// ALL: ret i32 0
}

int check_ptr() {
  return sizeof(void *);
// O32: ret i32 4
// N32: ret i32 4
// N64: ret i32 8
}

