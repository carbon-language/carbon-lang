// RUN: %clang_cc1 %s -triple=i686-pc-linux -emit-llvm -o - | FileCheck %s 
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:            -triple i686--windows -emit-llvm %s -o - \ 
// RUN:   | FileCheck %s -check-prefix CHECK  -check-prefix MSC

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#ifdef _MSC_VER
#include <Intrin.h>
#else
#include <immintrin.h>
#endif

#ifdef _MSC_VER
unsigned char test_rotl8(unsigned char v, unsigned char s) {
  //MSC-LABEL: test_rotl8
  //MSC-NOT: call
  return _rotl8(v, s);
}

unsigned char test_rotr8(unsigned char v, unsigned char s) {
  //MSC-LABEL: test_rotr8
  //MSC-NOT: call
  return _rotr8(v, s);
}

unsigned short test_rotl16(unsigned short v, unsigned char s) {
  //MSC-LABEL: test_rotl16
  //MSC-NOT: call
  return _rotl16(v, s);
}

unsigned short test_rotr16(unsigned short v, unsigned char s) {
  //MSC-LABEL: test_rotr16
  //MSC-NOT: call
  return _rotr16(v, s);
}

unsigned __int64 test_rotl64(unsigned __int64 v, int s) {
  //MSC-LABEL: test_rotl64
  //MSC-NOT: call
  return _rotl64(v, s);
}

unsigned __int64 test_rotr64(unsigned __int64 v, int s) {
  //MSC-LABEL: test_rotr64
  //MSC-NOT: call
  return _rotr64(v, s);
}
#endif

unsigned short test_rotwl(unsigned short v, unsigned short s) {
  //CHECK-LABEL: test_rotwl
  //CHECK-NOT: call
  return _rotwl(v, s);
}

unsigned short test_rotwr(unsigned short v, unsigned short s) {
  //CHECK-LABEL: test_rotwr
  //CHECK-NOT: call
  return _rotwr(v, s);
}

unsigned int test_rotl(unsigned int v, int s) {
  //CHECK-LABEL: test_rotl
  //CHECK-NOT: call
  return _rotl(v, s);
}

unsigned int test_rotr(unsigned int v, int s) {
  //CHECK-LABEL: test_rotr
  //CHECK-NOT: call
  return _rotr(v, s);
}

unsigned long test_lrotl(unsigned long v, int s) {
  //CHECK-LABEL: test_lrotl
  //CHECK-NOT: call
  return _lrotl(v, s);
}

unsigned long test_lrotr(unsigned long v, int s) {
  //CHECK-LABEL: test_lrotr
  //CHECK-NOT: call
  return _lrotr(v, s);
}

//CHECK-LABEL: attributes
