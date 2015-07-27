// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-feature +prfchw -emit-llvm -o - %s | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

void test_m_prefetch(void *p) {
  return _m_prefetch(p);
// CHECK-LABEL: define void @test_m_prefetch
// CHECK: call void @llvm.prefetch({{.*}}, i32 0, i32 3, i32 1)
}

void test_m_prefetch_w(void *p) {
  return _m_prefetchw(p);
// CHECK-LABEL: define void @test_m_prefetch_w
// CHECK: call void @llvm.prefetch({{.*}}, i32 1, i32 3, i32 1)
}
