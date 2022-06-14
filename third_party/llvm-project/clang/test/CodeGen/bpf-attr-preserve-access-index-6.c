// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

#define __reloc__ __attribute__((preserve_access_index))

// chain of records, both inner and outer record have attributes.
struct s1 {
  int c;
} __reloc__;
typedef struct s1 __s1;

struct s2 {
  union {
    __s1 b[3];
  } __reloc__;
} __reloc__;
typedef struct s2 __s2;

struct s3 {
  __s2 a;
} __reloc__;
typedef struct s3 __s3;

int test(__s3 *arg) {
  return arg->a.b[2].c;
}

// CHECK: call %struct.s2* @llvm.preserve.struct.access.index.p0s_struct.s2s.p0s_struct.s3s(%struct.s3* elementtype(%struct.s3) %{{[0-9a-z]+}}, i32 0, i32 0)
// CHECK: call %union.anon* @llvm.preserve.struct.access.index.p0s_union.anons.p0s_struct.s2s(%struct.s2* elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 0, i32 0)
// CHECK: call %union.anon* @llvm.preserve.union.access.index.p0s_union.anons.p0s_union.anons(%union.anon* %{{[0-9a-z]+}}, i32 0)
// CHECK: call %struct.s1* @llvm.preserve.array.access.index.p0s_struct.s1s.p0a3s_struct.s1s([3 x %struct.s1]* elementtype([3 x %struct.s1]) %{{[0-9a-z]+}}, i32 1, i32 2)
// CHECK: call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 0)
