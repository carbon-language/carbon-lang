// RUN: %clang_cc1 -emit-llvm -o - -g %s | FileCheck %s
void foo(float A[100], float B[100]) {
  B[12] = A[10] + A[20] + A[30];
}
// CHECK-LABEL: foo
// CHECK: load float* %arrayidx{{.*}}, !dbg [[LOC0:.*]]
// CHECK: load float* %arrayidx{{.*}}, !dbg [[LOC1:.*]]
// CHECK: load float* %arrayidx{{.*}}, !dbg [[LOC2:.*]]
// CHECK: store float {{.*}} float* %arrayidx{{.*}}, !dbg [[LOC3:.*]]
// CHECK-DAG: [[LOC0]] = metadata !{i32 3, i32 11, metadata {{.*}}, null}
// CHECK-DAG: [[LOC1]] = metadata !{i32 3, i32 19, metadata {{.*}}, null}
// CHECK-DAG: [[LOC2]] = metadata !{i32 3, i32 27, metadata {{.*}}, null}
// CHECK-DAG: [[LOC3]] = metadata !{i32 3, i32 3, metadata {{.*}}, null}
