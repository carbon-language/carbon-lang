// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 4] = (
    8,
    9,
    0x8,
    0b1000,
  );
  var floats: [f64; 6] = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8,
  );
}

// CHECK:STDOUT: ; ModuleID = 'numeric_literals.carbon'
// CHECK:STDOUT: source_filename = "numeric_literals.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.1.loc15_4 = internal constant [4 x i32] [i32 8, i32 9, i32 8, i32 8]
// CHECK:STDOUT: @array.2.loc23_4 = internal constant [6 x double] [double 9.000000e-01, double 8.000000e+00, double 8.000000e+01, double 1.000000e+07, double 1.000000e+08, double 1.000000e-08]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ints.var = alloca [4 x i32], align 4
// CHECK:STDOUT:   %.loc15_3.3.array.index = getelementptr inbounds [4 x i32], ptr %ints.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc15_3.6.array.index = getelementptr inbounds [4 x i32], ptr %ints.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc15_3.9.array.index = getelementptr inbounds [4 x i32], ptr %ints.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc15_3.12.array.index = getelementptr inbounds [4 x i32], ptr %ints.var, i32 0, i32 3
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %ints.var, ptr align 4 @array.1.loc15_4, i64 16, i1 false)
// CHECK:STDOUT:   %floats.var = alloca [6 x double], align 8
// CHECK:STDOUT:   %.loc23_3.3.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc23_3.6.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc23_3.9.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc23_3.12.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 3
// CHECK:STDOUT:   %.loc23_3.15.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 4
// CHECK:STDOUT:   %.loc23_3.18.array.index = getelementptr inbounds [6 x double], ptr %floats.var, i32 0, i32 5
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %floats.var, ptr align 8 @array.2.loc23_4, i64 48, i1 false)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
