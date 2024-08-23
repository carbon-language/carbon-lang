// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/index/array_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/index/array_element_access.carbon
fn A() -> (i32, i32) { return (1, 2); }

fn B() -> [i32; 2] { return (1, 2); }

fn Run() {
  var a: [i32; 2] = A();
  var b: i32 = 1;
  var c: i32 = a[b];
  var d: i32 = B()[1];
}

// CHECK:STDOUT: ; ModuleID = 'array_element_access.carbon'
// CHECK:STDOUT: source_filename = "array_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.loc10_37 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @array.loc12_35 = internal constant [2 x i32] [i32 1, i32 2]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @A(ptr sret({ i32, i32 }) %return) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_36.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_36.4.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.loc10_37, i64 8, i1 false)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @B(ptr sret([2 x i32]) %return) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_34.3.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i32 0
// CHECK:STDOUT:   %.loc12_34.6.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @array.loc12_35, i64 8, i1 false)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [2 x i32], align 4
// CHECK:STDOUT:   %.loc15_22.1.temp = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @A(ptr %.loc15_22.1.temp)
// CHECK:STDOUT:   %.loc15_22.3.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc15_22.1.temp, i32 0, i32 0
// CHECK:STDOUT:   %.loc15_22.4 = load i32, ptr %.loc15_22.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc15_22.6.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc15_22.4, ptr %.loc15_22.6.array.index, align 4
// CHECK:STDOUT:   %.loc15_22.8.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc15_22.1.temp, i32 0, i32 1
// CHECK:STDOUT:   %.loc15_22.9 = load i32, ptr %.loc15_22.8.tuple.elem, align 4
// CHECK:STDOUT:   %.loc15_22.11.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc15_22.9, ptr %.loc15_22.11.array.index, align 4
// CHECK:STDOUT:   %b.var = alloca i32, align 4
// CHECK:STDOUT:   store i32 1, ptr %b.var, align 4
// CHECK:STDOUT:   %c.var = alloca i32, align 4
// CHECK:STDOUT:   %.loc17_18 = load i32, ptr %b.var, align 4
// CHECK:STDOUT:   %.loc17_19.1.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i32 %.loc17_18
// CHECK:STDOUT:   %.loc17_19.2 = load i32, ptr %.loc17_19.1.array.index, align 4
// CHECK:STDOUT:   store i32 %.loc17_19.2, ptr %c.var, align 4
// CHECK:STDOUT:   %d.var = alloca i32, align 4
// CHECK:STDOUT:   %.loc18_17.1.temp = alloca [2 x i32], align 4
// CHECK:STDOUT:   call void @B(ptr %.loc18_17.1.temp)
// CHECK:STDOUT:   %.loc18_21.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc18_17.1.temp, i32 0, i32 1
// CHECK:STDOUT:   %.loc18_21.2 = load i32, ptr %.loc18_21.1.array.index, align 4
// CHECK:STDOUT:   store i32 %.loc18_21.2, ptr %d.var, align 4
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
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "array_element_access.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "A", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "B", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
