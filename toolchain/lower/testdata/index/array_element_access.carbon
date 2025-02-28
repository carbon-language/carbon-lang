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

fn B() -> array(i32, 2) { return (1, 2); }

fn Run() {
  var a: array(i32, 2) = A();
  var b: i32 = 1;
  var c: i32 = a[b];
  var d: i32 = B()[1];
}

// CHECK:STDOUT: ; ModuleID = 'array_element_access.carbon'
// CHECK:STDOUT: source_filename = "array_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.loc10_37 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @array.loc12_40 = internal constant [2 x i32] [i32 1, i32 2]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CA.Main(ptr sret({ i32, i32 }) %return) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.loc10_37, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CB.Main(ptr sret([2 x i32]) %return) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_39.3.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 0, !dbg !10
// CHECK:STDOUT:   %.loc12_39.6.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 1, !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @array.loc12_40, i64 8, i1 false), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [2 x i32], align 4, !dbg !13
// CHECK:STDOUT:   %b.var = alloca i32, align 4, !dbg !13
// CHECK:STDOUT:   %c.var = alloca i32, align 4, !dbg !13
// CHECK:STDOUT:   %d.var = alloca i32, align 4, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %a.var), !dbg !13
// CHECK:STDOUT:   %.loc15_28.1.temp = alloca { i32, i32 }, align 8, !dbg !14
// CHECK:STDOUT:   call void @_CA.Main(ptr %.loc15_28.1.temp), !dbg !14
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc15_28.1.temp, i32 0, i32 0, !dbg !14
// CHECK:STDOUT:   %.loc15_28.3 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !14
// CHECK:STDOUT:   %.loc15_28.4.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 0, !dbg !14
// CHECK:STDOUT:   store i32 %.loc15_28.3, ptr %.loc15_28.4.array.index, align 4, !dbg !14
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc15_28.1.temp, i32 0, i32 1, !dbg !14
// CHECK:STDOUT:   %.loc15_28.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !14
// CHECK:STDOUT:   %.loc15_28.7.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 1, !dbg !14
// CHECK:STDOUT:   store i32 %.loc15_28.6, ptr %.loc15_28.7.array.index, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %b.var), !dbg !13
// CHECK:STDOUT:   store i32 1, ptr %b.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %c.var), !dbg !13
// CHECK:STDOUT:   %.loc17_18 = load i32, ptr %b.var, align 4, !dbg !16
// CHECK:STDOUT:   %.loc17_19.1.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i32 %.loc17_18, !dbg !17
// CHECK:STDOUT:   %.loc17_19.2 = load i32, ptr %.loc17_19.1.array.index, align 4, !dbg !17
// CHECK:STDOUT:   store i32 %.loc17_19.2, ptr %c.var, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %d.var), !dbg !13
// CHECK:STDOUT:   %.loc18_18.1.temp = alloca [2 x i32], align 4, !dbg !19
// CHECK:STDOUT:   call void @_CB.Main(ptr %.loc18_18.1.temp), !dbg !19
// CHECK:STDOUT:   %.loc18_21.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc18_18.1.temp, i32 0, i32 1, !dbg !19
// CHECK:STDOUT:   %.loc18_21.2 = load i32, ptr %.loc18_21.1.array.index, align 4, !dbg !19
// CHECK:STDOUT:   store i32 %.loc18_21.2, ptr %d.var, align 4, !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 0, 1, 4, 6, 7, 2, 3, 5, 8, 9, 10 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "array_element_access.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 31, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 10, column: 24, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 34, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 12, column: 27, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 15, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 26, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 18, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 16, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 16, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 3, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 14, column: 1, scope: !12)
