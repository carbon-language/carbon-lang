// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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
  var _: i32 = a[b];
  var _: i32 = B()[1];
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'array_element_access.carbon'
// CHECK:STDOUT: source_filename = "array_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.102.loc12_37 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @array.loc14_40 = internal constant [2 x i32] [i32 1, i32 2]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CA.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.102.loc12_37, i64 8, i1 false), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CB.Main(ptr sret([2 x i32]) %return) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_39.3.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 0, !dbg !38
// CHECK:STDOUT:   %.loc14_39.6.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 1, !dbg !38
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @array.loc14_40, i64 8, i1 false), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [2 x i32], align 4, !dbg !41
// CHECK:STDOUT:   %.loc17_28.1.temp = alloca { i32, i32 }, align 4, !dbg !42
// CHECK:STDOUT:   %b.var = alloca i32, align 4, !dbg !44
// CHECK:STDOUT:   %_.var.loc19 = alloca i32, align 4, !dbg !46
// CHECK:STDOUT:   %_.var.loc20 = alloca i32, align 4, !dbg !50
// CHECK:STDOUT:   %.loc20_18.1.temp = alloca [2 x i32], align 4, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_28.1.temp), !dbg !42
// CHECK:STDOUT:   call void @_CA.Main(ptr %.loc17_28.1.temp), !dbg !42
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc17_28.1.temp, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   %.loc17_28.3 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !42
// CHECK:STDOUT:   %.loc17_28.4.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 0, !dbg !42
// CHECK:STDOUT:   store i32 %.loc17_28.3, ptr %.loc17_28.4.array.index, align 4, !dbg !42
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc17_28.1.temp, i32 0, i32 1, !dbg !42
// CHECK:STDOUT:   %.loc17_28.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !42
// CHECK:STDOUT:   %.loc17_28.7.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 1, !dbg !42
// CHECK:STDOUT:   store i32 %.loc17_28.6, ptr %.loc17_28.7.array.index, align 4, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !44
// CHECK:STDOUT:   store i32 1, ptr %b.var, align 4, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc19), !dbg !46
// CHECK:STDOUT:   %.loc19_18 = load i32, ptr %b.var, align 4, !dbg !48
// CHECK:STDOUT:   %.loc19_19.1.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i32 %.loc19_18, !dbg !47
// CHECK:STDOUT:   %.loc19_19.2 = load i32, ptr %.loc19_19.1.array.index, align 4, !dbg !47
// CHECK:STDOUT:   store i32 %.loc19_19.2, ptr %_.var.loc19, align 4, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc20), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_18.1.temp), !dbg !51
// CHECK:STDOUT:   call void @_CB.Main(ptr %.loc20_18.1.temp), !dbg !51
// CHECK:STDOUT:   %.loc20_21.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc20_18.1.temp, i32 0, i32 1, !dbg !51
// CHECK:STDOUT:   %.loc20_21.2 = load i32, ptr %.loc20_21.1.array.index, align 4, !dbg !51
// CHECK:STDOUT:   store i32 %.loc20_21.2, ptr %_.var.loc20, align 4, !dbg !50
// CHECK:STDOUT:   call void @"_COp.8014ee1b1a984df0:core.Destroy.Core"(ptr %.loc20_18.1.temp), !dbg !51
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %_.var.loc20), !dbg !50
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %_.var.loc19), !dbg !46
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %b.var), !dbg !44
// CHECK:STDOUT:   call void @"_COp.e554695bf635d9c1:core.Destroy.Core"(ptr %.loc17_28.1.temp), !dbg !42
// CHECK:STDOUT:   call void @"_COp.8014ee1b1a984df0:core.Destroy.Core"(ptr %a.var), !dbg !41
// CHECK:STDOUT:   ret i32 0, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8014ee1b1a984df0:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e554695bf635d9c1:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.f58524831c37807e:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.8014ee1b1a984df0:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "array_element_access.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !21}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main", scope: null, file: !18, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !33 = !DILocation(line: 12, column: 31, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 12, column: 24, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main", scope: null, file: !18, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !38 = !DILocation(line: 14, column: 34, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 14, column: 27, scope: !35)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 16, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 3, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 17, column: 26, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 18, column: 3, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 3, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 19, column: 16, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 19, column: 18, scope: !40)
// CHECK:STDOUT: !50 = !DILocation(line: 20, column: 3, scope: !40)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 16, scope: !40)
// CHECK:STDOUT: !53 = !DILocation(line: 16, column: 1, scope: !40)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !20)
// CHECK:STDOUT: !56 = !DILocation(line: 20, column: 16, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8014ee1b1a984df0:core.Destroy.Core", scope: null, file: !18, line: 20, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !21)
// CHECK:STDOUT: !60 = !DILocation(line: 20, column: 16, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e554695bf635d9c1:core.Destroy.Core", scope: null, file: !18, line: 17, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !21)
// CHECK:STDOUT: !64 = !DILocation(line: 17, column: 26, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT:
