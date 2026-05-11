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

// CHECK:STDOUT: ; ModuleID = 'array_element_access.carbon'
// CHECK:STDOUT: source_filename = "array_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.21c.loc12_37 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @array.loc14_40 = internal constant [2 x i32] [i32 1, i32 2]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CA.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.21c.loc12_37, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CB.Main(ptr sret([2 x i32]) %return) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_39.3.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 0, !dbg !11
// CHECK:STDOUT:   %.loc14_39.6.array.index = getelementptr inbounds [2 x i32], ptr %return, i32 0, i64 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @array.loc14_40, i64 8, i1 false), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [2 x i32], align 4, !dbg !17
// CHECK:STDOUT:   %.loc17_28.1.temp = alloca { i32, i32 }, align 8, !dbg !18
// CHECK:STDOUT:   %b.var = alloca i32, align 4, !dbg !19
// CHECK:STDOUT:   %_.var.loc19 = alloca i32, align 4, !dbg !20
// CHECK:STDOUT:   %_.var.loc20 = alloca i32, align 4, !dbg !21
// CHECK:STDOUT:   %.loc20_18.1.temp = alloca [2 x i32], align 4, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_28.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_CA.Main(ptr %.loc17_28.1.temp), !dbg !18
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc17_28.1.temp, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   %.loc17_28.3 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !18
// CHECK:STDOUT:   %.loc17_28.4.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 0, !dbg !18
// CHECK:STDOUT:   store i32 %.loc17_28.3, ptr %.loc17_28.4.array.index, align 4, !dbg !18
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc17_28.1.temp, i32 0, i32 1, !dbg !18
// CHECK:STDOUT:   %.loc17_28.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !18
// CHECK:STDOUT:   %.loc17_28.7.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i64 1, !dbg !18
// CHECK:STDOUT:   store i32 %.loc17_28.6, ptr %.loc17_28.7.array.index, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !19
// CHECK:STDOUT:   store i32 1, ptr %b.var, align 4, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc19), !dbg !20
// CHECK:STDOUT:   %.loc19_18 = load i32, ptr %b.var, align 4, !dbg !23
// CHECK:STDOUT:   %.loc19_19.1.array.index = getelementptr inbounds [2 x i32], ptr %a.var, i32 0, i32 %.loc19_18, !dbg !24
// CHECK:STDOUT:   %.loc19_19.2 = load i32, ptr %.loc19_19.1.array.index, align 4, !dbg !24
// CHECK:STDOUT:   store i32 %.loc19_19.2, ptr %_.var.loc19, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc20), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_18.1.temp), !dbg !22
// CHECK:STDOUT:   call void @_CB.Main(ptr %.loc20_18.1.temp), !dbg !22
// CHECK:STDOUT:   %.loc20_21.1.array.index = getelementptr inbounds [2 x i32], ptr %.loc20_18.1.temp, i32 0, i32 1, !dbg !22
// CHECK:STDOUT:   %.loc20_21.2 = load i32, ptr %.loc20_21.1.array.index, align 4, !dbg !22
// CHECK:STDOUT:   store i32 %.loc20_21.2, ptr %_.var.loc20, align 4, !dbg !21
// CHECK:STDOUT:   call void @"_COp.f53f00a5f9e218d2:core.Destroy.Core"(ptr %.loc20_18.1.temp), !dbg !22
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %_.var.loc20), !dbg !21
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %_.var.loc19), !dbg !20
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %b.var), !dbg !19
// CHECK:STDOUT:   call void @"_COp.8e354c83fe88ab86:core.Destroy.Core"(ptr %.loc17_28.1.temp), !dbg !18
// CHECK:STDOUT:   call void @"_COp.f53f00a5f9e218d2:core.Destroy.Core"(ptr %a.var), !dbg !17
// CHECK:STDOUT:   ret i32 0, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f53f00a5f9e218d2:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8e354c83fe88ab86:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.7e389eab4a7e5487:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.f53f00a5f9e218d2:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "array_element_access.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 31, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 12, column: 24, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 34, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 27, scope: !10)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 16, type: !14, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 3, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 26, scope: !13)
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 3, scope: !13)
// CHECK:STDOUT: !20 = !DILocation(line: 19, column: 3, scope: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 20, column: 3, scope: !13)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 16, scope: !13)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 18, scope: !13)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 16, scope: !13)
// CHECK:STDOUT: !25 = !DILocation(line: 16, column: 1, scope: !13)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 20, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{null, !16}
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !26, type: !16)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 16, scope: !26)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f53f00a5f9e218d2:core.Destroy.Core", scope: null, file: !3, line: 20, type: !33, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !34 = !{null, !7}
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 16, scope: !32)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8e354c83fe88ab86:core.Destroy.Core", scope: null, file: !3, line: 17, type: !33, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 26, scope: !38)
