// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/assignment.carbon

fn F() -> i32 {
  var a: i32 = 12;
  a = 9;
  return a;
}

fn G() -> (i32, i32) {
  var b: (i32, i32);
  b = (1, 2);
  return b;
}

// CHECK:STDOUT: ; ModuleID = 'assignment.carbon'
// CHECK:STDOUT: source_filename = "assignment.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.cd5.loc21_5 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !8
// CHECK:STDOUT:   store i32 12, ptr %a.var, align 4, !dbg !8
// CHECK:STDOUT:   store i32 9, ptr %a.var, align 4, !dbg !9
// CHECK:STDOUT:   %.loc16 = load i32, ptr %a.var, align 4, !dbg !10
// CHECK:STDOUT:   call void @"_COp.1fa36cd3e044ac37:core.Destroy.Core"(ptr %a.var), !dbg !8
// CHECK:STDOUT:   ret i32 %.loc16, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1fa36cd3e044ac37:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 4, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !22
// CHECK:STDOUT:   %tuple.elem0.loc21.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !23
// CHECK:STDOUT:   %tuple.elem1.loc21.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !23
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.cd5.loc21_5, i64 8, i1 false), !dbg !24
// CHECK:STDOUT:   %tuple.elem0.loc22_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !25
// CHECK:STDOUT:   %.loc22_10.1 = load i32, ptr %tuple.elem0.loc22_10.1.tuple.elem, align 4, !dbg !25
// CHECK:STDOUT:   %tuple.elem0.loc22_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !25
// CHECK:STDOUT:   store i32 %.loc22_10.1, ptr %tuple.elem0.loc22_10.2.tuple.elem, align 4, !dbg !25
// CHECK:STDOUT:   %tuple.elem1.loc22_10.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !25
// CHECK:STDOUT:   %.loc22_10.3 = load i32, ptr %tuple.elem1.loc22_10.1.tuple.elem, align 4, !dbg !25
// CHECK:STDOUT:   %tuple.elem1.loc22_10.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !25
// CHECK:STDOUT:   store i32 %.loc22_10.3, ptr %tuple.elem1.loc22_10.2.tuple.elem, align 4, !dbg !25
// CHECK:STDOUT:   call void @"_COp.11d8dd9278293aa3:core.Destroy.Core"(ptr %b.var), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.11d8dd9278293aa3:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "assignment.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 16, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1fa36cd3e044ac37:core.Destroy.Core", scope: null, file: !3, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !7}
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 14, column: 3, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 19, type: !19, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 3, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 21, column: 7, scope: !18)
// CHECK:STDOUT: !24 = !DILocation(line: 21, column: 3, scope: !18)
// CHECK:STDOUT: !25 = !DILocation(line: 22, column: 10, scope: !18)
// CHECK:STDOUT: !26 = !DILocation(line: 22, column: 3, scope: !18)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.11d8dd9278293aa3:core.Destroy.Core", scope: null, file: !3, line: 20, type: !28, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !21}
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !27, type: !21)
// CHECK:STDOUT: !32 = !DILocation(line: 20, column: 3, scope: !27)
