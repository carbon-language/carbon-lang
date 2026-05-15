// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/access/return_value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/access/return_value_access.carbon

fn F() -> (i32, i32) { return (12, 24); }

fn Run() {
  var _: i32 = F().1;
}

// CHECK:STDOUT: ; ModuleID = 'return_value_access.carbon'
// CHECK:STDOUT: source_filename = "return_value_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.014.loc13_39 = internal constant { i32, i32 } { i32 12, i32 24 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @tuple.014.loc13_39, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   %.loc16_18.1.temp = alloca { i32, i32 }, align 8, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_18.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_18.1.temp), !dbg !15
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc16_18.1.temp, i32 0, i32 1, !dbg !15
// CHECK:STDOUT:   %.loc16_19 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   store i32 %.loc16_19, ptr %_.var, align 4, !dbg !14
// CHECK:STDOUT:   call void @"_COp.8e354c83fe88ab86:core.Destroy.Core"(ptr %.loc16_18.1.temp), !dbg !15
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %_.var), !dbg !14
// CHECK:STDOUT:   ret i32 0, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8e354c83fe88ab86:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
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
// CHECK:STDOUT: !3 = !DIFile(filename: "return_value_access.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !DILocation(line: 13, column: 31, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 24, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 15, type: !11, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !14 = !DILocation(line: 16, column: 3, scope: !10)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 16, scope: !10)
// CHECK:STDOUT: !16 = !DILocation(line: 15, column: 1, scope: !10)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 16, type: !18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !13}
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !17, type: !13)
// CHECK:STDOUT: !22 = !DILocation(line: 16, column: 16, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8e354c83fe88ab86:core.Destroy.Core", scope: null, file: !3, line: 16, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !25 = !{null, !7}
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !28 = !DILocation(line: 16, column: 16, scope: !23)
