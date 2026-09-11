// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/nested.carbon

fn Run() -> i32 {
  var a: i32 = 1;
  while (true) {
    var b: i32 = a;
    a = b;
  }
  return a;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'nested.carbon'
// CHECK:STDOUT: source_filename = "nested.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %b.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !8
// CHECK:STDOUT:   store i32 1, ptr %a.var, align 4, !dbg !8
// CHECK:STDOUT:   br label %while.cond, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %while.body, %entry
// CHECK:STDOUT:   br i1 true, label %while.body, label %while.done, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !9
// CHECK:STDOUT:   %.loc16_18 = load i32, ptr %a.var, align 4, !dbg !11
// CHECK:STDOUT:   store i32 %.loc16_18, ptr %b.var, align 4, !dbg !9
// CHECK:STDOUT:   %.loc17 = load i32, ptr %b.var, align 4, !dbg !12
// CHECK:STDOUT:   store i32 %.loc17, ptr %a.var, align 4, !dbg !13
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %b.var), !dbg !9
// CHECK:STDOUT:   br label %while.cond, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   %.loc19 = load i32, ptr %a.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %a.var), !dbg !8
// CHECK:STDOUT:   ret i32 %.loc19, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self), !dbg !27
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.6580e8d2c4d79a48(ptr %self), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSubobjectDestroy.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "nested.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 9, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 16, column: 18, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 17, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 17, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 19, column: 10, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 16, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !7}
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 16, column: 5, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !24, line: 24, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !24 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 25, column: 5, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 5, scope: !23)
// CHECK:STDOUT: !29 = !DILocation(line: 24, column: 3, scope: !23)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !24, line: 17, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !7)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 3, scope: !30)
// CHECK:STDOUT:
