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
// CHECK:STDOUT: define i32 @main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !26
// CHECK:STDOUT:   %b.var = alloca i32, align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !26
// CHECK:STDOUT:   store i32 1, ptr %a.var, align 4, !dbg !26
// CHECK:STDOUT:   br label %while.cond, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %while.body, %entry
// CHECK:STDOUT:   br i1 true, label %while.body, label %while.done, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !29
// CHECK:STDOUT:   %.loc16_18 = load i32, ptr %a.var, align 4, !dbg !30
// CHECK:STDOUT:   store i32 %.loc16_18, ptr %b.var, align 4, !dbg !29
// CHECK:STDOUT:   %.loc17 = load i32, ptr %b.var, align 4, !dbg !33
// CHECK:STDOUT:   store i32 %.loc17, ptr %a.var, align 4, !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %b.var), !dbg !29
// CHECK:STDOUT:   br label %while.cond, !dbg !36
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   %.loc19 = load i32, ptr %a.var, align 4, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %a.var), !dbg !26
// CHECK:STDOUT:   ret i32 %.loc19, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !39 {
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "nested.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 14, column: 3, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 15, column: 9, scope: !23)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 5, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 18, scope: !23)
// CHECK:STDOUT: !32 = !DILocation(line: 17, column: 5, scope: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 9, scope: !23)
// CHECK:STDOUT: !34 = !{null, !20}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 15, column: 3, scope: !23)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 10, scope: !23)
// CHECK:STDOUT: !38 = !DILocation(line: 19, column: 3, scope: !23)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !35, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !20)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 1, scope: !39)
// CHECK:STDOUT: !45 = !{!40}
// CHECK:STDOUT:
