// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/char.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/char.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

fn Example() -> char {
  var c: char = 'a';
  return c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CExample.Main() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca i8, align 1, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !12
// CHECK:STDOUT:   store i8 97, ptr %c.var, align 1, !dbg !12
// CHECK:STDOUT:   %.loc6 = load i8, ptr %c.var, align 1, !dbg !14
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.00174ca91d483df7(ptr %c.var), !dbg !12
// CHECK:STDOUT:   ret i8 %.loc6, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.00174ca91d483df7(ptr %t) #0 !dbg !19 {
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !{!6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Example", linkageName: "_CExample.Main", scope: null, file: !4, line: 4, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 3, scope: !9)
// CHECK:STDOUT: !14 = !DILocation(line: 6, column: 10, scope: !9)
// CHECK:STDOUT: !15 = !{null, !6}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 6, column: 3, scope: !9)
// CHECK:STDOUT: !18 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.00174ca91d483df7", scope: null, file: !18, line: 35, type: !16, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !6)
// CHECK:STDOUT: !24 = !DILocation(line: 35, column: 1, scope: !19)
// CHECK:STDOUT: !25 = !{!20}
// CHECK:STDOUT:
