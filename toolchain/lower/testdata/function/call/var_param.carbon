// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/var_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/var_param.carbon

fn DoNothing(_: i32) {}

fn Main() {
  var a: i32 = 0;
  DoNothing(a);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_param.carbon'
// CHECK:STDOUT: source_filename = "var_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDoNothing.Main(i32 %_) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !33
// CHECK:STDOUT:   store i32 0, ptr %a.var, align 4, !dbg !33
// CHECK:STDOUT:   %.loc17 = load i32, ptr %a.var, align 4, !dbg !35
// CHECK:STDOUT:   call void @_CDoNothing.Main(i32 %.loc17), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "var_param.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "DoNothing", linkageName: "_CDoNothing.Main", scope: null, file: !20, line: 13, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !22)
// CHECK:STDOUT: !30 = !DILocation(line: 13, column: 1, scope: !27)
// CHECK:STDOUT: !31 = !{!28}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !20, line: 15, type: !26, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 16, column: 3, scope: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 17, column: 13, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 3, scope: !32)
// CHECK:STDOUT: !37 = !DILocation(line: 15, column: 1, scope: !32)
// CHECK:STDOUT:
