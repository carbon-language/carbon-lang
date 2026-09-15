// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/no_op.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/no_op.carbon

fn NoOp() = "no_op";

fn NoOp2(x: ()) -> () = "no_op";

fn F() {
  NoOp();
}

fn G() -> () {
  var a: () = NoOp();
  return a;
}

fn H() -> () {
  return NoOp2(());
}

fn I();

fn J() {
  NoOp2(I());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_op.carbon'
// CHECK:STDOUT: source_filename = "no_op.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CI.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CJ.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33_11.1.temp = alloca {}, align 1, !dbg !24
// CHECK:STDOUT:   call void @_CI.Main(), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_11.1.temp), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!7}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "no_op.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !8 = !{null}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 17, type: !9, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 1, scope: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !6, line: 21, type: !9, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 22, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 23, column: 3, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !6, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 27, column: 3, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "J", linkageName: "_CJ.Main", scope: null, file: !6, line: 32, type: !9, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !24 = !DILocation(line: 33, column: 9, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 32, column: 1, scope: !23)
// CHECK:STDOUT:
