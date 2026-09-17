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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %a.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CI.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CJ.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33_11.1.temp = alloca {}, align 1, !dbg !27
// CHECK:STDOUT:   call void @_CI.Main(), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_11.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %.loc33_11.1.temp), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.c7a99711ca3557ac, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!7}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !6 = !DIFile(filename: "no_op.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !8 = !{null}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 17, type: !9, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 1, scope: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !6, line: 21, type: !9, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 22, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{null, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 3, scope: !12)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !6, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 3, scope: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "J", linkageName: "_CJ.Main", scope: null, file: !6, line: 32, type: !9, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 33, column: 9, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 1, scope: !26)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c7a99711ca3557ac", scope: null, file: !4, line: 24, type: !20, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !35)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !18)
// CHECK:STDOUT: !34 = !DILocation(line: 24, column: 3, scope: !29)
// CHECK:STDOUT: !35 = !{!30}
// CHECK:STDOUT:
