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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6d185494fc1773c8(ptr %a.var), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CI.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CJ.Main() #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33_11.1.temp = alloca {}, align 1, !dbg !35
// CHECK:STDOUT:   call void @_CI.Main(), !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_11.1.temp), !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6d185494fc1773c8(ptr %.loc33_11.1.temp), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6d185494fc1773c8(ptr %t) #0 !dbg !37 {
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6d185494fc1773c8, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "no_op.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !14, line: 17, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 17, column: 1, scope: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !14, line: 21, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !23 = !DILocation(line: 22, column: 3, scope: !20)
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{null, !26}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 23, column: 3, scope: !20)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !14, line: 26, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !33 = !DILocation(line: 27, column: 3, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "J", linkageName: "_CJ.Main", scope: null, file: !14, line: 32, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !35 = !DILocation(line: 33, column: 9, scope: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 32, column: 1, scope: !34)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6d185494fc1773c8", scope: null, file: !4, line: 30, type: !28, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !43)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !26)
// CHECK:STDOUT: !42 = !DILocation(line: 30, column: 1, scope: !37)
// CHECK:STDOUT: !43 = !{!38}
// CHECK:STDOUT:
