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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !21
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %a.var), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CI.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CJ.Main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33_11.1.temp = alloca {}, align 1, !dbg !33
// CHECK:STDOUT:   call void @_CI.Main(), !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_11.1.temp), !dbg !33
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %.loc33_11.1.temp), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9045abb8230cbcdf(ptr %t) #0 !dbg !35 {
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.9045abb8230cbcdf, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!13}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !12 = !DIFile(filename: "no_op.carbon", directory: "")
// CHECK:STDOUT: !13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !12, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !12, line: 17, type: !15, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 1, scope: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !12, line: 21, type: !15, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 22, column: 3, scope: !18)
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 23, column: 3, scope: !18)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !12, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !31 = !DILocation(line: 27, column: 3, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "J", linkageName: "_CJ.Main", scope: null, file: !12, line: 32, type: !15, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !33 = !DILocation(line: 33, column: 9, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 32, column: 1, scope: !32)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9045abb8230cbcdf", scope: null, file: !4, line: 32, type: !26, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !41)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !24)
// CHECK:STDOUT: !40 = !DILocation(line: 32, column: 1, scope: !35)
// CHECK:STDOUT: !41 = !{!36}
// CHECK:STDOUT:
