// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_different_associated_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_different_associated_const.carbon

// Cannot coalesce the lowering for G specifics, as they have associated contants of different types.
// Check different functions are still emitted when trying to deduplicate emitted definitons.

interface I {
  let T: type;
  // TODO: An interface with multiple associated entities is not yet
  // implemented in mangling.
}

fn G(generic U: I) {
  var _: U.T*;
}

class C {}

impl C as I where .T = bool {
}

class D {}
impl D as I where .T = i32 {
}

fn H() {
  G(C);
  G(D);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_different_associated_const.carbon'
// CHECK:STDOUT: source_filename = "call_different_associated_const.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.0676fa98bb7dbd66(), !dbg !37
// CHECK:STDOUT:   call void @_CG.Main.3e979ea3a37f4bf1(), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.0676fa98bb7dbd66() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca ptr, align 8, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !41
// CHECK:STDOUT:   store ptr poison, ptr %_.var, align 8, !dbg !41
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.3e979ea3a37f4bf1() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca ptr, align 8, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !45
// CHECK:STDOUT:   store ptr poison, ptr %_.var, align 8, !dbg !45
// CHECK:STDOUT:   ret void, !dbg !47
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "call_different_associated_const.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !{null}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !32, line: 35, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 36, column: 3, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 37, column: 3, scope: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 35, column: 1, scope: !36)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.0676fa98bb7dbd66", scope: null, file: !32, line: 22, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !41 = !DILocation(line: 23, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !DILocation(line: 22, column: 1, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.3e979ea3a37f4bf1", scope: null, file: !32, line: 22, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 22, column: 1, scope: !44)
// CHECK:STDOUT:
