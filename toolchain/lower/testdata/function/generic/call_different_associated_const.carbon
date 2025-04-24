// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_different_associated_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_different_associated_const.carbon

// Cannot coalesce the lowering for G specifics, as they have associated contants of different types.
// Check different functions are still emitted when trying to deduplicate emitted definitons.

interface I {
  let T:! type;
  // TODO: An interface with multiple associated entities is not yet
  // implemented in mangling.
}

fn G(U:! I) {
  var x: U.T;
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


// CHECK:STDOUT: ; ModuleID = 'call_different_associated_const.carbon'
// CHECK:STDOUT: source_filename = "call_different_associated_const.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CH.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.47c020a542e13d96(), !dbg !7
// CHECK:STDOUT:   call void @_CG.Main.a10ac7cbfa858009(), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main.47c020a542e13d96() !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i1, align 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 1, ptr %x.var), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main.a10ac7cbfa858009() !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %x.var), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_different_associated_const.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 34, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 35, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 33, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.47c020a542e13d96", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 21, column: 3, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 20, column: 1, scope: !10)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.a10ac7cbfa858009", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !14 = !DILocation(line: 21, column: 3, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 20, column: 1, scope: !13)
