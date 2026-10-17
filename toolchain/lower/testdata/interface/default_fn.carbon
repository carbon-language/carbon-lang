// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interface/default_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interface/default_fn.carbon

// --- use_self.carbon

interface I {
  fn G();
  default fn F() {
    Self.G();
  }
}

interface J {
  fn H();
}

impl forall [T: J] T as I {
  fn G() { T.H(); }
}

fn Success();

impl () as J {
  fn H() {
    // Should be called by Test below.
    Success();
  }
}

fn CallF[T: I](x: T) {
  x.F();
}

fn Test() {
  CallF(());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_self.carbon'
// CHECK:STDOUT: source_filename = "use_self.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CSuccess.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CH.61ea2aba74ab3bf1:J.Main"() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSuccess.Main(), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCallF.Main.6ad5e082e471d985(), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallF.Main.6ad5e082e471d985() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.I.Main.6ad5e082e471d985(), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.I.Main.6ad5e082e471d985() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CG.d90909b043297e37:I.Main.a9d0daf36987b80b"(), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CG.d90909b043297e37:I.Main.a9d0daf36987b80b"() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CH.61ea2aba74ab3bf1:J.Main"(), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "use_self.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "H", linkageName: "_CH.61ea2aba74ab3bf1:J.Main", scope: null, file: !16, line: 20, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 22, column: 5, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 3, scope: !20)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !16, line: 30, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 31, column: 3, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 30, column: 1, scope: !23)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main.6ad5e082e471d985", scope: null, file: !16, line: 26, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !28 = !DILocation(line: 27, column: 3, scope: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 26, column: 1, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "F", linkageName: "_CF.I.Main.6ad5e082e471d985", scope: null, file: !16, line: 4, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !31 = !DILocation(line: 5, column: 5, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 4, column: 3, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "G", linkageName: "_CG.d90909b043297e37:I.Main.a9d0daf36987b80b", scope: null, file: !16, line: 14, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !34 = !DILocation(line: 14, column: 12, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 14, column: 3, scope: !33)
// CHECK:STDOUT:
