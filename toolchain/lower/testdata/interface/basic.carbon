// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interface/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interface/basic.carbon

interface I {
  fn Assoc();
}

// There are no interesting runtime operations here, but there's no rule saying
// you can't pass a facet around at runtime, so make sure it works.
fn F(_: I) {}

fn G() -> I;

fn H() -> I { return G(); }

class C {
  impl as I {
    fn Assoc() {}
  }
}

fn CallF() {
  F(C);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type.10 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(%type.10 %_) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare %type.10 @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define %type.10 @_CH.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.10 @_CG.Main(), !dbg !14
// CHECK:STDOUT:   ret %type.10 %G.call, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CAssoc.C.Main:I.Main"() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(%type.10 zeroinitializer), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 19, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !1, line: 23, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!7}
// CHECK:STDOUT: !14 = !DILocation(line: 23, column: 22, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 15, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Assoc", linkageName: "_CAssoc.C.Main:I.Main", scope: null, file: !1, line: 27, type: !17, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DILocation(line: 27, column: 5, scope: !16)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !1, line: 31, type: !17, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !21 = !DILocation(line: 32, column: 3, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 31, column: 1, scope: !20)
// CHECK:STDOUT:
