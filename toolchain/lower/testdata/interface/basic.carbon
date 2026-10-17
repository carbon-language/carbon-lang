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
// CHECK:STDOUT: %type.12 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(%type.12 %_) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare %type.12 @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define %type.12 @_CH.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call %type.12 @_CG.Main(), !dbg !33
// CHECK:STDOUT:   ret %type.12 %G.call, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CAssoc.C.Main:I.Main"() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallF.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(%type.12 zeroinitializer), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{null, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{!18}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !16, line: 19, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !18)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 1, scope: !25)
// CHECK:STDOUT: !29 = !{!26}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !16, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !33 = !DILocation(line: 23, column: 22, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 23, column: 15, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Assoc", linkageName: "_CAssoc.C.Main:I.Main", scope: null, file: !16, line: 27, type: !24, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !36 = !DILocation(line: 27, column: 5, scope: !35)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "CallF", linkageName: "_CCallF.Main", scope: null, file: !16, line: 31, type: !24, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !38 = !DILocation(line: 32, column: 3, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 31, column: 1, scope: !37)
// CHECK:STDOUT:
