// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interface/where.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interface/where.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

interface I {
  let T: type;
  fn F();
}

class C {}

impl C as I where .T = () {
  fn F() {
  }
}

// --- no_period_self_dependence.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}
class E {}

interface I(T: type) { fn F(self); }
interface J(T: type) {}

impl D as J(E) {}

// The type of `T` is not `.Self`-dependent, because `.Self` appears after
// `where`.
impl forall [T: type where D impls J(.Self)] C as I(T) {
  fn F(unused self) {}
}

fn F(a: C) {
  a.(I(E).F)();
}

fn G() {
  // This is a concrete, non-`.Self`-dependent type.
  // (Lowering would assert if it sees a dependent constant.)
  let unused t: type = type where D impls J(.Self);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:I.Main"() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.Main", scope: null, file: !14, line: 12, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 12, column: 3, scope: !18)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_period_self_dependence.carbon'
// CHECK:STDOUT: source_filename = "no_period_self_dependence.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %a) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.C.Main:I.5d7343568c0b5715.Main.c8e85d84fc558684"(ptr %a), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.C.Main:I.5d7343568c0b5715.Main.c8e85d84fc558684"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !20 = !DIFile(filename: "no_period_self_dependence.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 19, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !30)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !22)
// CHECK:STDOUT: !28 = !DILocation(line: 20, column: 3, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 19, column: 1, scope: !25)
// CHECK:STDOUT: !30 = !{!26}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 23, type: !17, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !34 = !DILocation(line: 23, column: 1, scope: !31)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.5d7343568c0b5715.Main.c8e85d84fc558684", scope: null, file: !20, line: 16, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !39)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !22)
// CHECK:STDOUT: !38 = !DILocation(line: 16, column: 3, scope: !35)
// CHECK:STDOUT: !39 = !{!36}
// CHECK:STDOUT:
