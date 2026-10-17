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
// CHECK:STDOUT: define void @"_CF.C.Main:I.Main"() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
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
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.Main", scope: null, file: !16, line: 12, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 12, column: 3, scope: !20)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_period_self_dependence.carbon'
// CHECK:STDOUT: source_filename = "no_period_self_dependence.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %a) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.C.Main:I.5d7343568c0b5715.Main.33da5f7a783feb7b"(ptr %a), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.C.Main:I.5d7343568c0b5715.Main.33da5f7a783feb7b"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !22 = !DIFile(filename: "no_period_self_dependence.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !22, line: 19, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !32)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 20, column: 3, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 19, column: 1, scope: !27)
// CHECK:STDOUT: !32 = !{!28}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !22, line: 23, type: !19, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !36 = !DILocation(line: 23, column: 1, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.5d7343568c0b5715.Main.33da5f7a783feb7b", scope: null, file: !22, line: 16, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !41)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !24)
// CHECK:STDOUT: !40 = !DILocation(line: 16, column: 3, scope: !37)
// CHECK:STDOUT: !41 = !{!38}
// CHECK:STDOUT:
