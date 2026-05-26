// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/packages/imported_package_mangle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/packages/imported_package_mangle.carbon

// --- a.carbon
package A;

class C {}

interface I {
  fn F() {}
}

impl array(C, 1) as I {
  fn F() {}
}

impl forall [N:! Core.IntLiteral()] array(C, N) as I {
  fn F() {}
}

// --- b.carbon
package B;
import A;

fn G() {
  // This causes a declaration of `A._CF` which should have the same name
  // mangling across both files.
  (array(A.C, 1) as A.I).F();

  // Same thing with a generic impl.
  (array(A.C, 2) as A.I).F();
}

// --- c.carbon

package C;
import A;

class C {}

interface I {
  fn F() {}
}

impl array(C, 1) as I {
  fn F() {}
}

impl forall [N:! Core.IntLiteral()] array(C, N) as I {
  fn F() {}
}

impl array(C, 1) as A.I {
  fn F() {}
}

impl forall [N:! Core.IntLiteral()] array(C, N) as A.I {
  fn F() {}
}

impl array(A.C, 1) as I {
  fn F() {}
}

impl forall [N:! Core.IntLiteral()] array(A.C, N) as I {
  fn F() {}
}

fn G() {
  // These should have four different manglings.
  (array(A.C, 1) as A.I).F();
  (array(C, 1) as A.I).F();
  (array(A.C, 1) as I).F();
  (array(C, 1) as I).F();

  // Likewise for a generic impl.
  (array(A.C, 2) as A.I).F();
  (array(C, 2) as A.I).F();
  (array(A.C, 2) as I).F();
  (array(C, 2) as I).F();
}

// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.cab92310f18b02bb:I.A"() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.cab92310f18b02bb:I.A", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.B() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.cab92310f18b02bb:I.A"(), !dbg !7
// CHECK:STDOUT:   call void @"_CF.6fd31dc02304be88:I.A.91a8a89714fbc21c"(), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.cab92310f18b02bb:I.A"()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.6fd31dc02304be88:I.A.91a8a89714fbc21c"() #0 !dbg !10 {
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.B", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F", linkageName: "_CF.6fd31dc02304be88:I.A.91a8a89714fbc21c", scope: null, file: !11, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 3, scope: !10)
// CHECK:STDOUT: ; ModuleID = 'c.carbon'
// CHECK:STDOUT: source_filename = "c.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.98d334a081741bc3:I.C"() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.98d334a081741bc3:I.A"() #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.cab92310f18b02bb:I.C"() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.C() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.cab92310f18b02bb:I.A"(), !dbg !13
// CHECK:STDOUT:   call void @"_CF.98d334a081741bc3:I.A"(), !dbg !14
// CHECK:STDOUT:   call void @"_CF.cab92310f18b02bb:I.C"(), !dbg !15
// CHECK:STDOUT:   call void @"_CF.98d334a081741bc3:I.C"(), !dbg !16
// CHECK:STDOUT:   call void @"_CF.6fd31dc02304be88:I.A.91a8a89714fbc21c"(), !dbg !17
// CHECK:STDOUT:   call void @"_CF.f0ae628b73c908db:I.A.91a8a89714fbc21c"(), !dbg !18
// CHECK:STDOUT:   call void @"_CF.6fd31dc02304be88:I.C.91a8a89714fbc21c"(), !dbg !19
// CHECK:STDOUT:   call void @"_CF.f0ae628b73c908db:I.C.91a8a89714fbc21c"(), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.cab92310f18b02bb:I.A"()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.6fd31dc02304be88:I.A.91a8a89714fbc21c"() #0 !dbg !22 {
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.f0ae628b73c908db:I.A.91a8a89714fbc21c"() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.6fd31dc02304be88:I.C.91a8a89714fbc21c"() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.f0ae628b73c908db:I.C.91a8a89714fbc21c"() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "c.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.98d334a081741bc3:I.C", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "F", linkageName: "_CF.98d334a081741bc3:I.A", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 3, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F", linkageName: "_CF.cab92310f18b02bb:I.C", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 28, column: 3, scope: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "G", linkageName: "_CG.C", scope: null, file: !3, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 37, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 38, column: 3, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 39, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 40, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 43, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 44, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 45, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 46, column: 3, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 35, column: 1, scope: !12)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF.6fd31dc02304be88:I.A.91a8a89714fbc21c", scope: null, file: !23, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !24 = !DILocation(line: 14, column: 3, scope: !22)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.f0ae628b73c908db:I.A.91a8a89714fbc21c", scope: null, file: !3, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !26 = !DILocation(line: 24, column: 3, scope: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.6fd31dc02304be88:I.C.91a8a89714fbc21c", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 3, scope: !27)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "F", linkageName: "_CF.f0ae628b73c908db:I.C.91a8a89714fbc21c", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 3, scope: !29)
