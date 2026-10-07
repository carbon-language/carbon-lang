// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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

impl forall [N: Core.IntLiteral] array(C, N) as I {
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

impl forall [N: Core.IntLiteral] array(C, N) as I {
  fn F() {}
}

impl array(C, 1) as A.I {
  fn F() {}
}

impl forall [N: Core.IntLiteral] array(C, N) as A.I {
  fn F() {}
}

impl array(A.C, 1) as I {
  fn F() {}
}

impl forall [N: Core.IntLiteral] array(A.C, N) as I {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.Array.c34a6de08fb29ead.Core:I.A"() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !{null}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.c34a6de08fb29ead.Core:I.A", scope: null, file: !34, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 10, column: 3, scope: !38)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.B() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.Array.c34a6de08fb29ead.Core:I.A"(), !dbg !43
// CHECK:STDOUT:   call void @"_CF.Array.4bcaa430271e27dc.Core:I.A.91a8a89714fbc21c"(), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.Array.c34a6de08fb29ead.Core:I.A"()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.Array.4bcaa430271e27dc.Core:I.A.91a8a89714fbc21c"() #0 !dbg !46 {
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !36 = !{null}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !40 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "G", linkageName: "_CG.B", scope: null, file: !40, line: 4, type: !37, spFlags: DISPFlagDefinition, unit: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 7, column: 3, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 10, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !DILocation(line: 4, column: 1, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.4bcaa430271e27dc.Core:I.A.91a8a89714fbc21c", scope: null, file: !34, line: 14, type: !37, spFlags: DISPFlagDefinition, unit: !41)
// CHECK:STDOUT: !47 = !DILocation(line: 14, column: 3, scope: !46)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'c.carbon'
// CHECK:STDOUT: source_filename = "c.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.Array.64dce905b462d459.Core:I.C"() #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.Array.64dce905b462d459.Core:I.A"() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.Array.c34a6de08fb29ead.Core:I.C"() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.C() #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.Array.c34a6de08fb29ead.Core:I.A"(), !dbg !57
// CHECK:STDOUT:   call void @"_CF.Array.64dce905b462d459.Core:I.A"(), !dbg !58
// CHECK:STDOUT:   call void @"_CF.Array.c34a6de08fb29ead.Core:I.C"(), !dbg !59
// CHECK:STDOUT:   call void @"_CF.Array.64dce905b462d459.Core:I.C"(), !dbg !60
// CHECK:STDOUT:   call void @"_CF.Array.4bcaa430271e27dc.Core:I.A.91a8a89714fbc21c"(), !dbg !61
// CHECK:STDOUT:   call void @"_CF.Array.3fc281a4708ed92c.Core:I.A.91a8a89714fbc21c"(), !dbg !62
// CHECK:STDOUT:   call void @"_CF.Array.4bcaa430271e27dc.Core:I.C.91a8a89714fbc21c"(), !dbg !63
// CHECK:STDOUT:   call void @"_CF.Array.3fc281a4708ed92c.Core:I.C.91a8a89714fbc21c"(), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.Array.c34a6de08fb29ead.Core:I.A"()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.Array.4bcaa430271e27dc.Core:I.A.91a8a89714fbc21c"() #0 !dbg !66 {
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.Array.3fc281a4708ed92c.Core:I.A.91a8a89714fbc21c"() #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.Array.4bcaa430271e27dc.Core:I.C.91a8a89714fbc21c"() #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.Array.3fc281a4708ed92c.Core:I.C.91a8a89714fbc21c"() #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!49}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !36 = !{null}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !48 = !DIFile(filename: "c.carbon", directory: "")
// CHECK:STDOUT: !49 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !48, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.64dce905b462d459.Core:I.C", scope: null, file: !48, line: 12, type: !37, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 12, column: 3, scope: !50)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.64dce905b462d459.Core:I.A", scope: null, file: !48, line: 20, type: !37, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 20, column: 3, scope: !52)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.c34a6de08fb29ead.Core:I.C", scope: null, file: !48, line: 28, type: !37, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !55 = !DILocation(line: 28, column: 3, scope: !54)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "G", linkageName: "_CG.C", scope: null, file: !48, line: 35, type: !37, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !57 = !DILocation(line: 37, column: 3, scope: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 38, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !DILocation(line: 39, column: 3, scope: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 40, column: 3, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 43, column: 3, scope: !56)
// CHECK:STDOUT: !62 = !DILocation(line: 44, column: 3, scope: !56)
// CHECK:STDOUT: !63 = !DILocation(line: 45, column: 3, scope: !56)
// CHECK:STDOUT: !64 = !DILocation(line: 46, column: 3, scope: !56)
// CHECK:STDOUT: !65 = !DILocation(line: 35, column: 1, scope: !56)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.4bcaa430271e27dc.Core:I.A.91a8a89714fbc21c", scope: null, file: !34, line: 14, type: !37, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !67 = !DILocation(line: 14, column: 3, scope: !66)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.3fc281a4708ed92c.Core:I.A.91a8a89714fbc21c", scope: null, file: !48, line: 24, type: !37, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !69 = !DILocation(line: 24, column: 3, scope: !68)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.4bcaa430271e27dc.Core:I.C.91a8a89714fbc21c", scope: null, file: !48, line: 32, type: !37, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !71 = !DILocation(line: 32, column: 3, scope: !70)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "F", linkageName: "_CF.Array.3fc281a4708ed92c.Core:I.C.91a8a89714fbc21c", scope: null, file: !48, line: 16, type: !37, spFlags: DISPFlagDefinition, unit: !49)
// CHECK:STDOUT: !73 = !DILocation(line: 16, column: 3, scope: !72)
// CHECK:STDOUT:
