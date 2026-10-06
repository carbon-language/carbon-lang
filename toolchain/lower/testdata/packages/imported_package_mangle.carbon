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
// CHECK:STDOUT: define void @"_CF.f327915d949d024f:I.A"() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF.f327915d949d024f:I.A", scope: null, file: !18, line: 10, type: !21, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 10, column: 3, scope: !22)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.B() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.f327915d949d024f:I.A"(), !dbg !27
// CHECK:STDOUT:   call void @"_CF.b11ad1d729c16944:I.A.91a8a89714fbc21c"(), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.f327915d949d024f:I.A"()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.b11ad1d729c16944:I.A.91a8a89714fbc21c"() #0 !dbg !30 {
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!25}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !24 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !25 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !24, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "G", linkageName: "_CG.B", scope: null, file: !24, line: 4, type: !21, spFlags: DISPFlagDefinition, unit: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 7, column: 3, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 10, column: 3, scope: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 4, column: 1, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "F", linkageName: "_CF.b11ad1d729c16944:I.A.91a8a89714fbc21c", scope: null, file: !18, line: 14, type: !21, spFlags: DISPFlagDefinition, unit: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 3, scope: !30)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'c.carbon'
// CHECK:STDOUT: source_filename = "c.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.f6938aeb3cf77e6f:I.C"() #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.f6938aeb3cf77e6f:I.A"() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.f327915d949d024f:I.C"() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.C() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.f327915d949d024f:I.A"(), !dbg !41
// CHECK:STDOUT:   call void @"_CF.f6938aeb3cf77e6f:I.A"(), !dbg !42
// CHECK:STDOUT:   call void @"_CF.f327915d949d024f:I.C"(), !dbg !43
// CHECK:STDOUT:   call void @"_CF.f6938aeb3cf77e6f:I.C"(), !dbg !44
// CHECK:STDOUT:   call void @"_CF.b11ad1d729c16944:I.A.91a8a89714fbc21c"(), !dbg !45
// CHECK:STDOUT:   call void @"_CF.c2375ae0b92fd5ec:I.A.91a8a89714fbc21c"(), !dbg !46
// CHECK:STDOUT:   call void @"_CF.b11ad1d729c16944:I.C.91a8a89714fbc21c"(), !dbg !47
// CHECK:STDOUT:   call void @"_CF.c2375ae0b92fd5ec:I.C.91a8a89714fbc21c"(), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.f327915d949d024f:I.A"()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.b11ad1d729c16944:I.A.91a8a89714fbc21c"() #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.c2375ae0b92fd5ec:I.A.91a8a89714fbc21c"() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.b11ad1d729c16944:I.C.91a8a89714fbc21c"() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CF.c2375ae0b92fd5ec:I.C.91a8a89714fbc21c"() #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !32 = !DIFile(filename: "c.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "F", linkageName: "_CF.f6938aeb3cf77e6f:I.C", scope: null, file: !32, line: 12, type: !21, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 12, column: 3, scope: !34)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "F", linkageName: "_CF.f6938aeb3cf77e6f:I.A", scope: null, file: !32, line: 20, type: !21, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 3, scope: !36)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "F", linkageName: "_CF.f327915d949d024f:I.C", scope: null, file: !32, line: 28, type: !21, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !39 = !DILocation(line: 28, column: 3, scope: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "G", linkageName: "_CG.C", scope: null, file: !32, line: 35, type: !21, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !41 = !DILocation(line: 37, column: 3, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 38, column: 3, scope: !40)
// CHECK:STDOUT: !43 = !DILocation(line: 39, column: 3, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 40, column: 3, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 43, column: 3, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 44, column: 3, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 45, column: 3, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 46, column: 3, scope: !40)
// CHECK:STDOUT: !49 = !DILocation(line: 35, column: 1, scope: !40)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "F", linkageName: "_CF.b11ad1d729c16944:I.A.91a8a89714fbc21c", scope: null, file: !18, line: 14, type: !21, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !51 = !DILocation(line: 14, column: 3, scope: !50)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "F", linkageName: "_CF.c2375ae0b92fd5ec:I.A.91a8a89714fbc21c", scope: null, file: !32, line: 24, type: !21, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !53 = !DILocation(line: 24, column: 3, scope: !52)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "F", linkageName: "_CF.b11ad1d729c16944:I.C.91a8a89714fbc21c", scope: null, file: !32, line: 32, type: !21, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !55 = !DILocation(line: 32, column: 3, scope: !54)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "F", linkageName: "_CF.c2375ae0b92fd5ec:I.C.91a8a89714fbc21c", scope: null, file: !32, line: 16, type: !21, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !57 = !DILocation(line: 16, column: 3, scope: !56)
// CHECK:STDOUT:
