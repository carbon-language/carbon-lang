// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// EXTRA-ARGS: --mangle-string-fingerprint
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/packages/imported_package_mangle_string.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/packages/imported_package_mangle_string.carbon

// --- a.carbon
package A;

class C {}

interface I {
  fn F() {}
}

impl array(C, 1) as I {
  fn F() {}
}

namespace N;
class N.D;

impl array(N.D, 1) as I {
  fn F() {}
}

// --- b.carbon
package B;
import A;

fn G() {
  (array(A.C, 1) as A.I).F();
  (array(A.N.D, 1) as A.I).F();
}

// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.{array_type,{int_value,{Core.IntLiteral},1},{class_type,C,<package>,A,!invalid,!invalid}}:I.A"() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.{array_type,{int_value,{Core.IntLiteral},1},{class_type,D,N,<package>,A,!invalid,!invalid}}:I.A"() #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.{array_type,{int_value,{Core.IntLiteral},1},{class_type,C,<package>,A,!invalid,!invalid}}:I.A", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "F", linkageName: "_CF.{array_type,{int_value,{Core.IntLiteral},1},{class_type,D,N,<package>,A,!invalid,!invalid}}:I.A", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 17, column: 3, scope: !8)
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.B() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.{array_type,{int_value,{Core.IntLiteral},1},{class_type,C,<package>,A,!invalid,!invalid}}:I.A"(), !dbg !7
// CHECK:STDOUT:   call void @"_CF.{array_type,{int_value,{Core.IntLiteral},1},{class_type,D,N,<package>,A,!invalid,!invalid}}:I.A"(), !dbg !8
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.{array_type,{int_value,{Core.IntLiteral},1},{class_type,C,<package>,A,!invalid,!invalid}}:I.A"()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.{array_type,{int_value,{Core.IntLiteral},1},{class_type,D,N,<package>,A,!invalid,!invalid}}:I.A"()
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
// CHECK:STDOUT: !7 = !DILocation(line: 5, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 4, column: 1, scope: !4)
