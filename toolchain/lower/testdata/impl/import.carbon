// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/import.carbon

// --- impl_def.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F(self) -> i32;
}

class A {
  var n: i32;
}

impl A as I {
  fn F(self: A) -> i32 {
    return self.n;
  }
}

// --- impl_use.carbon

import library "impl_def";

fn Call(a: A) -> i32 {
  return a.(I.F)();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'impl_def.carbon'
// CHECK:STDOUT: source_filename = "impl_def.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.A.Main:I.Main"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_16.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc14_16.2 = load i32, ptr %.loc14_16.1.n, align 4, !dbg !31
// CHECK:STDOUT:   ret i32 %.loc14_16.2, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "impl_def.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main:I.Main", scope: null, file: !20, line: 13, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 12, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 14, column: 5, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'impl_use.carbon'
// CHECK:STDOUT: source_filename = "impl_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main(ptr %a) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.I.impl.F.call = call i32 @"_CF.A.Main:I.Main"(ptr %a), !dbg !41
// CHECK:STDOUT:   ret i32 %A.as.I.impl.F.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_CF.A.Main:I.Main"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!22, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !34 = !DIFile(filename: "impl_use.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !34, line: 4, type: !25, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !43)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !23)
// CHECK:STDOUT: !41 = !DILocation(line: 5, column: 10, scope: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 5, column: 3, scope: !36)
// CHECK:STDOUT: !43 = !{!37}
// CHECK:STDOUT:
