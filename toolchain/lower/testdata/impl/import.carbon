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

// CHECK:STDOUT: ; ModuleID = 'impl_def.carbon'
// CHECK:STDOUT: source_filename = "impl_def.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.A.Main:I.Main"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_16.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc14_16.2 = load i32, ptr %.loc14_16.1.n, align 4, !dbg !11
// CHECK:STDOUT:   ret i32 %.loc14_16.2, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "impl_def.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main:I.Main", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 12, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 5, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'impl_use.carbon'
// CHECK:STDOUT: source_filename = "impl_use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main(ptr %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.I.impl.F.call = call i32 @"_CF.A.Main:I.Main"(ptr %a), !dbg !11
// CHECK:STDOUT:   ret i32 %A.as.I.impl.F.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_CF.A.Main:I.Main"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "impl_use.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !1, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 5, column: 10, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 3, scope: !4)
