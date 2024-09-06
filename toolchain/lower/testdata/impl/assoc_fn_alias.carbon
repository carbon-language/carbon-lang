// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/assoc_fn_alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/assoc_fn_alias.carbon

interface I {
  fn F[self: Self]() -> i32;
}

class A {
  alias F = I.F;
  var n: i32;
}

impl A as I {
  fn F[self: A]() -> i32 {
    return self.n;
  }
}

fn Call(a: A) -> i32 {
  return a.F();
}

// CHECK:STDOUT: ; ModuleID = 'assoc_fn_alias.carbon'
// CHECK:STDOUT: source_filename = "assoc_fn_alias.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @"_CF.A.Main:I.Main"(ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_16.1.n = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc22_16.2 = load i32, ptr %.loc22_16.1.n, align 4, !dbg !7
// CHECK:STDOUT:   ret i32 %.loc22_16.2, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CCall.Main(ptr %a) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @"_CF.A.Main:I.Main"(ptr %a), !dbg !10
// CHECK:STDOUT:   ret i32 %F.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "assoc_fn_alias.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main:I.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 22, column: 12, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 22, column: 5, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 27, column: 10, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 27, column: 3, scope: !9)
