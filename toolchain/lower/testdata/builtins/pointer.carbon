// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/pointer.carbon

fn MakeUnformed(t: type) -> type = "maybe_unformed.make_type";
fn MakeNull(T:! type) -> MakeUnformed(T*) = "pointer.make_null";
fn IsNull[T:! type](p: MakeUnformed(T*)) -> bool = "pointer.is_null";

class C {}

fn Null() -> MakeUnformed(C*) {
  return MakeNull(C);
}

fn Check(p: MakeUnformed(C*)) -> bool {
  return IsNull(p);
}

// CHECK:STDOUT: ; ModuleID = 'pointer.carbon'
// CHECK:STDOUT: source_filename = "pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CNull.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr null, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCheck.Main(ptr %p) #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %IsNull.call = icmp eq ptr %p, null, !dbg !9
// CHECK:STDOUT:   ret i1 %IsNull.call, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "pointer.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Null", linkageName: "_CNull.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Check", linkageName: "_CCheck.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 24, column: 10, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 24, column: 3, scope: !8)
