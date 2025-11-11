// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is an integration test for the Core.String in the prelude, so use the
// real prelude.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/string.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/string.carbon

fn F(s: str);

fn G() {
  F("Hello");
}

fn H() {
  // This can reuse the same string literal object.
  let v: str = "Hello";
  F(v);
}

fn I() {
  // This can't.
  let v: str = "World";
  F(v);
}

fn Copy(s: str) -> str {
  return s;
}

// CHECK:STDOUT: ; ModuleID = 'string.carbon'
// CHECK:STDOUT: source_filename = "string.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @0 = private unnamed_addr constant [6 x i8] c"Hello\00", align 1
// CHECK:STDOUT: @1 = private unnamed_addr constant [6 x i8] c"World\00", align 1
// CHECK:STDOUT: @String.val.13d.String.val = internal constant { ptr, i64 } { ptr @0, i64 5 }
// CHECK:STDOUT: @String.val.afc.String.val = internal constant { ptr, i64 } { ptr @1, i64 5 }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr @String.val.13d.String.val), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr @String.val.13d.String.val), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CI.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(ptr @String.val.afc.String.val), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.Main(ptr sret({ ptr, i64 }) %return, ptr %s) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.String.Core:Copy.Core"(ptr %return, ptr %s), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.String.Core:Copy.Core"(ptr sret({ ptr, i64 }), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "string.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 18, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 25, column: 3, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 22, column: 1, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "I", linkageName: "_CI.Main", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 31, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 28, column: 1, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !3, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 35, column: 10, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 35, column: 3, scope: !15)
