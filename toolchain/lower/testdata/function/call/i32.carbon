// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/i32.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/i32.carbon

fn Echo(a: i32) -> i32 {
  return a;
}

fn Main() {
  var b: i32 = Echo(1);
}

// CHECK:STDOUT: ; ModuleID = 'i32.carbon'
// CHECK:STDOUT: source_filename = "i32.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Echo(i32 %a) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca i32, align 4
// CHECK:STDOUT:   %Echo.call = call i32 @Echo(i32 1)
// CHECK:STDOUT:   store i32 %Echo.call, ptr %b.var, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "i32.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Echo", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
