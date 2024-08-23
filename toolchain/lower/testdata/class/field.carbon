// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/field.carbon

class C {
  var a: i32;
  var b: C*;
}

fn F(c: C) -> i32 {
  return (*c.b).a;
}

fn Run() -> i32 {
  var c: C;
  c.a = 1;
  c.b = &c;
  return F(c);
}

// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %c) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.1.b = getelementptr inbounds nuw { i32, ptr }, ptr %c, i32 0, i32 1
// CHECK:STDOUT:   %.loc17_13.2 = load ptr, ptr %.loc17_13.1.b, align 8
// CHECK:STDOUT:   %.loc17_16.1.a = getelementptr inbounds nuw { i32, ptr }, ptr %.loc17_13.2, i32 0, i32 0
// CHECK:STDOUT:   %.loc17_16.2 = load i32, ptr %.loc17_16.1.a, align 4
// CHECK:STDOUT:   ret i32 %.loc17_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, ptr }, align 8
// CHECK:STDOUT:   %.loc22_4.a = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %.loc22_4.a, align 4
// CHECK:STDOUT:   %.loc23_4.b = getelementptr inbounds nuw { i32, ptr }, ptr %c.var, i32 0, i32 1
// CHECK:STDOUT:   store ptr %c.var, ptr %.loc23_4.b, align 8
// CHECK:STDOUT:   %F.call = call i32 @F(ptr %c.var)
// CHECK:STDOUT:   ret i32 %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
