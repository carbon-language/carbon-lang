// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/call/implicit_empty_tuple_as_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/call/implicit_empty_tuple_as_arg.carbon


fn Foo() {}

fn Bar(a: ()) -> () { return a; }

fn Main() {
  var x: () = Bar(Foo());
}

// CHECK:STDOUT: ; ModuleID = 'implicit_empty_tuple_as_arg.carbon'
// CHECK:STDOUT: source_filename = "implicit_empty_tuple_as_arg.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Foo() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Bar() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca {}, align 8
// CHECK:STDOUT:   call void @Foo()
// CHECK:STDOUT:   %.loc17_22.1.temp = alloca {}, align 8
// CHECK:STDOUT:   call void @Bar()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "implicit_empty_tuple_as_arg.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Foo", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Bar", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
