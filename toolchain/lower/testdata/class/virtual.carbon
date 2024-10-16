// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/virtual.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/virtual.carbon

base class Base {
}
base class Intermediate {
  extend base: Base;
  virtual fn Fn();
}
class Derived {
  extend base: Intermediate;
  impl fn Fn();
}

fn Create() {
  var b: Base;
  var i: Intermediate;
  var d: Derived;
}

fn Use(v: Intermediate*) {
  v->Fn();
}

// CHECK:STDOUT: ; ModuleID = 'virtual.carbon'
// CHECK:STDOUT: source_filename = "virtual.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Intermediate.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFn.Derived.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCreate.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca {}, align 8, !dbg !7
// CHECK:STDOUT:   %i.var = alloca { ptr, {} }, align 8, !dbg !8
// CHECK:STDOUT:   %d.var = alloca { { ptr, {} } }, align 8, !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUse.Main(ptr %v) !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CFn.Intermediate.Main(), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "virtual.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Create", linkageName: "_CCreate.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 23, column: 7, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 24, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 25, column: 7, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DILocation(line: 29, column: 3, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 28, column: 1, scope: !11)
