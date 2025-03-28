// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/method.carbon

class C {
  var a: i32;

  fn Get[self: C]() -> i32;
  fn Set[addr self: C*](n: i32);
}

fn F(p: C*) {
  let n: i32 = (*p).Get();
  (*p).Set(n);
}

// CHECK:STDOUT: ; ModuleID = 'method.carbon'
// CHECK:STDOUT: source_filename = "method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CGet.C.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CSet.C.Main(ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main(ptr %p) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Get.call = call i32 @_CGet.C.Main(ptr %p), !dbg !7
// CHECK:STDOUT:   %.loc19_25.1.temp = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   store i32 %Get.call, ptr %.loc19_25.1.temp, align 4, !dbg !7
// CHECK:STDOUT:   %.loc20_12 = load i32, ptr %.loc19_25.1.temp, align 4, !dbg !8
// CHECK:STDOUT:   call void @_CSet.C.Main(ptr %p, i32 %.loc20_12), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "method.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 19, column: 16, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 20, column: 12, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 1, scope: !4)
