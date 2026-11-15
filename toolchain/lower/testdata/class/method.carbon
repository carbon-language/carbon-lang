// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/method.carbon

class C {
  var a: i32;

  fn Get(self: C) -> i32;
  fn Set(ref self: C, n: i32);
}

fn F(p: C*) {
  let n: i32 = (*p).Get();
  (*p).Set(n);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'method.carbon'
// CHECK:STDOUT: source_filename = "method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CGet.C.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CSet.C.Main(ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %p) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %C.Get.call = call i32 @_CGet.C.Main(ptr %p), !dbg !35
// CHECK:STDOUT:   call void @_CSet.C.Main(ptr %p, i32 %C.Get.call), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "method.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !28 = !{null, !23}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 20, type: !29, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !43)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !23)
// CHECK:STDOUT: !35 = !DILocation(line: 21, column: 16, scope: !30)
// CHECK:STDOUT: !40 = !DILocation(line: 22, column: 3, scope: !30)
// CHECK:STDOUT: !42 = !DILocation(line: 20, column: 1, scope: !30)
// CHECK:STDOUT: !43 = !{!31}
// CHECK:STDOUT:
