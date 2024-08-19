// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/extend_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/extend_impl.carbon

interface I {
  fn F() -> i32;
}

class A {
  extend impl as I {
    fn F() -> i32 {
      return 0;
    }
  }
}

fn TypeAccess() -> i32 {
  return A.F();
}

fn InstanceAccess(a: A) -> i32 {
  return a.F();
}

// CHECK:STDOUT: ; ModuleID = 'extend_impl.carbon'
// CHECK:STDOUT: source_filename = "extend_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TypeAccess() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @F()
// CHECK:STDOUT:   ret i32 %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @InstanceAccess(ptr %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @F()
// CHECK:STDOUT:   ret i32 %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "extend_impl.carbon", directory: "")
