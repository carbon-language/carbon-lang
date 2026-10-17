// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/typeof/not_evaluated.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/typeof/not_evaluated.carbon

// The operand of `typeof` should not be evaluated at runtime.

class X {
  var a: bool;
}

fn Get() -> X;
fn Make(x: X) -> bool;

fn F(a: bool, b: bool) -> bool {
  let c: typeof(Make(Get())) = a;
  let unused d: typeof(a and b) = c;
  return c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'not_evaluated.carbon'
// CHECK:STDOUT: source_filename = "not_evaluated.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CGet.Main(ptr sret({ i1 }))
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CMake.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CF.Main(i1 %a, i1 %b) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 %a, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "not_evaluated.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{!20, !20, !20}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 22, type: !26, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !42)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !20)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !27, type: !20)
// CHECK:STDOUT: !41 = !DILocation(line: 25, column: 3, scope: !27)
// CHECK:STDOUT: !42 = !{!28, !29}
// CHECK:STDOUT:
