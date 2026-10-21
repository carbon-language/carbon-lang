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
// CHECK:STDOUT: define i1 @_CF.Main(i1 %a, i1 %b) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 %a, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "not_evaluated.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!18, !18, !18}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !16, line: 22, type: !24, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !40)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !18)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 2, scope: !25, type: !18)
// CHECK:STDOUT: !39 = !DILocation(line: 25, column: 3, scope: !25)
// CHECK:STDOUT: !40 = !{!26, !27}
// CHECK:STDOUT:
