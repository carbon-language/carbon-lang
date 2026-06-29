// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/instance_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/instance_method.carbon

class A;

interface GetSelf {
  fn Get(ref self) -> Self*;
}

class A {
  extend impl as GetSelf {
    fn Get(ref self) -> Self* {
      return &self;
    }
  }
}

fn Call(a: A*) -> A* {
  return (*a).Get();
}

// CHECK:STDOUT: ; ModuleID = 'instance_method.carbon'
// CHECK:STDOUT: source_filename = "instance_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @"_CGet.A.Main:GetSelf.Main"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %self, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CCall.Main(ptr %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.GetSelf.impl.Get.call = call ptr @"_CGet.A.Main:GetSelf.Main"(ptr %a), !dbg !14
// CHECK:STDOUT:   ret ptr %A.as.GetSelf.impl.Get.call, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "instance_method.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.A.Main:GetSelf.Main", scope: null, file: !1, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 7, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !1, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 28, column: 10, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 28, column: 3, scope: !11)
