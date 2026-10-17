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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'instance_method.carbon'
// CHECK:STDOUT: source_filename = "instance_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @"_CGet.A.Main:GetSelf.Main"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %self, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CCall.Main(ptr %a) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.as.GetSelf.impl.Get.call = call ptr @"_CGet.A.Main:GetSelf.Main"(ptr %a), !dbg !37
// CHECK:STDOUT:   ret ptr %A.as.GetSelf.impl.Get.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "instance_method.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!18, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.A.Main:GetSelf.Main", scope: null, file: !16, line: 21, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !29)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !18)
// CHECK:STDOUT: !28 = !DILocation(line: 22, column: 7, scope: !21)
// CHECK:STDOUT: !29 = !{!22}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !16, line: 27, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !39)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !18)
// CHECK:STDOUT: !37 = !DILocation(line: 28, column: 10, scope: !30)
// CHECK:STDOUT: !38 = !DILocation(line: 28, column: 3, scope: !30)
// CHECK:STDOUT: !39 = !{!31}
// CHECK:STDOUT:
