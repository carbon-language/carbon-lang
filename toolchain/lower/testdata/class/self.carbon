// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/self.carbon

class C {
  var a: i32;

  fn Get(self: C) -> i32;
  fn Set(ref self: C);
}

fn C.Get(self: C) -> i32 {
  return self.a;
}

fn C.Set(ref self: C) {
  self.a = 1;
}

// CHECK:STDOUT: ; ModuleID = 'self.carbon'
// CHECK:STDOUT: source_filename = "self.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGet.C.Main(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_14.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc21_14.2 = load i32, ptr %.loc21_14.1.a, align 4, !dbg !11
// CHECK:STDOUT:   ret i32 %.loc21_14.2, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CSet.C.Main(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc25_7.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   store i32 1, ptr %.loc25_7.a, align 4, !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "self.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.C.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 21, column: 10, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Set", linkageName: "_CSet.C.Main", scope: null, file: !3, line: 24, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !8}
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !13, type: !8)
// CHECK:STDOUT: !18 = !DILocation(line: 25, column: 3, scope: !13)
// CHECK:STDOUT: !19 = !DILocation(line: 24, column: 1, scope: !13)
