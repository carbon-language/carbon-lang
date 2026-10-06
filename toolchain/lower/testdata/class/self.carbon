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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'self.carbon'
// CHECK:STDOUT: source_filename = "self.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CGet.C.Main(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_14.1.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc21_14.2 = load i32, ptr %.loc21_14.1.a, align 4, !dbg !31
// CHECK:STDOUT:   ret i32 %.loc21_14.2, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CSet.C.Main(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc25_7.a = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   store i32 1, ptr %.loc25_7.a, align 4, !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "self.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.C.Main", scope: null, file: !18, line: 20, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 21, column: 10, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 21, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Set", linkageName: "_CSet.C.Main", scope: null, file: !18, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !39)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 25, column: 3, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 24, column: 1, scope: !34)
// CHECK:STDOUT: !39 = !{!35}
// CHECK:STDOUT:
