// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/period_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/period_self.carbon

// --- library_with_period_self_in_impl.carbon

library "[[@TEST_NAME]]";

interface J {}

class A(T: type) {}
class B {}

impl forall [T: type] A(T) as J {}

impl forall [T: type where A(.Self) impls J] A(T) as Core.As(T*) {
  fn Convert(self) -> T* {
    return self as T*;
  }
}

fn F[T: type](x: A(T)) {
  x as T*;
}

// --- import_period_self_in_impl.carbon

library "[[@TEST_NAME]]";

import library "library_with_period_self_in_impl";

fn G(a: A(B)) {
  F(a);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'library_with_period_self_in_impl.carbon'
// CHECK:STDOUT: source_filename = "library_with_period_self_in_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "library_with_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_period_self_in_impl.carbon'
// CHECK:STDOUT: source_filename = "import_period_self_in_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr %a) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.a6a60dfee2d7d273(ptr %a), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.a6a60dfee2d7d273(ptr %x) #0 !dbg !32 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.06ad18aa23ecf9a8"(ptr %x), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.06ad18aa23ecf9a8"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.06ad18aa23ecf9a8"(ptr %self), !dbg !45
// CHECK:STDOUT:   ret ptr %1, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "library_with_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIFile(filename: "import_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !31)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !22)
// CHECK:STDOUT: !29 = !DILocation(line: 7, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 6, column: 1, scope: !25)
// CHECK:STDOUT: !31 = !{!26}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a6a60dfee2d7d273", scope: null, file: !18, line: 17, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !39)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !22)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 3, scope: !32)
// CHECK:STDOUT: !36 = !{!22, !22}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 17, column: 1, scope: !32)
// CHECK:STDOUT: !39 = !{!33}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.06ad18aa23ecf9a8", scope: null, file: !18, line: 12, type: !37, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !47)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !22)
// CHECK:STDOUT: !45 = !DILocation(line: 13, column: 12, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 13, column: 5, scope: !40)
// CHECK:STDOUT: !47 = !{!41}
// CHECK:STDOUT:
