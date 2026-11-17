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
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "library_with_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_period_self_in_impl.carbon'
// CHECK:STDOUT: source_filename = "import_period_self_in_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr %a) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.a6a60dfee2d7d273(ptr %a), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.a6a60dfee2d7d273(ptr %x) #0 !dbg !34 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.229c89d4d5cd2a3b"(ptr %x), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.229c89d4d5cd2a3b"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.229c89d4d5cd2a3b"(ptr %self), !dbg !47
// CHECK:STDOUT:   ret ptr %1, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "library_with_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIFile(filename: "import_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !22, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !33)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 6, column: 3, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 5, column: 1, scope: !27)
// CHECK:STDOUT: !33 = !{!28}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a6a60dfee2d7d273", scope: null, file: !20, line: 16, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !24)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 3, scope: !34)
// CHECK:STDOUT: !38 = !{!24, !24}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 16, column: 1, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.229c89d4d5cd2a3b", scope: null, file: !20, line: 11, type: !39, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !49)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !24)
// CHECK:STDOUT: !47 = !DILocation(line: 12, column: 12, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 12, column: 5, scope: !42)
// CHECK:STDOUT: !49 = !{!43}
// CHECK:STDOUT:
