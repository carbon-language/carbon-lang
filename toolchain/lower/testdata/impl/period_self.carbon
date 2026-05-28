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

class A(T:! type) {}
class B {}

impl forall [T:! type] A(T) as J {}

impl forall [T:! type where A(.Self) impls J] A(T) as Core.As(T*) {
  fn Convert[self: A(T)]() -> T* {
    return self as T*;
  }
}

fn F[T:! type](x: A(T)) {
  x as T*;
}

// --- import_period_self_in_impl.carbon

library "[[@TEST_NAME]]";

import library "library_with_period_self_in_impl";

fn G(a: A(B)) {
  F(a);
}

// CHECK:STDOUT: ; ModuleID = 'library_with_period_self_in_impl.carbon'
// CHECK:STDOUT: source_filename = "library_with_period_self_in_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "library_with_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'import_period_self_in_impl.carbon'
// CHECK:STDOUT: source_filename = "import_period_self_in_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.b4fc7929164fb220(ptr %a), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.b4fc7929164fb220(ptr %x) #0 !dbg !12 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.d9476ddaeffa1656"(ptr %x), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.d9476ddaeffa1656"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.d9476ddaeffa1656"(ptr %self), !dbg !23
// CHECK:STDOUT:   ret ptr %1, !dbg !24
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
// CHECK:STDOUT: !3 = !DIFile(filename: "import_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b4fc7929164fb220", scope: null, file: !13, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !13 = !DIFile(filename: "library_with_period_self_in_impl.carbon", directory: "")
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 1, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.A.5d7343568c0b5715.Main:As.00d5ad878904f3d1.Core.d9476ddaeffa1656", scope: null, file: !13, line: 12, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{!7, !7}
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !23 = !DILocation(line: 13, column: 12, scope: !18)
// CHECK:STDOUT: !24 = !DILocation(line: 13, column: 5, scope: !18)
