// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/partial/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/partial/generic.carbon

// --- base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

fn F[T: Core.Extendable](unused p: partial T*) {}

fn G(b: partial B*) {
  F(b);
}

// --- abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class A {}

fn F[T: Core.Extendable](unused p: partial T*) {}

fn G(a: partial A*) {
  F(a);
}

// --- convert.carbon
library "[[@TEST_NAME]]";

base class B {}

fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}

fn Test(ref b: B) {
  F(ref b);
}

// --- period_self.carbon
library "[[@TEST_NAME]]";

interface I {
  fn Op(ref self);
}

base class B {}

impl partial B as I {
  fn Op(ref unused self) {}
}

fn F[T: Core.Extendable where partial .Self impls I](ref t: T) {
  (t as partial T).(I.Op)();
}

fn G(ref b: B) {
  F(ref b);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'base_class.carbon'
// CHECK:STDOUT: source_filename = "base_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.0b07873d58d5cfb0(ptr %b), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.0b07873d58d5cfb0(ptr %p) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "base_class.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !14, line: 7, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !25)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !16)
// CHECK:STDOUT: !23 = !DILocation(line: 8, column: 3, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 7, column: 1, scope: !19)
// CHECK:STDOUT: !25 = !{!20}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.0b07873d58d5cfb0", scope: null, file: !14, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !30)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !16)
// CHECK:STDOUT: !29 = !DILocation(line: 5, column: 1, scope: !26)
// CHECK:STDOUT: !30 = !{!27}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'abstract_class.carbon'
// CHECK:STDOUT: source_filename = "abstract_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr %a) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.9157eb1f747dd594(ptr %a), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.9157eb1f747dd594(ptr %p) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!32}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !31 = !DIFile(filename: "abstract_class.carbon", directory: "")
// CHECK:STDOUT: !32 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !31, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !31, line: 7, type: !18, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !39)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !16)
// CHECK:STDOUT: !37 = !DILocation(line: 8, column: 3, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 7, column: 1, scope: !33)
// CHECK:STDOUT: !39 = !{!34}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9157eb1f747dd594", scope: null, file: !31, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !44)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !16)
// CHECK:STDOUT: !43 = !DILocation(line: 5, column: 1, scope: !40)
// CHECK:STDOUT: !44 = !{!41}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTest.Main(ptr %b) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.0b07873d58d5cfb0(ptr %b), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.0b07873d58d5cfb0(ptr %t) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %t, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!46}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !45 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !46 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !45, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !45, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !46, retainedNodes: !56)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !16)
// CHECK:STDOUT: !52 = !DILocation(line: 10, column: 3, scope: !47)
// CHECK:STDOUT: !53 = !{!16, !16}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 9, column: 1, scope: !47)
// CHECK:STDOUT: !56 = !{!48}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.0b07873d58d5cfb0", scope: null, file: !45, line: 5, type: !54, spFlags: DISPFlagDefinition, unit: !46, retainedNodes: !65)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !16)
// CHECK:STDOUT: !64 = !DILocation(line: 6, column: 3, scope: !57)
// CHECK:STDOUT: !65 = !{!58}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'period_self.carbon'
// CHECK:STDOUT: source_filename = "period_self.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.640ddcd79407625a:I.Main"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(ptr %b) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.2eeda7c1181f8065(ptr %b), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.2eeda7c1181f8065(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.640ddcd79407625a:I.Main"(ptr %t), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!67}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !66 = !DIFile(filename: "period_self.carbon", directory: "")
// CHECK:STDOUT: !67 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !66, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Op", linkageName: "_COp.640ddcd79407625a:I.Main", scope: null, file: !66, line: 10, type: !18, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !72)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !16)
// CHECK:STDOUT: !71 = !DILocation(line: 10, column: 3, scope: !68)
// CHECK:STDOUT: !72 = !{!69}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !66, line: 17, type: !18, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !80)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !16)
// CHECK:STDOUT: !78 = !DILocation(line: 18, column: 3, scope: !73)
// CHECK:STDOUT: !79 = !DILocation(line: 17, column: 1, scope: !73)
// CHECK:STDOUT: !80 = !{!74}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.2eeda7c1181f8065", scope: null, file: !66, line: 13, type: !18, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !16)
// CHECK:STDOUT: !85 = !DILocation(line: 14, column: 3, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 13, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT:
