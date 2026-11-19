// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/deduced_return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/deduced_return.carbon

// --- generic.carbon
library "[[@TEST_NAME]]";

fn Id[T: Core.Copy](x: T) -> auto { return x; }
fn Pair[T: Core.Copy](x: T) -> auto { return (x, x); }
fn IdFromGeneric[T: Core.Copy](x: T) -> T { return Id(x); }

fn Use(n: i32) -> i32 {
  let unused p: (i32, i32) = Pair(n);
  let unused b: bool = IdFromGeneric(true);
  return Id(n);
}

// --- template.carbon
library "[[@TEST_NAME]]";

fn Negate[template T: type](a: T) -> auto { return -a; }

fn CallNegate(a: i32) -> i32 { return Negate(a); }

// --- lib.carbon
library "[[@TEST_NAME]]";

fn Pair[T: Core.Copy](x: T) -> auto { return (x, x); }

// --- use_lib.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn Use(n: i32) -> (i32, i32) { return Pair(n); }

// --- form.carbon
library "[[@TEST_NAME]]";

fn RefId[T: type](ref x: T) -> ref auto { return x; }
fn ValId[T: type](x: T) -> val auto { return x; }
fn RefIdFromGeneric[T: type](ref x: T) -> ref T { return RefId(ref x); }

fn Use(ref n: i32) -> i32 {
  let unused p: i32* = &RefIdFromGeneric(ref n);
  let unused q: i32* = &RefId(ref n);
  return ValId(n);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic.carbon'
// CHECK:STDOUT: source_filename = "generic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main(i32 %n) #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_36.3.temp = alloca { i32, i32 }, align 4, !dbg !15
// CHECK:STDOUT:   %tuple = alloca { i32, i32 }, align 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_36.3.temp), !dbg !15
// CHECK:STDOUT:   call void @_CPair.Main.2fad9574e050607e(ptr %.loc8_36.3.temp, i32 %n), !dbg !15
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc8_36.3.temp, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   %.loc8_36.5 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc8_36.3.temp, i32 0, i32 1, !dbg !15
// CHECK:STDOUT:   %.loc8_36.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple), !dbg !15
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   store i32 %.loc8_36.5, ptr %tuple1, align 4, !dbg !15
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple, i32 0, i32 1, !dbg !15
// CHECK:STDOUT:   store i32 %.loc8_36.6, ptr %tuple2, align 4, !dbg !15
// CHECK:STDOUT:   %IdFromGeneric.call = call i1 @_CIdFromGeneric.Main.1ac5a6b0e5910c46(i1 true), !dbg !21
// CHECK:STDOUT:   %Id.call = call i32 @_CId.Main.2fad9574e050607e(i32 %n), !dbg !27
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %.loc8_36.3.temp), !dbg !15
// CHECK:STDOUT:   ret i32 %Id.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CPair.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem0.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem1.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CIdFromGeneric.Main.1ac5a6b0e5910c46(i1 %x) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Id.call = call i1 @_CId.Main.1ac5a6b0e5910c46(i1 %x), !dbg !46
// CHECK:STDOUT:   ret i1 %Id.call, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CId.Main.2fad9574e050607e(i32 %x) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %t) #0 !dbg !56 {
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CId.Main.1ac5a6b0e5910c46(i1 %x) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 %x, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "generic.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !4, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !9, type: !6)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 30, scope: !9)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!16, !6}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 24, scope: !9)
// CHECK:STDOUT: !22 = !{!16, !16}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 10, column: 10, scope: !9)
// CHECK:STDOUT: !28 = !{null, !16}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 10, column: 3, scope: !9)
// CHECK:STDOUT: !31 = !{!10}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Pair", linkageName: "_CPair.Main.2fad9574e050607e", scope: null, file: !4, line: 4, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !39)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !6)
// CHECK:STDOUT: !35 = !DILocation(line: 4, column: 46, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 4, column: 39, scope: !32)
// CHECK:STDOUT: !39 = !{!33}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "IdFromGeneric", linkageName: "_CIdFromGeneric.Main.1ac5a6b0e5910c46", scope: null, file: !4, line: 5, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !48)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !16)
// CHECK:STDOUT: !46 = !DILocation(line: 5, column: 52, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 5, column: 45, scope: !40)
// CHECK:STDOUT: !48 = !{!41}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Id", linkageName: "_CId.Main.2fad9574e050607e", scope: null, file: !4, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !6)
// CHECK:STDOUT: !53 = !DILocation(line: 3, column: 37, scope: !49)
// CHECK:STDOUT: !54 = !{!50}
// CHECK:STDOUT: !55 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ffbc82e0019b7bf2", scope: null, file: !55, line: 35, type: !29, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !16)
// CHECK:STDOUT: !61 = !DILocation(line: 35, column: 1, scope: !56)
// CHECK:STDOUT: !62 = !{!57}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Id", linkageName: "_CId.Main.1ac5a6b0e5910c46", scope: null, file: !4, line: 3, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !16)
// CHECK:STDOUT: !67 = !DILocation(line: 3, column: 37, scope: !63)
// CHECK:STDOUT: !68 = !{!64}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'template.carbon'
// CHECK:STDOUT: source_filename = "template.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallNegate.Main(i32 %a) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = call i32 @_CNegate.Main.55456748a8a2eb70(i32 %a), !dbg !77
// CHECK:STDOUT:   ret i32 %Negate.call, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CNegate.Main.55456748a8a2eb70(i32 %a) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !83
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !69 = !DIFile(filename: "template.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "CallNegate", linkageName: "_CCallNegate.Main", scope: null, file: !69, line: 5, type: !8, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !79)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !6)
// CHECK:STDOUT: !77 = !DILocation(line: 5, column: 39, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 5, column: 32, scope: !71)
// CHECK:STDOUT: !79 = !{!72}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Negate", linkageName: "_CNegate.Main.55456748a8a2eb70", scope: null, file: !69, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !86)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !6)
// CHECK:STDOUT: !83 = !DILocation(line: 3, column: 52, scope: !80)
// CHECK:STDOUT: !85 = !DILocation(line: 3, column: 45, scope: !80)
// CHECK:STDOUT: !86 = !{!81}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!88}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !87 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !88 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !87, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_lib.carbon'
// CHECK:STDOUT: source_filename = "use_lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main(ptr sret({ i32, i32 }) %return, i32 %n) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CPair.Main.2fad9574e050607e(ptr %return, i32 %n), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CPair.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x) #0 !dbg !100 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !103
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem, align 4, !dbg !103
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !103
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem1, align 4, !dbg !103
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!90}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!16, !6}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !87 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !89 = !DIFile(filename: "use_lib.carbon", directory: "")
// CHECK:STDOUT: !90 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !89, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !89, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !99)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !6)
// CHECK:STDOUT: !97 = !DILocation(line: 5, column: 39, scope: !91)
// CHECK:STDOUT: !98 = !DILocation(line: 5, column: 32, scope: !91)
// CHECK:STDOUT: !99 = !{!92}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Pair", linkageName: "_CPair.Main.2fad9574e050607e", scope: null, file: !87, line: 3, type: !18, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !107)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !6)
// CHECK:STDOUT: !103 = !DILocation(line: 3, column: 46, scope: !100)
// CHECK:STDOUT: !106 = !DILocation(line: 3, column: 39, scope: !100)
// CHECK:STDOUT: !107 = !{!101}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'form.carbon'
// CHECK:STDOUT: source_filename = "form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main(ptr %n) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RefIdFromGeneric.call = call ptr @_CRefIdFromGeneric.Main.55456748a8a2eb70(ptr %n), !dbg !117
// CHECK:STDOUT:   %RefId.call = call ptr @_CRefId.Main.55456748a8a2eb70(ptr %n), !dbg !123
// CHECK:STDOUT:   %.loc10 = load i32, ptr %n, align 4, !dbg !127
// CHECK:STDOUT:   %ValId.call = call i32 @_CValId.Main.55456748a8a2eb70(i32 %.loc10), !dbg !128
// CHECK:STDOUT:   ret i32 %ValId.call, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CRefIdFromGeneric.Main.55456748a8a2eb70(ptr %x) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RefId.call = call ptr @_CRefId.Main.55456748a8a2eb70(ptr %x), !dbg !138
// CHECK:STDOUT:   ret ptr %RefId.call, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CRefId.Main.55456748a8a2eb70(ptr %x) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CValId.Main.55456748a8a2eb70(i32 %x) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CRefId.Main.55456748a8a2eb70, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!109}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!16, !6}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !108 = !DIFile(filename: "form.carbon", directory: "")
// CHECK:STDOUT: !109 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !108, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !108, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !130)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !6)
// CHECK:STDOUT: !117 = !DILocation(line: 8, column: 25, scope: !110)
// CHECK:STDOUT: !123 = !DILocation(line: 9, column: 25, scope: !110)
// CHECK:STDOUT: !127 = !DILocation(line: 10, column: 16, scope: !110)
// CHECK:STDOUT: !128 = !DILocation(line: 10, column: 10, scope: !110)
// CHECK:STDOUT: !129 = !DILocation(line: 10, column: 3, scope: !110)
// CHECK:STDOUT: !130 = !{!111}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "RefIdFromGeneric", linkageName: "_CRefIdFromGeneric.Main.55456748a8a2eb70", scope: null, file: !108, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !140)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !6)
// CHECK:STDOUT: !138 = !DILocation(line: 5, column: 58, scope: !131)
// CHECK:STDOUT: !139 = !DILocation(line: 5, column: 51, scope: !131)
// CHECK:STDOUT: !140 = !{!132}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "RefId", linkageName: "_CRefId.Main.55456748a8a2eb70", scope: null, file: !108, line: 3, type: !18, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !146)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !6)
// CHECK:STDOUT: !145 = !DILocation(line: 3, column: 43, scope: !141)
// CHECK:STDOUT: !146 = !{!142}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "ValId", linkageName: "_CValId.Main.55456748a8a2eb70", scope: null, file: !108, line: 4, type: !8, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !152)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !6)
// CHECK:STDOUT: !151 = !DILocation(line: 4, column: 39, scope: !147)
// CHECK:STDOUT: !152 = !{!148}
// CHECK:STDOUT:
