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
// CHECK:STDOUT: define i32 @_CUse.Main(i32 %n) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_36.3.temp = alloca { i32, i32 }, align 4, !dbg !18
// CHECK:STDOUT:   %tuple = alloca { i32, i32 }, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_36.3.temp), !dbg !18
// CHECK:STDOUT:   call void @_CPair.Main.2fad9574e050607e(ptr %.loc8_36.3.temp, i32 %n), !dbg !18
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc8_36.3.temp, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   %.loc8_36.5 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !18
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc8_36.3.temp, i32 0, i32 1, !dbg !18
// CHECK:STDOUT:   %.loc8_36.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple), !dbg !18
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   store i32 %.loc8_36.5, ptr %tuple1, align 4, !dbg !18
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple, i32 0, i32 1, !dbg !18
// CHECK:STDOUT:   store i32 %.loc8_36.6, ptr %tuple2, align 4, !dbg !18
// CHECK:STDOUT:   %IdFromGeneric.call = call i1 @_CIdFromGeneric.Main.1ac5a6b0e5910c46(i1 true), !dbg !23
// CHECK:STDOUT:   %Id.call = call i32 @_CId.Main.2fad9574e050607e(i32 %n), !dbg !29
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.eda669d623214d41(ptr %.loc8_36.3.temp), !dbg !18
// CHECK:STDOUT:   ret i32 %Id.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.35f6012581c5aabb:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %tuple.elem1.tuple.elem), !dbg !34
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %tuple.elem0.tuple.elem), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CPair.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem0.tuple.elem, align 4, !dbg !41
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !41
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem1.tuple.elem, align 4, !dbg !41
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CIdFromGeneric.Main.1ac5a6b0e5910c46(i1 %x) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Id.call = call i1 @_CId.Main.1ac5a6b0e5910c46(i1 %x), !dbg !52
// CHECK:STDOUT:   ret i1 %Id.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CId.Main.2fad9574e050607e(i32 %x) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.eda669d623214d41(ptr %t) #0 !dbg !62 {
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:core.SubobjectDestroy.Core"(ptr %t), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dc5658e0ac74b2aa(ptr %t) #0 !dbg !69 {
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CId.Main.1ac5a6b0e5910c46(i1 %x) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 %x, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.dc5658e0ac74b2aa, { 1, 0 }
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
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !4, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !12, type: !6)
// CHECK:STDOUT: !18 = !DILocation(line: 8, column: 30, scope: !12)
// CHECK:STDOUT: !19 = !{!9, !6}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 9, column: 24, scope: !12)
// CHECK:STDOUT: !24 = !{!9, !9}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 10, column: 10, scope: !12)
// CHECK:STDOUT: !30 = !DILocation(line: 10, column: 3, scope: !12)
// CHECK:STDOUT: !31 = !{!13}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.35f6012581c5aabb:core.SubobjectDestroy.Core", scope: null, file: !4, line: 8, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !9)
// CHECK:STDOUT: !34 = !DILocation(line: 8, column: 30, scope: !32)
// CHECK:STDOUT: !35 = !{null, !6}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{!33}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Pair", linkageName: "_CPair.Main.2fad9574e050607e", scope: null, file: !4, line: 4, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !6)
// CHECK:STDOUT: !41 = !DILocation(line: 4, column: 46, scope: !38)
// CHECK:STDOUT: !44 = !DILocation(line: 4, column: 39, scope: !38)
// CHECK:STDOUT: !45 = !{!39}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "IdFromGeneric", linkageName: "_CIdFromGeneric.Main.1ac5a6b0e5910c46", scope: null, file: !4, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !9)
// CHECK:STDOUT: !52 = !DILocation(line: 5, column: 52, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 5, column: 45, scope: !46)
// CHECK:STDOUT: !54 = !{!47}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Id", linkageName: "_CId.Main.2fad9574e050607e", scope: null, file: !4, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !60)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !6)
// CHECK:STDOUT: !59 = !DILocation(line: 3, column: 37, scope: !55)
// CHECK:STDOUT: !60 = !{!56}
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.eda669d623214d41", scope: null, file: !61, line: 34, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !9)
// CHECK:STDOUT: !66 = !DILocation(line: 36, column: 5, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 34, column: 1, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dc5658e0ac74b2aa", scope: null, file: !61, line: 34, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !75)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !6)
// CHECK:STDOUT: !74 = !DILocation(line: 34, column: 1, scope: !69)
// CHECK:STDOUT: !75 = !{!70}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Id", linkageName: "_CId.Main.1ac5a6b0e5910c46", scope: null, file: !4, line: 3, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !81)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !9)
// CHECK:STDOUT: !80 = !DILocation(line: 3, column: 37, scope: !76)
// CHECK:STDOUT: !81 = !{!77}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'template.carbon'
// CHECK:STDOUT: source_filename = "template.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallNegate.Main(i32 %a) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = call i32 @_CNegate.Main.55456748a8a2eb70(i32 %a), !dbg !90
// CHECK:STDOUT:   ret i32 %Negate.call, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CNegate.Main.55456748a8a2eb70(i32 %a) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !96
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!83}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !82 = !DIFile(filename: "template.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !82, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "CallNegate", linkageName: "_CCallNegate.Main", scope: null, file: !82, line: 5, type: !8, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !92)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !6)
// CHECK:STDOUT: !90 = !DILocation(line: 5, column: 39, scope: !84)
// CHECK:STDOUT: !91 = !DILocation(line: 5, column: 32, scope: !84)
// CHECK:STDOUT: !92 = !{!85}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Negate", linkageName: "_CNegate.Main.55456748a8a2eb70", scope: null, file: !82, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !83, retainedNodes: !99)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !6)
// CHECK:STDOUT: !96 = !DILocation(line: 3, column: 52, scope: !93)
// CHECK:STDOUT: !98 = !DILocation(line: 3, column: 45, scope: !93)
// CHECK:STDOUT: !99 = !{!94}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!101}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !100 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !101 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !100, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_lib.carbon'
// CHECK:STDOUT: source_filename = "use_lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main(ptr sret({ i32, i32 }) %return, i32 %n) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CPair.Main.2fad9574e050607e(ptr %return, i32 %n), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CPair.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x) #0 !dbg !113 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem, align 4, !dbg !116
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !116
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem1, align 4, !dbg !116
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!103}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!9, !6}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !100 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !102 = !DIFile(filename: "use_lib.carbon", directory: "")
// CHECK:STDOUT: !103 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !102, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !102, line: 5, type: !20, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !112)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !6)
// CHECK:STDOUT: !110 = !DILocation(line: 5, column: 39, scope: !104)
// CHECK:STDOUT: !111 = !DILocation(line: 5, column: 32, scope: !104)
// CHECK:STDOUT: !112 = !{!105}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Pair", linkageName: "_CPair.Main.2fad9574e050607e", scope: null, file: !100, line: 3, type: !20, spFlags: DISPFlagDefinition, unit: !103, retainedNodes: !120)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !6)
// CHECK:STDOUT: !116 = !DILocation(line: 3, column: 46, scope: !113)
// CHECK:STDOUT: !119 = !DILocation(line: 3, column: 39, scope: !113)
// CHECK:STDOUT: !120 = !{!114}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'form.carbon'
// CHECK:STDOUT: source_filename = "form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main(ptr %n) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RefIdFromGeneric.call = call ptr @_CRefIdFromGeneric.Main.55456748a8a2eb70(ptr %n), !dbg !130
// CHECK:STDOUT:   %RefId.call = call ptr @_CRefId.Main.55456748a8a2eb70(ptr %n), !dbg !136
// CHECK:STDOUT:   %.loc10 = load i32, ptr %n, align 4, !dbg !140
// CHECK:STDOUT:   %ValId.call = call i32 @_CValId.Main.55456748a8a2eb70(i32 %.loc10), !dbg !141
// CHECK:STDOUT:   ret i32 %ValId.call, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CRefIdFromGeneric.Main.55456748a8a2eb70(ptr %x) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RefId.call = call ptr @_CRefId.Main.55456748a8a2eb70(ptr %x), !dbg !151
// CHECK:STDOUT:   ret ptr %RefId.call, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CRefId.Main.55456748a8a2eb70(ptr %x) #0 !dbg !154 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CValId.Main.55456748a8a2eb70(i32 %x) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CRefId.Main.55456748a8a2eb70, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!122}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!9, !6}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !121 = !DIFile(filename: "form.carbon", directory: "")
// CHECK:STDOUT: !122 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !121, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !121, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !143)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !6)
// CHECK:STDOUT: !130 = !DILocation(line: 8, column: 25, scope: !123)
// CHECK:STDOUT: !136 = !DILocation(line: 9, column: 25, scope: !123)
// CHECK:STDOUT: !140 = !DILocation(line: 10, column: 16, scope: !123)
// CHECK:STDOUT: !141 = !DILocation(line: 10, column: 10, scope: !123)
// CHECK:STDOUT: !142 = !DILocation(line: 10, column: 3, scope: !123)
// CHECK:STDOUT: !143 = !{!124}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "RefIdFromGeneric", linkageName: "_CRefIdFromGeneric.Main.55456748a8a2eb70", scope: null, file: !121, line: 5, type: !20, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !153)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !6)
// CHECK:STDOUT: !151 = !DILocation(line: 5, column: 58, scope: !144)
// CHECK:STDOUT: !152 = !DILocation(line: 5, column: 51, scope: !144)
// CHECK:STDOUT: !153 = !{!145}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "RefId", linkageName: "_CRefId.Main.55456748a8a2eb70", scope: null, file: !121, line: 3, type: !20, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !159)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !6)
// CHECK:STDOUT: !158 = !DILocation(line: 3, column: 43, scope: !154)
// CHECK:STDOUT: !159 = !{!155}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "ValId", linkageName: "_CValId.Main.55456748a8a2eb70", scope: null, file: !121, line: 4, type: !8, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !165)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !6)
// CHECK:STDOUT: !164 = !DILocation(line: 4, column: 39, scope: !160)
// CHECK:STDOUT: !165 = !{!161}
// CHECK:STDOUT:
