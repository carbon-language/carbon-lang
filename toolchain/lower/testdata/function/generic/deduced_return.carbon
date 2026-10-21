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
// CHECK:STDOUT: define i32 @_CUse.Main(i32 %n) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_36.3.temp = alloca { i32, i32 }, align 4, !dbg !20
// CHECK:STDOUT:   %tuple = alloca { i32, i32 }, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_36.3.temp), !dbg !20
// CHECK:STDOUT:   call void @_CPair.Main.2fad9574e050607e(ptr %.loc8_36.3.temp, i32 %n), !dbg !20
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc8_36.3.temp, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   %.loc8_36.5 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !20
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %.loc8_36.3.temp, i32 0, i32 1, !dbg !20
// CHECK:STDOUT:   %.loc8_36.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple), !dbg !20
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   store i32 %.loc8_36.5, ptr %tuple1, align 4, !dbg !20
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple, i32 0, i32 1, !dbg !20
// CHECK:STDOUT:   store i32 %.loc8_36.6, ptr %tuple2, align 4, !dbg !20
// CHECK:STDOUT:   %IdFromGeneric.call = call i1 @_CIdFromGeneric.Main.1ac5a6b0e5910c46(i1 true), !dbg !25
// CHECK:STDOUT:   %Id.call = call i32 @_CId.Main.2fad9574e050607e(i32 %n), !dbg !31
// CHECK:STDOUT:   call void @"_CSelfDestruct.35f6012581c5aabb:core.Destroy.Core"(ptr %.loc8_36.3.temp), !dbg !20
// CHECK:STDOUT:   ret i32 %Id.call, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.35f6012581c5aabb:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !40
// CHECK:STDOUT:   call void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %tuple.elem1.tuple.elem), !dbg !40
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   call void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %tuple.elem0.tuple.elem), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.35f6012581c5aabb:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.35f6012581c5aabb:core.Destroy.Core"(ptr %self), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CPair.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem0.tuple.elem, align 4, !dbg !49
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem1.tuple.elem, align 4, !dbg !49
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CIdFromGeneric.Main.1ac5a6b0e5910c46(i1 %x) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Id.call = call i1 @_CId.Main.1ac5a6b0e5910c46(i1 %x), !dbg !60
// CHECK:STDOUT:   ret i1 %Id.call, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CId.Main.2fad9574e050607e(i32 %x) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CId.Main.1ac5a6b0e5910c46(i1 %x) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 %x, !dbg !73
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
// CHECK:STDOUT: !9 = !{null, !6}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !12 = !{null, !11}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !4, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !14, type: !6)
// CHECK:STDOUT: !20 = !DILocation(line: 8, column: 30, scope: !14)
// CHECK:STDOUT: !21 = !{!11, !6}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 9, column: 24, scope: !14)
// CHECK:STDOUT: !26 = !{!11, !11}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 10, column: 10, scope: !14)
// CHECK:STDOUT: !32 = !DILocation(line: 10, column: 3, scope: !14)
// CHECK:STDOUT: !33 = !{!15}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core", scope: null, file: !4, line: 8, type: !10, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !6)
// CHECK:STDOUT: !36 = !DILocation(line: 8, column: 30, scope: !34)
// CHECK:STDOUT: !37 = !{!35}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.35f6012581c5aabb:core.Destroy.Core", scope: null, file: !4, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !11)
// CHECK:STDOUT: !40 = !DILocation(line: 8, column: 30, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.35f6012581c5aabb:core.Destroy.Core", scope: null, file: !4, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !11)
// CHECK:STDOUT: !44 = !DILocation(line: 8, column: 30, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Pair", linkageName: "_CPair.Main.2fad9574e050607e", scope: null, file: !4, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !6)
// CHECK:STDOUT: !49 = !DILocation(line: 4, column: 46, scope: !46)
// CHECK:STDOUT: !52 = !DILocation(line: 4, column: 39, scope: !46)
// CHECK:STDOUT: !53 = !{!47}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "IdFromGeneric", linkageName: "_CIdFromGeneric.Main.1ac5a6b0e5910c46", scope: null, file: !4, line: 5, type: !27, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !11)
// CHECK:STDOUT: !60 = !DILocation(line: 5, column: 52, scope: !54)
// CHECK:STDOUT: !61 = !DILocation(line: 5, column: 45, scope: !54)
// CHECK:STDOUT: !62 = !{!55}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Id", linkageName: "_CId.Main.2fad9574e050607e", scope: null, file: !4, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !6)
// CHECK:STDOUT: !67 = !DILocation(line: 3, column: 37, scope: !63)
// CHECK:STDOUT: !68 = !{!64}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Id", linkageName: "_CId.Main.1ac5a6b0e5910c46", scope: null, file: !4, line: 3, type: !27, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !74)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !11)
// CHECK:STDOUT: !73 = !DILocation(line: 3, column: 37, scope: !69)
// CHECK:STDOUT: !74 = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'template.carbon'
// CHECK:STDOUT: source_filename = "template.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallNegate.Main(i32 %a) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = call i32 @_CNegate.Main.55456748a8a2eb70(i32 %a), !dbg !83
// CHECK:STDOUT:   ret i32 %Negate.call, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CNegate.Main.55456748a8a2eb70(i32 %a) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !89
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!76}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !75 = !DIFile(filename: "template.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !75, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "CallNegate", linkageName: "_CCallNegate.Main", scope: null, file: !75, line: 5, type: !8, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !85)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !6)
// CHECK:STDOUT: !83 = !DILocation(line: 5, column: 39, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 5, column: 32, scope: !77)
// CHECK:STDOUT: !85 = !{!78}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Negate", linkageName: "_CNegate.Main.55456748a8a2eb70", scope: null, file: !75, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !92)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !6)
// CHECK:STDOUT: !89 = !DILocation(line: 3, column: 52, scope: !86)
// CHECK:STDOUT: !91 = !DILocation(line: 3, column: 45, scope: !86)
// CHECK:STDOUT: !92 = !{!87}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!94}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !93 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !94 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !93, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_lib.carbon'
// CHECK:STDOUT: source_filename = "use_lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main(ptr sret({ i32, i32 }) %return, i32 %n) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CPair.Main.2fad9574e050607e(ptr %return, i32 %n), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CPair.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x) #0 !dbg !106 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !109
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem, align 4, !dbg !109
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !109
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem1, align 4, !dbg !109
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!96}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!11, !6}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !93 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !95 = !DIFile(filename: "use_lib.carbon", directory: "")
// CHECK:STDOUT: !96 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !95, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !95, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !96, retainedNodes: !105)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !6)
// CHECK:STDOUT: !103 = !DILocation(line: 5, column: 39, scope: !97)
// CHECK:STDOUT: !104 = !DILocation(line: 5, column: 32, scope: !97)
// CHECK:STDOUT: !105 = !{!98}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Pair", linkageName: "_CPair.Main.2fad9574e050607e", scope: null, file: !93, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !96, retainedNodes: !113)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !6)
// CHECK:STDOUT: !109 = !DILocation(line: 3, column: 46, scope: !106)
// CHECK:STDOUT: !112 = !DILocation(line: 3, column: 39, scope: !106)
// CHECK:STDOUT: !113 = !{!107}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'form.carbon'
// CHECK:STDOUT: source_filename = "form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main(ptr %n) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RefIdFromGeneric.call = call ptr @_CRefIdFromGeneric.Main.55456748a8a2eb70(ptr %n), !dbg !123
// CHECK:STDOUT:   %RefId.call = call ptr @_CRefId.Main.55456748a8a2eb70(ptr %n), !dbg !129
// CHECK:STDOUT:   %.loc10 = load i32, ptr %n, align 4, !dbg !133
// CHECK:STDOUT:   %ValId.call = call i32 @_CValId.Main.55456748a8a2eb70(i32 %.loc10), !dbg !134
// CHECK:STDOUT:   ret i32 %ValId.call, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CRefIdFromGeneric.Main.55456748a8a2eb70(ptr %x) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RefId.call = call ptr @_CRefId.Main.55456748a8a2eb70(ptr %x), !dbg !144
// CHECK:STDOUT:   ret ptr %RefId.call, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CRefId.Main.55456748a8a2eb70(ptr %x) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CValId.Main.55456748a8a2eb70(i32 %x) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CRefId.Main.55456748a8a2eb70, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!115}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!11, !6}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !114 = !DIFile(filename: "form.carbon", directory: "")
// CHECK:STDOUT: !115 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !114, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !114, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !136)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !6)
// CHECK:STDOUT: !123 = !DILocation(line: 8, column: 25, scope: !116)
// CHECK:STDOUT: !129 = !DILocation(line: 9, column: 25, scope: !116)
// CHECK:STDOUT: !133 = !DILocation(line: 10, column: 16, scope: !116)
// CHECK:STDOUT: !134 = !DILocation(line: 10, column: 10, scope: !116)
// CHECK:STDOUT: !135 = !DILocation(line: 10, column: 3, scope: !116)
// CHECK:STDOUT: !136 = !{!117}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "RefIdFromGeneric", linkageName: "_CRefIdFromGeneric.Main.55456748a8a2eb70", scope: null, file: !114, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !146)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !6)
// CHECK:STDOUT: !144 = !DILocation(line: 5, column: 58, scope: !137)
// CHECK:STDOUT: !145 = !DILocation(line: 5, column: 51, scope: !137)
// CHECK:STDOUT: !146 = !{!138}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "RefId", linkageName: "_CRefId.Main.55456748a8a2eb70", scope: null, file: !114, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !152)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !6)
// CHECK:STDOUT: !151 = !DILocation(line: 3, column: 43, scope: !147)
// CHECK:STDOUT: !152 = !{!148}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "ValId", linkageName: "_CValId.Main.55456748a8a2eb70", scope: null, file: !114, line: 4, type: !8, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !158)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !6)
// CHECK:STDOUT: !157 = !DILocation(line: 4, column: 39, scope: !153)
// CHECK:STDOUT: !158 = !{!154}
// CHECK:STDOUT:
