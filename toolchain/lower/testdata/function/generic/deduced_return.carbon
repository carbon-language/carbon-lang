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
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw i8, ptr %.loc8_36.3.temp, i64 0, !dbg !15
// CHECK:STDOUT:   %.loc8_36.5 = load i32, ptr %tuple.elem0.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw i8, ptr %.loc8_36.3.temp, i64 4, !dbg !15
// CHECK:STDOUT:   %.loc8_36.6 = load i32, ptr %tuple.elem1.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple), !dbg !15
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds nuw i8, ptr %tuple, i64 0, !dbg !15
// CHECK:STDOUT:   store i32 %.loc8_36.5, ptr %tuple1, align 4, !dbg !15
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds nuw i8, ptr %tuple, i64 4, !dbg !15
// CHECK:STDOUT:   store i32 %.loc8_36.6, ptr %tuple2, align 4, !dbg !15
// CHECK:STDOUT:   %IdFromGeneric.call = call i1 @_CIdFromGeneric.Main.1ac5a6b0e5910c46(i1 true), !dbg !21
// CHECK:STDOUT:   %Id.call = call i32 @_CId.Main.2fad9574e050607e(i32 %n), !dbg !27
// CHECK:STDOUT:   ret i32 %Id.call, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CPair.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw i8, ptr %return, i64 0, !dbg !33
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem0.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw i8, ptr %return, i64 4, !dbg !33
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem1.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CIdFromGeneric.Main.1ac5a6b0e5910c46(i1 %x) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Id.call = call i1 @_CId.Main.1ac5a6b0e5910c46(i1 %x), !dbg !44
// CHECK:STDOUT:   ret i1 %Id.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CId.Main.2fad9574e050607e(i32 %x) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CId.Main.1ac5a6b0e5910c46(i1 %x) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 %x, !dbg !57
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
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !4, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !9, type: !6)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 30, scope: !9)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!16, !6}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 24, scope: !9)
// CHECK:STDOUT: !22 = !{!16, !16}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 10, column: 10, scope: !9)
// CHECK:STDOUT: !28 = !DILocation(line: 10, column: 3, scope: !9)
// CHECK:STDOUT: !29 = !{!10}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Pair", linkageName: "_CPair.Main.2fad9574e050607e", scope: null, file: !4, line: 4, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !6)
// CHECK:STDOUT: !33 = !DILocation(line: 4, column: 46, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 4, column: 39, scope: !30)
// CHECK:STDOUT: !37 = !{!31}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "IdFromGeneric", linkageName: "_CIdFromGeneric.Main.1ac5a6b0e5910c46", scope: null, file: !4, line: 5, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !16)
// CHECK:STDOUT: !44 = !DILocation(line: 5, column: 52, scope: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 5, column: 45, scope: !38)
// CHECK:STDOUT: !46 = !{!39}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Id", linkageName: "_CId.Main.2fad9574e050607e", scope: null, file: !4, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !52)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !6)
// CHECK:STDOUT: !51 = !DILocation(line: 3, column: 37, scope: !47)
// CHECK:STDOUT: !52 = !{!48}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Id", linkageName: "_CId.Main.1ac5a6b0e5910c46", scope: null, file: !4, line: 3, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !16)
// CHECK:STDOUT: !57 = !DILocation(line: 3, column: 37, scope: !53)
// CHECK:STDOUT: !58 = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'template.carbon'
// CHECK:STDOUT: source_filename = "template.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallNegate.Main(i32 %a) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = call i32 @_CNegate.Main.55456748a8a2eb70(i32 %a), !dbg !67
// CHECK:STDOUT:   ret i32 %Negate.call, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CNegate.Main.55456748a8a2eb70(i32 %a) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !73
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!60}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !59 = !DIFile(filename: "template.carbon", directory: "")
// CHECK:STDOUT: !60 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !59, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "CallNegate", linkageName: "_CCallNegate.Main", scope: null, file: !59, line: 5, type: !8, spFlags: DISPFlagDefinition, unit: !60, retainedNodes: !69)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !6)
// CHECK:STDOUT: !67 = !DILocation(line: 5, column: 39, scope: !61)
// CHECK:STDOUT: !68 = !DILocation(line: 5, column: 32, scope: !61)
// CHECK:STDOUT: !69 = !{!62}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Negate", linkageName: "_CNegate.Main.55456748a8a2eb70", scope: null, file: !59, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !60, retainedNodes: !76)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !6)
// CHECK:STDOUT: !73 = !DILocation(line: 3, column: 52, scope: !70)
// CHECK:STDOUT: !75 = !DILocation(line: 3, column: 45, scope: !70)
// CHECK:STDOUT: !76 = !{!71}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!78}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !77 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !77, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_lib.carbon'
// CHECK:STDOUT: source_filename = "use_lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUse.Main(ptr sret({ i32, i32 }) %return, i32 %n) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CPair.Main.2fad9574e050607e(ptr %return, i32 %n), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CPair.Main.2fad9574e050607e(ptr sret({ i32, i32 }) %return, i32 %x) #0 !dbg !90 {
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds nuw i8, ptr %return, i64 0, !dbg !93
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem, align 4, !dbg !93
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds nuw i8, ptr %return, i64 4, !dbg !93
// CHECK:STDOUT:   store i32 %x, ptr %tuple.elem1, align 4, !dbg !93
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!80}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!16, !6}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !77 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !79 = !DIFile(filename: "use_lib.carbon", directory: "")
// CHECK:STDOUT: !80 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !79, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !79, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !89)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !6)
// CHECK:STDOUT: !87 = !DILocation(line: 5, column: 39, scope: !81)
// CHECK:STDOUT: !88 = !DILocation(line: 5, column: 32, scope: !81)
// CHECK:STDOUT: !89 = !{!82}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Pair", linkageName: "_CPair.Main.2fad9574e050607e", scope: null, file: !77, line: 3, type: !18, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !97)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !6)
// CHECK:STDOUT: !93 = !DILocation(line: 3, column: 46, scope: !90)
// CHECK:STDOUT: !96 = !DILocation(line: 3, column: 39, scope: !90)
// CHECK:STDOUT: !97 = !{!91}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'form.carbon'
// CHECK:STDOUT: source_filename = "form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main(ptr %n) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RefIdFromGeneric.call = call ptr @_CRefIdFromGeneric.Main.55456748a8a2eb70(ptr %n), !dbg !107
// CHECK:STDOUT:   %RefId.call = call ptr @_CRefId.Main.55456748a8a2eb70(ptr %n), !dbg !113
// CHECK:STDOUT:   %.loc10 = load i32, ptr %n, align 4, !dbg !117
// CHECK:STDOUT:   %ValId.call = call i32 @_CValId.Main.55456748a8a2eb70(i32 %.loc10), !dbg !118
// CHECK:STDOUT:   ret i32 %ValId.call, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CRefIdFromGeneric.Main.55456748a8a2eb70(ptr %x) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RefId.call = call ptr @_CRefId.Main.55456748a8a2eb70(ptr %x), !dbg !128
// CHECK:STDOUT:   ret ptr %RefId.call, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CRefId.Main.55456748a8a2eb70(ptr %x) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CValId.Main.55456748a8a2eb70(i32 %x) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CRefId.Main.55456748a8a2eb70, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!99}
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
// CHECK:STDOUT: !98 = !DIFile(filename: "form.carbon", directory: "")
// CHECK:STDOUT: !99 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !98, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !98, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !99, retainedNodes: !120)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !6)
// CHECK:STDOUT: !107 = !DILocation(line: 8, column: 25, scope: !100)
// CHECK:STDOUT: !113 = !DILocation(line: 9, column: 25, scope: !100)
// CHECK:STDOUT: !117 = !DILocation(line: 10, column: 16, scope: !100)
// CHECK:STDOUT: !118 = !DILocation(line: 10, column: 10, scope: !100)
// CHECK:STDOUT: !119 = !DILocation(line: 10, column: 3, scope: !100)
// CHECK:STDOUT: !120 = !{!101}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "RefIdFromGeneric", linkageName: "_CRefIdFromGeneric.Main.55456748a8a2eb70", scope: null, file: !98, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !99, retainedNodes: !130)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !6)
// CHECK:STDOUT: !128 = !DILocation(line: 5, column: 58, scope: !121)
// CHECK:STDOUT: !129 = !DILocation(line: 5, column: 51, scope: !121)
// CHECK:STDOUT: !130 = !{!122}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "RefId", linkageName: "_CRefId.Main.55456748a8a2eb70", scope: null, file: !98, line: 3, type: !18, spFlags: DISPFlagDefinition, unit: !99, retainedNodes: !136)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !6)
// CHECK:STDOUT: !135 = !DILocation(line: 3, column: 43, scope: !131)
// CHECK:STDOUT: !136 = !{!132}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "ValId", linkageName: "_CValId.Main.55456748a8a2eb70", scope: null, file: !98, line: 4, type: !8, spFlags: DISPFlagDefinition, unit: !99, retainedNodes: !142)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !6)
// CHECK:STDOUT: !141 = !DILocation(line: 4, column: 39, scope: !137)
// CHECK:STDOUT: !142 = !{!138}
// CHECK:STDOUT:
