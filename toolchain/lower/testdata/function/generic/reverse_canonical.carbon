// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/default.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/reverse_canonical.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/reverse_canonical.carbon

// --- core.carbon

package Core library "[[@TEST_NAME]]";

export import library "prelude/parts/default";

// --- reverse_canonical.carbon

import Core library "core";

class C1 {}

fn first[T: type, U: type, V: type](arg1: T, arg2: U, arg3: V);
fn second[T: type, U: type, V: type](arg1: T, arg2: U, arg3: V);
fn third[T: type, U: type, V: type](arg1: T, arg2: U, arg3: V);
fn fourth[T: type, U: type, V: type](arg1: T, arg2: U, arg3: V);

fn first[T: type, U: type, V: type](unused arg1: T, arg2: U, unused arg3: V) {
  var local_name: C1*;

  second(arg2, arg2, local_name);
}

fn second[T: type, U: type, V: type](arg1: T, arg2: U, arg3: V) {
  var local_name: C1*;

  first(arg3, arg3, arg1);
  third(arg2, arg2, local_name);
}

fn third[T: type, U: type, V: type](arg1: T, arg2: U, arg3: V) {
  var local_name: C1**;

  fourth(local_name, local_name, arg1);
  fourth(arg3, arg3, arg2);
}

fn fourth[T: type, U: type, V: type](unused arg1: T, unused arg2: U, unused arg3: V) {
}

fn Main() {
  var x: bool;
  var y: bool*;

  first(x, x, y);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'core.carbon'
// CHECK:STDOUT: source_filename = "core.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "core.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'reverse_canonical.carbon'
// CHECK:STDOUT: source_filename = "reverse_canonical.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i1, align 1, !dbg !23
// CHECK:STDOUT:   %y.var = alloca ptr, align 8, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !23
// CHECK:STDOUT:   store i8 poison, ptr %x.var, align 1, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !25
// CHECK:STDOUT:   store ptr poison, ptr %y.var, align 8, !dbg !25
// CHECK:STDOUT:   %.loc38_9 = load i8, ptr %x.var, align 1, !dbg !27
// CHECK:STDOUT:   %.loc38_91 = trunc i8 %.loc38_9 to i1, !dbg !27
// CHECK:STDOUT:   %.loc38_12 = load i8, ptr %x.var, align 1, !dbg !28
// CHECK:STDOUT:   %.loc38_122 = trunc i8 %.loc38_12 to i1, !dbg !28
// CHECK:STDOUT:   %.loc38_15 = load ptr, ptr %y.var, align 8, !dbg !29
// CHECK:STDOUT:   call void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %.loc38_91, i1 %.loc38_122, ptr %.loc38_15), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !42
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !42
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !46
// CHECK:STDOUT:   call void @_Csecond.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc14), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !57
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !57
// CHECK:STDOUT:   call void @_Cfirst.Main.81f1c9de445b895c(ptr %arg3, ptr %arg3, i1 %arg1), !dbg !62
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !65
// CHECK:STDOUT:   call void @_Cthird.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc21), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.81f1c9de445b895c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !76
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !76
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !80
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc14), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !91
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !91
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !91
// CHECK:STDOUT:   %.loc27_10 = load ptr, ptr %local_name.var, align 8, !dbg !93
// CHECK:STDOUT:   %.loc27_22 = load ptr, ptr %local_name.var, align 8, !dbg !94
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %.loc27_10, ptr %.loc27_22, i1 %arg1), !dbg !96
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !110
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !110
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !110
// CHECK:STDOUT:   call void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !115
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !118
// CHECK:STDOUT:   call void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc21), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !140 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !147
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !147
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !147
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !151
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc14), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !162
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !162
// CHECK:STDOUT:   %.loc27_10 = load ptr, ptr %local_name.var, align 8, !dbg !164
// CHECK:STDOUT:   %.loc27_22 = load ptr, ptr %local_name.var, align 8, !dbg !165
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %.loc27_10, ptr %.loc27_22, ptr %arg1), !dbg !167
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 8, 7 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.c6787c0e2572c86c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.901b24db5c2cfe42, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "reverse_canonical.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !18, line: 34, type: !21, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 35, column: 3, scope: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 36, column: 3, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 38, column: 9, scope: !22)
// CHECK:STDOUT: !28 = !DILocation(line: 38, column: 12, scope: !22)
// CHECK:STDOUT: !29 = !DILocation(line: 38, column: 15, scope: !22)
// CHECK:STDOUT: !30 = !DILocation(line: 38, column: 3, scope: !22)
// CHECK:STDOUT: !31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !32 = !{null, !31, !31, !31}
// CHECK:STDOUT: !33 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 34, column: 1, scope: !22)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !18, line: 11, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !49)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !31)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 2, scope: !35, type: !31)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 3, scope: !35, type: !31)
// CHECK:STDOUT: !42 = !DILocation(line: 12, column: 3, scope: !35)
// CHECK:STDOUT: !46 = !DILocation(line: 14, column: 22, scope: !35)
// CHECK:STDOUT: !47 = !DILocation(line: 14, column: 3, scope: !35)
// CHECK:STDOUT: !48 = !DILocation(line: 11, column: 1, scope: !35)
// CHECK:STDOUT: !49 = !{!36, !37, !38}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.0f29a29bf45af723", scope: null, file: !18, line: 17, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !68)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !31)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !50, type: !31)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 3, scope: !50, type: !31)
// CHECK:STDOUT: !57 = !DILocation(line: 18, column: 3, scope: !50)
// CHECK:STDOUT: !62 = !DILocation(line: 20, column: 3, scope: !50)
// CHECK:STDOUT: !65 = !DILocation(line: 21, column: 21, scope: !50)
// CHECK:STDOUT: !66 = !DILocation(line: 21, column: 3, scope: !50)
// CHECK:STDOUT: !67 = !DILocation(line: 17, column: 1, scope: !50)
// CHECK:STDOUT: !68 = !{!51, !52, !53}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.81f1c9de445b895c", scope: null, file: !18, line: 11, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !83)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !31)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 2, scope: !69, type: !31)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 3, scope: !69, type: !31)
// CHECK:STDOUT: !76 = !DILocation(line: 12, column: 3, scope: !69)
// CHECK:STDOUT: !80 = !DILocation(line: 14, column: 22, scope: !69)
// CHECK:STDOUT: !81 = !DILocation(line: 14, column: 3, scope: !69)
// CHECK:STDOUT: !82 = !DILocation(line: 11, column: 1, scope: !69)
// CHECK:STDOUT: !83 = !{!70, !71, !72}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.0f29a29bf45af723", scope: null, file: !18, line: 24, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !102)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !31)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 2, scope: !84, type: !31)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 3, scope: !84, type: !31)
// CHECK:STDOUT: !91 = !DILocation(line: 25, column: 3, scope: !84)
// CHECK:STDOUT: !93 = !DILocation(line: 27, column: 10, scope: !84)
// CHECK:STDOUT: !94 = !DILocation(line: 27, column: 22, scope: !84)
// CHECK:STDOUT: !96 = !DILocation(line: 27, column: 3, scope: !84)
// CHECK:STDOUT: !100 = !DILocation(line: 28, column: 3, scope: !84)
// CHECK:STDOUT: !101 = !DILocation(line: 24, column: 1, scope: !84)
// CHECK:STDOUT: !102 = !{!85, !86, !87}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.e1ce67f3c23ab61f", scope: null, file: !18, line: 17, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !121)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !31)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 2, scope: !103, type: !31)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 3, scope: !103, type: !31)
// CHECK:STDOUT: !110 = !DILocation(line: 18, column: 3, scope: !103)
// CHECK:STDOUT: !115 = !DILocation(line: 20, column: 3, scope: !103)
// CHECK:STDOUT: !118 = !DILocation(line: 21, column: 21, scope: !103)
// CHECK:STDOUT: !119 = !DILocation(line: 21, column: 3, scope: !103)
// CHECK:STDOUT: !120 = !DILocation(line: 17, column: 1, scope: !103)
// CHECK:STDOUT: !121 = !{!104, !105, !106}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.c6787c0e2572c86c", scope: null, file: !18, line: 31, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !130)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !31)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !122, type: !31)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 3, scope: !122, type: !31)
// CHECK:STDOUT: !129 = !DILocation(line: 31, column: 1, scope: !122)
// CHECK:STDOUT: !130 = !{!123, !124, !125}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.e1ce67f3c23ab61f", scope: null, file: !18, line: 11, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !154)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !31)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 2, scope: !140, type: !31)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 3, scope: !140, type: !31)
// CHECK:STDOUT: !147 = !DILocation(line: 12, column: 3, scope: !140)
// CHECK:STDOUT: !151 = !DILocation(line: 14, column: 22, scope: !140)
// CHECK:STDOUT: !152 = !DILocation(line: 14, column: 3, scope: !140)
// CHECK:STDOUT: !153 = !DILocation(line: 11, column: 1, scope: !140)
// CHECK:STDOUT: !154 = !{!141, !142, !143}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.e1ce67f3c23ab61f", scope: null, file: !18, line: 24, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !173)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !31)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 2, scope: !155, type: !31)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 3, scope: !155, type: !31)
// CHECK:STDOUT: !162 = !DILocation(line: 25, column: 3, scope: !155)
// CHECK:STDOUT: !164 = !DILocation(line: 27, column: 10, scope: !155)
// CHECK:STDOUT: !165 = !DILocation(line: 27, column: 22, scope: !155)
// CHECK:STDOUT: !167 = !DILocation(line: 27, column: 3, scope: !155)
// CHECK:STDOUT: !171 = !DILocation(line: 28, column: 3, scope: !155)
// CHECK:STDOUT: !172 = !DILocation(line: 24, column: 1, scope: !155)
// CHECK:STDOUT: !173 = !{!156, !157, !158}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.901b24db5c2cfe42", scope: null, file: !18, line: 31, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !182)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !31)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 2, scope: !174, type: !31)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 3, scope: !174, type: !31)
// CHECK:STDOUT: !181 = !DILocation(line: 31, column: 1, scope: !174)
// CHECK:STDOUT: !182 = !{!175, !176, !177}
// CHECK:STDOUT:
