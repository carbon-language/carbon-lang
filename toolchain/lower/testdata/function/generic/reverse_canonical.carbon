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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4055ca6f11c20235(ptr %y.var), !dbg !25
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.27eb7fea99346bf0(ptr %x.var), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !44
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !44
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !48
// CHECK:STDOUT:   call void @_Csecond.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc14), !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.45935e46e8ff455b(ptr %local_name.var), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4055ca6f11c20235(ptr %t) #0 !dbg !52 {
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.27eb7fea99346bf0(ptr %t) #0 !dbg !59 {
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !73
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !73
// CHECK:STDOUT:   call void @_Cfirst.Main.81f1c9de445b895c(ptr %arg3, ptr %arg3, i1 %arg1), !dbg !78
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !81
// CHECK:STDOUT:   call void @_Cthird.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc21), !dbg !82
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.45935e46e8ff455b(ptr %local_name.var), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.45935e46e8ff455b(ptr %t) #0 !dbg !85 {
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.81f1c9de445b895c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !99
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !99
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !99
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !103
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc14), !dbg !104
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.45935e46e8ff455b(ptr %local_name.var), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !114
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !114
// CHECK:STDOUT:   %.loc27_10 = load ptr, ptr %local_name.var, align 8, !dbg !116
// CHECK:STDOUT:   %.loc27_22 = load ptr, ptr %local_name.var, align 8, !dbg !117
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %.loc27_10, ptr %.loc27_22, i1 %arg1), !dbg !119
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !123
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.123e1a22f3f15b38(ptr %local_name.var), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !133
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !133
// CHECK:STDOUT:   call void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !138
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !141
// CHECK:STDOUT:   call void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc21), !dbg !142
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.45935e46e8ff455b(ptr %local_name.var), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.123e1a22f3f15b38(ptr %t) #0 !dbg !163 {
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !177
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !177
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !177
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !181
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc14), !dbg !182
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.45935e46e8ff455b(ptr %local_name.var), !dbg !177
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !192
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !192
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !192
// CHECK:STDOUT:   %.loc27_10 = load ptr, ptr %local_name.var, align 8, !dbg !194
// CHECK:STDOUT:   %.loc27_22 = load ptr, ptr %local_name.var, align 8, !dbg !195
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %.loc27_10, ptr %.loc27_22, ptr %arg1), !dbg !197
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !201
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.123e1a22f3f15b38(ptr %local_name.var), !dbg !192
// CHECK:STDOUT:   ret void, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !204 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 8, 7 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.45935e46e8ff455b, { 0, 1, 2, 4, 3 }
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
// CHECK:STDOUT: !8 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !34 = !{null, !31}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 34, column: 1, scope: !22)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !18, line: 11, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !51)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !31)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !37, type: !31)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 3, scope: !37, type: !31)
// CHECK:STDOUT: !44 = !DILocation(line: 12, column: 3, scope: !37)
// CHECK:STDOUT: !48 = !DILocation(line: 14, column: 22, scope: !37)
// CHECK:STDOUT: !49 = !DILocation(line: 14, column: 3, scope: !37)
// CHECK:STDOUT: !50 = !DILocation(line: 11, column: 1, scope: !37)
// CHECK:STDOUT: !51 = !{!38, !39, !40}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4055ca6f11c20235", scope: null, file: !8, line: 30, type: !35, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !58)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !31)
// CHECK:STDOUT: !57 = !DILocation(line: 30, column: 1, scope: !52)
// CHECK:STDOUT: !58 = !{!53}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.27eb7fea99346bf0", scope: null, file: !8, line: 30, type: !35, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !31)
// CHECK:STDOUT: !64 = !DILocation(line: 30, column: 1, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.0f29a29bf45af723", scope: null, file: !18, line: 17, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !84)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !31)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 2, scope: !66, type: !31)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 3, scope: !66, type: !31)
// CHECK:STDOUT: !73 = !DILocation(line: 18, column: 3, scope: !66)
// CHECK:STDOUT: !78 = !DILocation(line: 20, column: 3, scope: !66)
// CHECK:STDOUT: !81 = !DILocation(line: 21, column: 21, scope: !66)
// CHECK:STDOUT: !82 = !DILocation(line: 21, column: 3, scope: !66)
// CHECK:STDOUT: !83 = !DILocation(line: 17, column: 1, scope: !66)
// CHECK:STDOUT: !84 = !{!67, !68, !69}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.45935e46e8ff455b", scope: null, file: !8, line: 30, type: !35, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !91)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !31)
// CHECK:STDOUT: !90 = !DILocation(line: 30, column: 1, scope: !85)
// CHECK:STDOUT: !91 = !{!86}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.81f1c9de445b895c", scope: null, file: !18, line: 11, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !106)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !31)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 2, scope: !92, type: !31)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 3, scope: !92, type: !31)
// CHECK:STDOUT: !99 = !DILocation(line: 12, column: 3, scope: !92)
// CHECK:STDOUT: !103 = !DILocation(line: 14, column: 22, scope: !92)
// CHECK:STDOUT: !104 = !DILocation(line: 14, column: 3, scope: !92)
// CHECK:STDOUT: !105 = !DILocation(line: 11, column: 1, scope: !92)
// CHECK:STDOUT: !106 = !{!93, !94, !95}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.0f29a29bf45af723", scope: null, file: !18, line: 24, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !125)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !31)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 2, scope: !107, type: !31)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 3, scope: !107, type: !31)
// CHECK:STDOUT: !114 = !DILocation(line: 25, column: 3, scope: !107)
// CHECK:STDOUT: !116 = !DILocation(line: 27, column: 10, scope: !107)
// CHECK:STDOUT: !117 = !DILocation(line: 27, column: 22, scope: !107)
// CHECK:STDOUT: !119 = !DILocation(line: 27, column: 3, scope: !107)
// CHECK:STDOUT: !123 = !DILocation(line: 28, column: 3, scope: !107)
// CHECK:STDOUT: !124 = !DILocation(line: 24, column: 1, scope: !107)
// CHECK:STDOUT: !125 = !{!108, !109, !110}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.e1ce67f3c23ab61f", scope: null, file: !18, line: 17, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !144)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !31)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 2, scope: !126, type: !31)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 3, scope: !126, type: !31)
// CHECK:STDOUT: !133 = !DILocation(line: 18, column: 3, scope: !126)
// CHECK:STDOUT: !138 = !DILocation(line: 20, column: 3, scope: !126)
// CHECK:STDOUT: !141 = !DILocation(line: 21, column: 21, scope: !126)
// CHECK:STDOUT: !142 = !DILocation(line: 21, column: 3, scope: !126)
// CHECK:STDOUT: !143 = !DILocation(line: 17, column: 1, scope: !126)
// CHECK:STDOUT: !144 = !{!127, !128, !129}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.c6787c0e2572c86c", scope: null, file: !18, line: 31, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !153)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !31)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 2, scope: !145, type: !31)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 3, scope: !145, type: !31)
// CHECK:STDOUT: !152 = !DILocation(line: 31, column: 1, scope: !145)
// CHECK:STDOUT: !153 = !{!146, !147, !148}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.123e1a22f3f15b38", scope: null, file: !8, line: 30, type: !35, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !169)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !31)
// CHECK:STDOUT: !168 = !DILocation(line: 30, column: 1, scope: !163)
// CHECK:STDOUT: !169 = !{!164}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.e1ce67f3c23ab61f", scope: null, file: !18, line: 11, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !184)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !31)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 2, scope: !170, type: !31)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 3, scope: !170, type: !31)
// CHECK:STDOUT: !177 = !DILocation(line: 12, column: 3, scope: !170)
// CHECK:STDOUT: !181 = !DILocation(line: 14, column: 22, scope: !170)
// CHECK:STDOUT: !182 = !DILocation(line: 14, column: 3, scope: !170)
// CHECK:STDOUT: !183 = !DILocation(line: 11, column: 1, scope: !170)
// CHECK:STDOUT: !184 = !{!171, !172, !173}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.e1ce67f3c23ab61f", scope: null, file: !18, line: 24, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !203)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !31)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 2, scope: !185, type: !31)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 3, scope: !185, type: !31)
// CHECK:STDOUT: !192 = !DILocation(line: 25, column: 3, scope: !185)
// CHECK:STDOUT: !194 = !DILocation(line: 27, column: 10, scope: !185)
// CHECK:STDOUT: !195 = !DILocation(line: 27, column: 22, scope: !185)
// CHECK:STDOUT: !197 = !DILocation(line: 27, column: 3, scope: !185)
// CHECK:STDOUT: !201 = !DILocation(line: 28, column: 3, scope: !185)
// CHECK:STDOUT: !202 = !DILocation(line: 24, column: 1, scope: !185)
// CHECK:STDOUT: !203 = !{!186, !187, !188}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.901b24db5c2cfe42", scope: null, file: !18, line: 31, type: !33, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !212)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !204, type: !31)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 2, scope: !204, type: !31)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 3, scope: !204, type: !31)
// CHECK:STDOUT: !211 = !DILocation(line: 31, column: 1, scope: !204)
// CHECK:STDOUT: !212 = !{!205, !206, !207}
// CHECK:STDOUT:
