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
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "core.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'reverse_canonical.carbon'
// CHECK:STDOUT: source_filename = "reverse_canonical.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i1, align 1, !dbg !25
// CHECK:STDOUT:   %y.var = alloca ptr, align 8, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !25
// CHECK:STDOUT:   store i8 poison, ptr %x.var, align 1, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !27
// CHECK:STDOUT:   store ptr poison, ptr %y.var, align 8, !dbg !27
// CHECK:STDOUT:   %.loc38_9 = load i8, ptr %x.var, align 1, !dbg !29
// CHECK:STDOUT:   %.loc38_91 = trunc i8 %.loc38_9 to i1, !dbg !29
// CHECK:STDOUT:   %.loc38_12 = load i8, ptr %x.var, align 1, !dbg !30
// CHECK:STDOUT:   %.loc38_122 = trunc i8 %.loc38_12 to i1, !dbg !30
// CHECK:STDOUT:   %.loc38_15 = load ptr, ptr %y.var, align 8, !dbg !31
// CHECK:STDOUT:   call void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %.loc38_91, i1 %.loc38_122, ptr %.loc38_15), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.393ac80c0c8a44df(ptr %y.var), !dbg !27
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %x.var), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !46
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !46
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !50
// CHECK:STDOUT:   call void @_Csecond.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc14), !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a297d85f341e5034(ptr %local_name.var), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.393ac80c0c8a44df(ptr %t) #0 !dbg !54 {
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %t) #0 !dbg !61 {
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !75
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !75
// CHECK:STDOUT:   call void @_Cfirst.Main.81f1c9de445b895c(ptr %arg3, ptr %arg3, i1 %arg1), !dbg !80
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !83
// CHECK:STDOUT:   call void @_Cthird.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc21), !dbg !84
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a297d85f341e5034(ptr %local_name.var), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a297d85f341e5034(ptr %t) #0 !dbg !87 {
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.81f1c9de445b895c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !101
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !101
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !101
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !105
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc14), !dbg !106
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a297d85f341e5034(ptr %local_name.var), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !116
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !116
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !116
// CHECK:STDOUT:   %.loc27_10 = load ptr, ptr %local_name.var, align 8, !dbg !118
// CHECK:STDOUT:   %.loc27_22 = load ptr, ptr %local_name.var, align 8, !dbg !119
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %.loc27_10, ptr %.loc27_22, i1 %arg1), !dbg !121
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !125
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3490326134093774(ptr %local_name.var), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !135
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !135
// CHECK:STDOUT:   call void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !140
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !143
// CHECK:STDOUT:   call void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc21), !dbg !144
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a297d85f341e5034(ptr %local_name.var), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3490326134093774(ptr %t) #0 !dbg !165 {
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !179
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !179
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !179
// CHECK:STDOUT:   %.loc14 = load ptr, ptr %local_name.var, align 8, !dbg !183
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc14), !dbg !184
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a297d85f341e5034(ptr %local_name.var), !dbg !179
// CHECK:STDOUT:   ret void, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !187 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !194
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !194
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !194
// CHECK:STDOUT:   %.loc27_10 = load ptr, ptr %local_name.var, align 8, !dbg !196
// CHECK:STDOUT:   %.loc27_22 = load ptr, ptr %local_name.var, align 8, !dbg !197
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %.loc27_10, ptr %.loc27_22, ptr %arg1), !dbg !199
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !203
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3490326134093774(ptr %local_name.var), !dbg !194
// CHECK:STDOUT:   ret void, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !206 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 8, 7 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.a297d85f341e5034, { 0, 1, 2, 4, 3 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.c6787c0e2572c86c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.901b24db5c2cfe42, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !8 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIFile(filename: "reverse_canonical.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !20, line: 34, type: !23, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 35, column: 3, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 36, column: 3, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 38, column: 9, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 38, column: 12, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 38, column: 15, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 38, column: 3, scope: !24)
// CHECK:STDOUT: !33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !34 = !{null, !33, !33, !33}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !{null, !33}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 34, column: 1, scope: !24)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !20, line: 11, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !53)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !33)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 2, scope: !39, type: !33)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 3, scope: !39, type: !33)
// CHECK:STDOUT: !46 = !DILocation(line: 12, column: 3, scope: !39)
// CHECK:STDOUT: !50 = !DILocation(line: 14, column: 22, scope: !39)
// CHECK:STDOUT: !51 = !DILocation(line: 14, column: 3, scope: !39)
// CHECK:STDOUT: !52 = !DILocation(line: 11, column: 1, scope: !39)
// CHECK:STDOUT: !53 = !{!40, !41, !42}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.393ac80c0c8a44df", scope: null, file: !8, line: 32, type: !37, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !60)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !33)
// CHECK:STDOUT: !59 = !DILocation(line: 32, column: 1, scope: !54)
// CHECK:STDOUT: !60 = !{!55}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.29e6aa423646b61a", scope: null, file: !8, line: 32, type: !37, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !67)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !33)
// CHECK:STDOUT: !66 = !DILocation(line: 32, column: 1, scope: !61)
// CHECK:STDOUT: !67 = !{!62}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.0f29a29bf45af723", scope: null, file: !20, line: 17, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !86)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !33)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 2, scope: !68, type: !33)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 3, scope: !68, type: !33)
// CHECK:STDOUT: !75 = !DILocation(line: 18, column: 3, scope: !68)
// CHECK:STDOUT: !80 = !DILocation(line: 20, column: 3, scope: !68)
// CHECK:STDOUT: !83 = !DILocation(line: 21, column: 21, scope: !68)
// CHECK:STDOUT: !84 = !DILocation(line: 21, column: 3, scope: !68)
// CHECK:STDOUT: !85 = !DILocation(line: 17, column: 1, scope: !68)
// CHECK:STDOUT: !86 = !{!69, !70, !71}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a297d85f341e5034", scope: null, file: !8, line: 32, type: !37, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !93)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !33)
// CHECK:STDOUT: !92 = !DILocation(line: 32, column: 1, scope: !87)
// CHECK:STDOUT: !93 = !{!88}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.81f1c9de445b895c", scope: null, file: !20, line: 11, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !108)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !33)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 2, scope: !94, type: !33)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 3, scope: !94, type: !33)
// CHECK:STDOUT: !101 = !DILocation(line: 12, column: 3, scope: !94)
// CHECK:STDOUT: !105 = !DILocation(line: 14, column: 22, scope: !94)
// CHECK:STDOUT: !106 = !DILocation(line: 14, column: 3, scope: !94)
// CHECK:STDOUT: !107 = !DILocation(line: 11, column: 1, scope: !94)
// CHECK:STDOUT: !108 = !{!95, !96, !97}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.0f29a29bf45af723", scope: null, file: !20, line: 24, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !127)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !33)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 2, scope: !109, type: !33)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 3, scope: !109, type: !33)
// CHECK:STDOUT: !116 = !DILocation(line: 25, column: 3, scope: !109)
// CHECK:STDOUT: !118 = !DILocation(line: 27, column: 10, scope: !109)
// CHECK:STDOUT: !119 = !DILocation(line: 27, column: 22, scope: !109)
// CHECK:STDOUT: !121 = !DILocation(line: 27, column: 3, scope: !109)
// CHECK:STDOUT: !125 = !DILocation(line: 28, column: 3, scope: !109)
// CHECK:STDOUT: !126 = !DILocation(line: 24, column: 1, scope: !109)
// CHECK:STDOUT: !127 = !{!110, !111, !112}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.e1ce67f3c23ab61f", scope: null, file: !20, line: 17, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !146)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !33)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 2, scope: !128, type: !33)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 3, scope: !128, type: !33)
// CHECK:STDOUT: !135 = !DILocation(line: 18, column: 3, scope: !128)
// CHECK:STDOUT: !140 = !DILocation(line: 20, column: 3, scope: !128)
// CHECK:STDOUT: !143 = !DILocation(line: 21, column: 21, scope: !128)
// CHECK:STDOUT: !144 = !DILocation(line: 21, column: 3, scope: !128)
// CHECK:STDOUT: !145 = !DILocation(line: 17, column: 1, scope: !128)
// CHECK:STDOUT: !146 = !{!129, !130, !131}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.c6787c0e2572c86c", scope: null, file: !20, line: 31, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !155)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !33)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 2, scope: !147, type: !33)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 3, scope: !147, type: !33)
// CHECK:STDOUT: !154 = !DILocation(line: 31, column: 1, scope: !147)
// CHECK:STDOUT: !155 = !{!148, !149, !150}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3490326134093774", scope: null, file: !8, line: 32, type: !37, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !171)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !33)
// CHECK:STDOUT: !170 = !DILocation(line: 32, column: 1, scope: !165)
// CHECK:STDOUT: !171 = !{!166}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.e1ce67f3c23ab61f", scope: null, file: !20, line: 11, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !186)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !33)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 2, scope: !172, type: !33)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 3, scope: !172, type: !33)
// CHECK:STDOUT: !179 = !DILocation(line: 12, column: 3, scope: !172)
// CHECK:STDOUT: !183 = !DILocation(line: 14, column: 22, scope: !172)
// CHECK:STDOUT: !184 = !DILocation(line: 14, column: 3, scope: !172)
// CHECK:STDOUT: !185 = !DILocation(line: 11, column: 1, scope: !172)
// CHECK:STDOUT: !186 = !{!173, !174, !175}
// CHECK:STDOUT: !187 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.e1ce67f3c23ab61f", scope: null, file: !20, line: 24, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !205)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 1, scope: !187, type: !33)
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 2, scope: !187, type: !33)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 3, scope: !187, type: !33)
// CHECK:STDOUT: !194 = !DILocation(line: 25, column: 3, scope: !187)
// CHECK:STDOUT: !196 = !DILocation(line: 27, column: 10, scope: !187)
// CHECK:STDOUT: !197 = !DILocation(line: 27, column: 22, scope: !187)
// CHECK:STDOUT: !199 = !DILocation(line: 27, column: 3, scope: !187)
// CHECK:STDOUT: !203 = !DILocation(line: 28, column: 3, scope: !187)
// CHECK:STDOUT: !204 = !DILocation(line: 24, column: 1, scope: !187)
// CHECK:STDOUT: !205 = !{!188, !189, !190}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.901b24db5c2cfe42", scope: null, file: !20, line: 31, type: !35, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !214)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !33)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 2, scope: !206, type: !33)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 3, scope: !206, type: !33)
// CHECK:STDOUT: !213 = !DILocation(line: 31, column: 1, scope: !206)
// CHECK:STDOUT: !214 = !{!207, !208, !209}
// CHECK:STDOUT:
