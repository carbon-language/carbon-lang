// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/reverse_canonical.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/reverse_canonical.carbon

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
// CHECK:STDOUT: ; ModuleID = 'reverse_canonical.carbon'
// CHECK:STDOUT: source_filename = "reverse_canonical.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i1, align 1, !dbg !21
// CHECK:STDOUT:   %y.var = alloca ptr, align 8, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !21
// CHECK:STDOUT:   store i8 poison, ptr %x.var, align 1, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !23
// CHECK:STDOUT:   store ptr poison, ptr %y.var, align 8, !dbg !23
// CHECK:STDOUT:   %.loc47_9 = load i8, ptr %x.var, align 1, !dbg !25
// CHECK:STDOUT:   %.loc47_91 = trunc i8 %.loc47_9 to i1, !dbg !25
// CHECK:STDOUT:   %.loc47_12 = load i8, ptr %x.var, align 1, !dbg !26
// CHECK:STDOUT:   %.loc47_122 = trunc i8 %.loc47_12 to i1, !dbg !26
// CHECK:STDOUT:   %.loc47_15 = load ptr, ptr %y.var, align 8, !dbg !27
// CHECK:STDOUT:   call void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %.loc47_91, i1 %.loc47_122, ptr %.loc47_15), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.4d8205a9809188d0(ptr %y.var), !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad5376756201bea4(ptr %x.var), !dbg !21
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
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !46
// CHECK:STDOUT:   call void @_Csecond.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc23), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.4d8205a9809188d0(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ad5376756201bea4(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !71
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !71
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !71
// CHECK:STDOUT:   call void @_Cfirst.Main.81f1c9de445b895c(ptr %arg3, ptr %arg3, i1 %arg1), !dbg !76
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !79
// CHECK:STDOUT:   call void @_Cthird.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc30), !dbg !80
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.81f1c9de445b895c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !97
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !97
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !97
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !101
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !102
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !112
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !112
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !112
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !114
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !115
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %.loc36_10, ptr %.loc36_22, i1 %arg1), !dbg !117
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !121
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.30c57425aa4cd11c(ptr %local_name.var), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !131
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !131
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !131
// CHECK:STDOUT:   call void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !136
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !139
// CHECK:STDOUT:   call void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc30), !dbg !140
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.30c57425aa4cd11c(ptr %self) #0 !dbg !161 {
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !175
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !175
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !175
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !179
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !180
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !175
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !190
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !190
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !192
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !193
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %.loc36_10, ptr %.loc36_22, ptr %arg1), !dbg !195
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !199
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.30c57425aa4cd11c(ptr %local_name.var), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !202 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 8, 7 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.00ee02cee901e22f, { 0, 1, 2, 4, 3 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.c6787c0e2572c86c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.901b24db5c2cfe42, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "reverse_canonical.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !16, line: 43, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 44, column: 3, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 45, column: 3, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 47, column: 9, scope: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 47, column: 12, scope: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 47, column: 15, scope: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 47, column: 3, scope: !20)
// CHECK:STDOUT: !29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !30 = !{null, !29, !29, !29}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = !{null, !29}
// CHECK:STDOUT: !33 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 43, column: 1, scope: !20)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !16, line: 20, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !49)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !29)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 2, scope: !35, type: !29)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 3, scope: !35, type: !29)
// CHECK:STDOUT: !42 = !DILocation(line: 21, column: 3, scope: !35)
// CHECK:STDOUT: !46 = !DILocation(line: 23, column: 22, scope: !35)
// CHECK:STDOUT: !47 = !DILocation(line: 23, column: 3, scope: !35)
// CHECK:STDOUT: !48 = !DILocation(line: 20, column: 1, scope: !35)
// CHECK:STDOUT: !49 = !{!36, !37, !38}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.4d8205a9809188d0", scope: null, file: !10, line: 24, type: !33, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !56)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !29)
// CHECK:STDOUT: !55 = !DILocation(line: 24, column: 3, scope: !50)
// CHECK:STDOUT: !56 = !{!51}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ad5376756201bea4", scope: null, file: !10, line: 24, type: !33, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !63)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !29)
// CHECK:STDOUT: !62 = !DILocation(line: 24, column: 3, scope: !57)
// CHECK:STDOUT: !63 = !{!58}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.0f29a29bf45af723", scope: null, file: !16, line: 26, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !82)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !29)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 2, scope: !64, type: !29)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 3, scope: !64, type: !29)
// CHECK:STDOUT: !71 = !DILocation(line: 27, column: 3, scope: !64)
// CHECK:STDOUT: !76 = !DILocation(line: 29, column: 3, scope: !64)
// CHECK:STDOUT: !79 = !DILocation(line: 30, column: 21, scope: !64)
// CHECK:STDOUT: !80 = !DILocation(line: 30, column: 3, scope: !64)
// CHECK:STDOUT: !81 = !DILocation(line: 26, column: 1, scope: !64)
// CHECK:STDOUT: !82 = !{!65, !66, !67}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.00ee02cee901e22f", scope: null, file: !10, line: 24, type: !33, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !89)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !29)
// CHECK:STDOUT: !88 = !DILocation(line: 24, column: 3, scope: !83)
// CHECK:STDOUT: !89 = !{!84}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.81f1c9de445b895c", scope: null, file: !16, line: 20, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !104)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !29)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 2, scope: !90, type: !29)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 3, scope: !90, type: !29)
// CHECK:STDOUT: !97 = !DILocation(line: 21, column: 3, scope: !90)
// CHECK:STDOUT: !101 = !DILocation(line: 23, column: 22, scope: !90)
// CHECK:STDOUT: !102 = !DILocation(line: 23, column: 3, scope: !90)
// CHECK:STDOUT: !103 = !DILocation(line: 20, column: 1, scope: !90)
// CHECK:STDOUT: !104 = !{!91, !92, !93}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.0f29a29bf45af723", scope: null, file: !16, line: 33, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !123)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !29)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 2, scope: !105, type: !29)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 3, scope: !105, type: !29)
// CHECK:STDOUT: !112 = !DILocation(line: 34, column: 3, scope: !105)
// CHECK:STDOUT: !114 = !DILocation(line: 36, column: 10, scope: !105)
// CHECK:STDOUT: !115 = !DILocation(line: 36, column: 22, scope: !105)
// CHECK:STDOUT: !117 = !DILocation(line: 36, column: 3, scope: !105)
// CHECK:STDOUT: !121 = !DILocation(line: 37, column: 3, scope: !105)
// CHECK:STDOUT: !122 = !DILocation(line: 33, column: 1, scope: !105)
// CHECK:STDOUT: !123 = !{!106, !107, !108}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 26, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !142)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !29)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 2, scope: !124, type: !29)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 3, scope: !124, type: !29)
// CHECK:STDOUT: !131 = !DILocation(line: 27, column: 3, scope: !124)
// CHECK:STDOUT: !136 = !DILocation(line: 29, column: 3, scope: !124)
// CHECK:STDOUT: !139 = !DILocation(line: 30, column: 21, scope: !124)
// CHECK:STDOUT: !140 = !DILocation(line: 30, column: 3, scope: !124)
// CHECK:STDOUT: !141 = !DILocation(line: 26, column: 1, scope: !124)
// CHECK:STDOUT: !142 = !{!125, !126, !127}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.c6787c0e2572c86c", scope: null, file: !16, line: 40, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !151)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !29)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 2, scope: !143, type: !29)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 3, scope: !143, type: !29)
// CHECK:STDOUT: !150 = !DILocation(line: 40, column: 1, scope: !143)
// CHECK:STDOUT: !151 = !{!144, !145, !146}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.30c57425aa4cd11c", scope: null, file: !10, line: 24, type: !33, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !167)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !29)
// CHECK:STDOUT: !166 = !DILocation(line: 24, column: 3, scope: !161)
// CHECK:STDOUT: !167 = !{!162}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 20, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !182)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !29)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 2, scope: !168, type: !29)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 3, scope: !168, type: !29)
// CHECK:STDOUT: !175 = !DILocation(line: 21, column: 3, scope: !168)
// CHECK:STDOUT: !179 = !DILocation(line: 23, column: 22, scope: !168)
// CHECK:STDOUT: !180 = !DILocation(line: 23, column: 3, scope: !168)
// CHECK:STDOUT: !181 = !DILocation(line: 20, column: 1, scope: !168)
// CHECK:STDOUT: !182 = !{!169, !170, !171}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 33, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !201)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !29)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 2, scope: !183, type: !29)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 3, scope: !183, type: !29)
// CHECK:STDOUT: !190 = !DILocation(line: 34, column: 3, scope: !183)
// CHECK:STDOUT: !192 = !DILocation(line: 36, column: 10, scope: !183)
// CHECK:STDOUT: !193 = !DILocation(line: 36, column: 22, scope: !183)
// CHECK:STDOUT: !195 = !DILocation(line: 36, column: 3, scope: !183)
// CHECK:STDOUT: !199 = !DILocation(line: 37, column: 3, scope: !183)
// CHECK:STDOUT: !200 = !DILocation(line: 33, column: 1, scope: !183)
// CHECK:STDOUT: !201 = !{!184, !185, !186}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.901b24db5c2cfe42", scope: null, file: !16, line: 40, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !210)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !202, type: !29)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 2, scope: !202, type: !29)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 3, scope: !202, type: !29)
// CHECK:STDOUT: !209 = !DILocation(line: 40, column: 1, scope: !202)
// CHECK:STDOUT: !210 = !{!203, !204, !205}
// CHECK:STDOUT:
