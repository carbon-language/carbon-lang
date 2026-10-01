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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cc2eb235f3a2109b:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.146e3bb0985f5a0c:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i1, align 1, !dbg !32
// CHECK:STDOUT:   %y.var = alloca ptr, align 8, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !32
// CHECK:STDOUT:   store i8 poison, ptr %x.var, align 1, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !34
// CHECK:STDOUT:   store ptr poison, ptr %y.var, align 8, !dbg !34
// CHECK:STDOUT:   %.loc47_9 = load i8, ptr %x.var, align 1, !dbg !36
// CHECK:STDOUT:   %.loc47_91 = trunc i8 %.loc47_9 to i1, !dbg !36
// CHECK:STDOUT:   %.loc47_12 = load i8, ptr %x.var, align 1, !dbg !37
// CHECK:STDOUT:   %.loc47_122 = trunc i8 %.loc47_12 to i1, !dbg !37
// CHECK:STDOUT:   %.loc47_15 = load ptr, ptr %y.var, align 8, !dbg !38
// CHECK:STDOUT:   call void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %.loc47_91, i1 %.loc47_122, ptr %.loc47_15), !dbg !39
// CHECK:STDOUT:   call void @"_CSelfDestruct.bb2383e5e4785e19:core.Destroy.Core"(ptr %y.var), !dbg !34
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %x.var), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb2383e5e4785e19:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !58
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !58
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !62
// CHECK:STDOUT:   call void @_Csecond.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc23), !dbg !63
// CHECK:STDOUT:   call void @"_CSelfDestruct.cc2eb235f3a2109b:core.Destroy.Core"(ptr %local_name.var), !dbg !58
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
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !81
// CHECK:STDOUT:   call void @_Cthird.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc30), !dbg !82
// CHECK:STDOUT:   call void @"_CSelfDestruct.cc2eb235f3a2109b:core.Destroy.Core"(ptr %local_name.var), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.81f1c9de445b895c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !92
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !92
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !96
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !97
// CHECK:STDOUT:   call void @"_CSelfDestruct.cc2eb235f3a2109b:core.Destroy.Core"(ptr %local_name.var), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !107
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !107
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !107
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !109
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !110
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %.loc36_10, ptr %.loc36_22, i1 %arg1), !dbg !112
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !116
// CHECK:STDOUT:   call void @"_CSelfDestruct.146e3bb0985f5a0c:core.Destroy.Core"(ptr %local_name.var), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !126
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !126
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !126
// CHECK:STDOUT:   call void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !131
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !134
// CHECK:STDOUT:   call void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc30), !dbg !135
// CHECK:STDOUT:   call void @"_CSelfDestruct.cc2eb235f3a2109b:core.Destroy.Core"(ptr %local_name.var), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !163
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !163
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !163
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !167
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !168
// CHECK:STDOUT:   call void @"_CSelfDestruct.cc2eb235f3a2109b:core.Destroy.Core"(ptr %local_name.var), !dbg !163
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !178
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !178
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !180
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !181
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %.loc36_10, ptr %.loc36_22, ptr %arg1), !dbg !183
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !187
// CHECK:STDOUT:   call void @"_CSelfDestruct.146e3bb0985f5a0c:core.Destroy.Core"(ptr %local_name.var), !dbg !178
// CHECK:STDOUT:   ret void, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !197
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "reverse_canonical.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{null, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cc2eb235f3a2109b:core.Destroy.Core", scope: null, file: !16, line: 21, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !18)
// CHECK:STDOUT: !25 = !DILocation(line: 21, column: 3, scope: !23)
// CHECK:STDOUT: !26 = !{!24}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.146e3bb0985f5a0c:core.Destroy.Core", scope: null, file: !16, line: 34, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !18)
// CHECK:STDOUT: !29 = !DILocation(line: 34, column: 3, scope: !27)
// CHECK:STDOUT: !30 = !{!28}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !16, line: 43, type: !22, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !32 = !DILocation(line: 44, column: 3, scope: !31)
// CHECK:STDOUT: !34 = !DILocation(line: 45, column: 3, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 47, column: 9, scope: !31)
// CHECK:STDOUT: !37 = !DILocation(line: 47, column: 12, scope: !31)
// CHECK:STDOUT: !38 = !DILocation(line: 47, column: 15, scope: !31)
// CHECK:STDOUT: !39 = !DILocation(line: 47, column: 3, scope: !31)
// CHECK:STDOUT: !40 = !{null, !18, !18, !18}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 43, column: 1, scope: !31)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb2383e5e4785e19:core.Destroy.Core", scope: null, file: !16, line: 45, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !18)
// CHECK:STDOUT: !45 = !DILocation(line: 45, column: 3, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !16, line: 44, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !18)
// CHECK:STDOUT: !49 = !DILocation(line: 44, column: 3, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !16, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !65)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !18)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 2, scope: !51, type: !18)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 3, scope: !51, type: !18)
// CHECK:STDOUT: !58 = !DILocation(line: 21, column: 3, scope: !51)
// CHECK:STDOUT: !62 = !DILocation(line: 23, column: 22, scope: !51)
// CHECK:STDOUT: !63 = !DILocation(line: 23, column: 3, scope: !51)
// CHECK:STDOUT: !64 = !DILocation(line: 20, column: 1, scope: !51)
// CHECK:STDOUT: !65 = !{!52, !53, !54}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.0f29a29bf45af723", scope: null, file: !16, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !84)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !18)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 2, scope: !66, type: !18)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 3, scope: !66, type: !18)
// CHECK:STDOUT: !73 = !DILocation(line: 27, column: 3, scope: !66)
// CHECK:STDOUT: !78 = !DILocation(line: 29, column: 3, scope: !66)
// CHECK:STDOUT: !81 = !DILocation(line: 30, column: 21, scope: !66)
// CHECK:STDOUT: !82 = !DILocation(line: 30, column: 3, scope: !66)
// CHECK:STDOUT: !83 = !DILocation(line: 26, column: 1, scope: !66)
// CHECK:STDOUT: !84 = !{!67, !68, !69}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.81f1c9de445b895c", scope: null, file: !16, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !99)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !18)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 2, scope: !85, type: !18)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 3, scope: !85, type: !18)
// CHECK:STDOUT: !92 = !DILocation(line: 21, column: 3, scope: !85)
// CHECK:STDOUT: !96 = !DILocation(line: 23, column: 22, scope: !85)
// CHECK:STDOUT: !97 = !DILocation(line: 23, column: 3, scope: !85)
// CHECK:STDOUT: !98 = !DILocation(line: 20, column: 1, scope: !85)
// CHECK:STDOUT: !99 = !{!86, !87, !88}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.0f29a29bf45af723", scope: null, file: !16, line: 33, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !118)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !18)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 2, scope: !100, type: !18)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 3, scope: !100, type: !18)
// CHECK:STDOUT: !107 = !DILocation(line: 34, column: 3, scope: !100)
// CHECK:STDOUT: !109 = !DILocation(line: 36, column: 10, scope: !100)
// CHECK:STDOUT: !110 = !DILocation(line: 36, column: 22, scope: !100)
// CHECK:STDOUT: !112 = !DILocation(line: 36, column: 3, scope: !100)
// CHECK:STDOUT: !116 = !DILocation(line: 37, column: 3, scope: !100)
// CHECK:STDOUT: !117 = !DILocation(line: 33, column: 1, scope: !100)
// CHECK:STDOUT: !118 = !{!101, !102, !103}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !137)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !18)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 2, scope: !119, type: !18)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 3, scope: !119, type: !18)
// CHECK:STDOUT: !126 = !DILocation(line: 27, column: 3, scope: !119)
// CHECK:STDOUT: !131 = !DILocation(line: 29, column: 3, scope: !119)
// CHECK:STDOUT: !134 = !DILocation(line: 30, column: 21, scope: !119)
// CHECK:STDOUT: !135 = !DILocation(line: 30, column: 3, scope: !119)
// CHECK:STDOUT: !136 = !DILocation(line: 26, column: 1, scope: !119)
// CHECK:STDOUT: !137 = !{!120, !121, !122}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.c6787c0e2572c86c", scope: null, file: !16, line: 40, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !146)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !18)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 2, scope: !138, type: !18)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 3, scope: !138, type: !18)
// CHECK:STDOUT: !145 = !DILocation(line: 40, column: 1, scope: !138)
// CHECK:STDOUT: !146 = !{!139, !140, !141}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !170)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !18)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 2, scope: !156, type: !18)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 3, scope: !156, type: !18)
// CHECK:STDOUT: !163 = !DILocation(line: 21, column: 3, scope: !156)
// CHECK:STDOUT: !167 = !DILocation(line: 23, column: 22, scope: !156)
// CHECK:STDOUT: !168 = !DILocation(line: 23, column: 3, scope: !156)
// CHECK:STDOUT: !169 = !DILocation(line: 20, column: 1, scope: !156)
// CHECK:STDOUT: !170 = !{!157, !158, !159}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 33, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !189)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !18)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 2, scope: !171, type: !18)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 3, scope: !171, type: !18)
// CHECK:STDOUT: !178 = !DILocation(line: 34, column: 3, scope: !171)
// CHECK:STDOUT: !180 = !DILocation(line: 36, column: 10, scope: !171)
// CHECK:STDOUT: !181 = !DILocation(line: 36, column: 22, scope: !171)
// CHECK:STDOUT: !183 = !DILocation(line: 36, column: 3, scope: !171)
// CHECK:STDOUT: !187 = !DILocation(line: 37, column: 3, scope: !171)
// CHECK:STDOUT: !188 = !DILocation(line: 33, column: 1, scope: !171)
// CHECK:STDOUT: !189 = !{!172, !173, !174}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.901b24db5c2cfe42", scope: null, file: !16, line: 40, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !198)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !190, type: !18)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 2, scope: !190, type: !18)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 3, scope: !190, type: !18)
// CHECK:STDOUT: !197 = !DILocation(line: 40, column: 1, scope: !190)
// CHECK:STDOUT: !198 = !{!191, !192, !193}
// CHECK:STDOUT:
