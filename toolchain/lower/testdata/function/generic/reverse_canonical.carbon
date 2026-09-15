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
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !40
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !40
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !44
// CHECK:STDOUT:   call void @_Csecond.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc23), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !55
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !55
// CHECK:STDOUT:   call void @_Cfirst.Main.81f1c9de445b895c(ptr %arg3, ptr %arg3, i1 %arg1), !dbg !60
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !63
// CHECK:STDOUT:   call void @_Cthird.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc30), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.81f1c9de445b895c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !74
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !74
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !78
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !89
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !89
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !91
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !92
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %.loc36_10, ptr %.loc36_22, i1 %arg1), !dbg !94
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !108
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !108
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !108
// CHECK:STDOUT:   call void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !113
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !116
// CHECK:STDOUT:   call void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc30), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !145
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !145
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !145
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !149
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !160
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !160
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !160
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !162
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !163
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %.loc36_10, ptr %.loc36_22, ptr %arg1), !dbg !165
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !179
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
// CHECK:STDOUT: !32 = !DILocation(line: 43, column: 1, scope: !20)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !16, line: 20, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !47)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !29)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !33, type: !29)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 3, scope: !33, type: !29)
// CHECK:STDOUT: !40 = !DILocation(line: 21, column: 3, scope: !33)
// CHECK:STDOUT: !44 = !DILocation(line: 23, column: 22, scope: !33)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 3, scope: !33)
// CHECK:STDOUT: !46 = !DILocation(line: 20, column: 1, scope: !33)
// CHECK:STDOUT: !47 = !{!34, !35, !36}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.0f29a29bf45af723", scope: null, file: !16, line: 26, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !66)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !29)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !48, type: !29)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 3, scope: !48, type: !29)
// CHECK:STDOUT: !55 = !DILocation(line: 27, column: 3, scope: !48)
// CHECK:STDOUT: !60 = !DILocation(line: 29, column: 3, scope: !48)
// CHECK:STDOUT: !63 = !DILocation(line: 30, column: 21, scope: !48)
// CHECK:STDOUT: !64 = !DILocation(line: 30, column: 3, scope: !48)
// CHECK:STDOUT: !65 = !DILocation(line: 26, column: 1, scope: !48)
// CHECK:STDOUT: !66 = !{!49, !50, !51}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.81f1c9de445b895c", scope: null, file: !16, line: 20, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !81)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !29)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 2, scope: !67, type: !29)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 3, scope: !67, type: !29)
// CHECK:STDOUT: !74 = !DILocation(line: 21, column: 3, scope: !67)
// CHECK:STDOUT: !78 = !DILocation(line: 23, column: 22, scope: !67)
// CHECK:STDOUT: !79 = !DILocation(line: 23, column: 3, scope: !67)
// CHECK:STDOUT: !80 = !DILocation(line: 20, column: 1, scope: !67)
// CHECK:STDOUT: !81 = !{!68, !69, !70}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.0f29a29bf45af723", scope: null, file: !16, line: 33, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !100)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !29)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !82, type: !29)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 3, scope: !82, type: !29)
// CHECK:STDOUT: !89 = !DILocation(line: 34, column: 3, scope: !82)
// CHECK:STDOUT: !91 = !DILocation(line: 36, column: 10, scope: !82)
// CHECK:STDOUT: !92 = !DILocation(line: 36, column: 22, scope: !82)
// CHECK:STDOUT: !94 = !DILocation(line: 36, column: 3, scope: !82)
// CHECK:STDOUT: !98 = !DILocation(line: 37, column: 3, scope: !82)
// CHECK:STDOUT: !99 = !DILocation(line: 33, column: 1, scope: !82)
// CHECK:STDOUT: !100 = !{!83, !84, !85}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 26, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !119)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !29)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 2, scope: !101, type: !29)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 3, scope: !101, type: !29)
// CHECK:STDOUT: !108 = !DILocation(line: 27, column: 3, scope: !101)
// CHECK:STDOUT: !113 = !DILocation(line: 29, column: 3, scope: !101)
// CHECK:STDOUT: !116 = !DILocation(line: 30, column: 21, scope: !101)
// CHECK:STDOUT: !117 = !DILocation(line: 30, column: 3, scope: !101)
// CHECK:STDOUT: !118 = !DILocation(line: 26, column: 1, scope: !101)
// CHECK:STDOUT: !119 = !{!102, !103, !104}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.c6787c0e2572c86c", scope: null, file: !16, line: 40, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !128)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !29)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !120, type: !29)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 3, scope: !120, type: !29)
// CHECK:STDOUT: !127 = !DILocation(line: 40, column: 1, scope: !120)
// CHECK:STDOUT: !128 = !{!121, !122, !123}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 20, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !152)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !29)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 2, scope: !138, type: !29)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 3, scope: !138, type: !29)
// CHECK:STDOUT: !145 = !DILocation(line: 21, column: 3, scope: !138)
// CHECK:STDOUT: !149 = !DILocation(line: 23, column: 22, scope: !138)
// CHECK:STDOUT: !150 = !DILocation(line: 23, column: 3, scope: !138)
// CHECK:STDOUT: !151 = !DILocation(line: 20, column: 1, scope: !138)
// CHECK:STDOUT: !152 = !{!139, !140, !141}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.e1ce67f3c23ab61f", scope: null, file: !16, line: 33, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !171)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !29)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 2, scope: !153, type: !29)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 3, scope: !153, type: !29)
// CHECK:STDOUT: !160 = !DILocation(line: 34, column: 3, scope: !153)
// CHECK:STDOUT: !162 = !DILocation(line: 36, column: 10, scope: !153)
// CHECK:STDOUT: !163 = !DILocation(line: 36, column: 22, scope: !153)
// CHECK:STDOUT: !165 = !DILocation(line: 36, column: 3, scope: !153)
// CHECK:STDOUT: !169 = !DILocation(line: 37, column: 3, scope: !153)
// CHECK:STDOUT: !170 = !DILocation(line: 33, column: 1, scope: !153)
// CHECK:STDOUT: !171 = !{!154, !155, !156}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.901b24db5c2cfe42", scope: null, file: !16, line: 40, type: !31, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !180)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !29)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 2, scope: !172, type: !29)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 3, scope: !172, type: !29)
// CHECK:STDOUT: !179 = !DILocation(line: 40, column: 1, scope: !172)
// CHECK:STDOUT: !180 = !{!173, !174, !175}
// CHECK:STDOUT:
