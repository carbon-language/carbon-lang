// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/self_canonical.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/self_canonical.carbon

// In this test, specific46 becomes canonical for others. Then, specific42 is
// found to be equivalent to specific46 and it becomes its canonical.
// Then through another equivalence from other specifics that point to
// specific46, specific46 is picked as the canonical. This is fine, we keep the
// chain of equivalences and eventually this is walked to get to specific42.
// The canonical for specific46, should not be updated to self in this case.
// Test that chains of equivalances are properly maintained and walked.

class class_name1 {}
class class_name2 {}
class class_name3 {}
class class_name4 {}
class class_name5 {}
class class_name6 {}

fn third_function[T: type, U: type, V: type](arg1: T, arg2: U, arg3: V);

fn fourth_function[T: type](arg: T);

fn first_function[T: type](arg: T);

fn second_function[T: type, U: type, V: type](arg1: T, arg2: U, arg3: V);

fn third_function[T: type, U: type, V: type](arg1: T, arg2: U, unused arg3: V) {
  var local1: class_name2*;
  var local2: class_name2*;

  third_function(arg1, arg2, local2);
  fourth_function(local1);
}

fn unused_1[T: type, U: type, V: type](arg1: T, unused arg2: U, arg3: V) {
  var local1: class_name6*;

  third_function(arg1, arg3, local1);
}

fn fourth_function[T: type](unused arg: T) {
  var local1: class_name5**;
  var local2: class_name6*;

  first_function(local1);
  first_function(local2);
}

fn first_function[T: type](arg: T) {
  var local1: class_name4*;
  var local2: class_name3*;

  second_function(local2, local1, arg);
}

fn unused_2[T: type, U: type](arg1: T, arg2: U) {
  var local1: class_name6*;

  second_function(arg1, arg2, local1);
}

fn second_function[T: type, U: type, V: type](arg1: T, unused arg2: U, arg3: V) {
  var local1: class_name3*;
  var local2: class_name4*;
  var local3: class_name2*;

  second_function(arg3, local3, local2);
  third_function(local1, arg3, arg1);
}

fn Main_method_no_template () {
  var local1: class_name1*;

  first_function(local1);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'self_canonical.carbon'
// CHECK:STDOUT: source_filename = "self_canonical.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain_method_no_template.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !21
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !21
// CHECK:STDOUT:   %.loc82 = load ptr, ptr %local1.var, align 8, !dbg !23
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc82), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.458ff0dea7bffb36(ptr %local1.var), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.458ff0dea7bffb36(ptr %t) #0 !dbg !44 {
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a2f88515bf42b4d2(ptr %t) #0 !dbg !74 {
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ef8f90c1036d7cb5(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.473bfd300d761a6f(ptr %t) #0 !dbg !130 {
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth_function.Main.4da09b2222fc2513(ptr %arg) #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !201
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !203
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !201
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !201
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !203
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !203
// CHECK:STDOUT:   %.loc53 = load ptr, ptr %local1.var, align 8, !dbg !205
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc53), !dbg !206
// CHECK:STDOUT:   %.loc54 = load ptr, ptr %local2.var, align 8, !dbg !207
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc54), !dbg !208
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2708f44bb6994785(ptr %local2.var), !dbg !203
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25557136f08dae8c(ptr %local1.var), !dbg !201
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst_function.Main.ee57965601b4d646(ptr %arg) #0 !dbg !249 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !252
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !254
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !252
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !252
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !254
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !254
// CHECK:STDOUT:   %.loc61_19 = load ptr, ptr %local2.var, align 8, !dbg !256
// CHECK:STDOUT:   %.loc61_27 = load ptr, ptr %local1.var, align 8, !dbg !257
// CHECK:STDOUT:   call void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %.loc61_19, ptr %.loc61_27, ptr %arg), !dbg !259
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a2f88515bf42b4d2(ptr %local2.var), !dbg !254
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ef8f90c1036d7cb5(ptr %local1.var), !dbg !252
// CHECK:STDOUT:   ret void, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2708f44bb6994785(ptr %t) #0 !dbg !275 {
// CHECK:STDOUT:   ret void, !dbg !280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.25557136f08dae8c(ptr %t) #0 !dbg !282 {
// CHECK:STDOUT:   ret void, !dbg !287
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !289 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !296
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !298
// CHECK:STDOUT:   %local3.var = alloca ptr, align 8, !dbg !300
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !296
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !296
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !298
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !298
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local3.var), !dbg !300
// CHECK:STDOUT:   store ptr poison, ptr %local3.var, align 8, !dbg !300
// CHECK:STDOUT:   %.loc75_25 = load ptr, ptr %local3.var, align 8, !dbg !303
// CHECK:STDOUT:   %.loc75_33 = load ptr, ptr %local2.var, align 8, !dbg !304
// CHECK:STDOUT:   call void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %arg3, ptr %.loc75_25, ptr %.loc75_33), !dbg !305
// CHECK:STDOUT:   %.loc76 = load ptr, ptr %local1.var, align 8, !dbg !306
// CHECK:STDOUT:   call void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %.loc76, ptr %arg3, ptr %arg1), !dbg !309
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.473bfd300d761a6f(ptr %local3.var), !dbg !300
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ef8f90c1036d7cb5(ptr %local2.var), !dbg !298
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a2f88515bf42b4d2(ptr %local1.var), !dbg !296
// CHECK:STDOUT:   ret void, !dbg !310
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !358 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !365
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !367
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !365
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !365
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !367
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !367
// CHECK:STDOUT:   %.loc39 = load ptr, ptr %local2.var, align 8, !dbg !371
// CHECK:STDOUT:   call void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %arg1, ptr %arg2, ptr %.loc39), !dbg !372
// CHECK:STDOUT:   %.loc40 = load ptr, ptr %local1.var, align 8, !dbg !373
// CHECK:STDOUT:   call void @_Cfourth_function.Main.4da09b2222fc2513(ptr %.loc40), !dbg !374
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.473bfd300d761a6f(ptr %local2.var), !dbg !367
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.473bfd300d761a6f(ptr %local1.var), !dbg !365
// CHECK:STDOUT:   ret void, !dbg !375
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cfirst_function.Main.ee57965601b4d646, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "self_canonical.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Main_method_no_template", linkageName: "_CMain_method_no_template.Main", scope: null, file: !16, line: 79, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 80, column: 3, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 82, column: 18, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 82, column: 3, scope: !20)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{null, !25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 79, column: 1, scope: !20)
// CHECK:STDOUT: !40 = !{null, !25, !25, !25}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.458ff0dea7bffb36", scope: null, file: !6, line: 30, type: !27, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !50)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !25)
// CHECK:STDOUT: !49 = !DILocation(line: 30, column: 1, scope: !44)
// CHECK:STDOUT: !50 = !{!45}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a2f88515bf42b4d2", scope: null, file: !6, line: 30, type: !27, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !25)
// CHECK:STDOUT: !79 = !DILocation(line: 30, column: 1, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ef8f90c1036d7cb5", scope: null, file: !6, line: 30, type: !27, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !25)
// CHECK:STDOUT: !86 = !DILocation(line: 30, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.473bfd300d761a6f", scope: null, file: !6, line: 30, type: !27, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !136)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !25)
// CHECK:STDOUT: !135 = !DILocation(line: 30, column: 1, scope: !130)
// CHECK:STDOUT: !136 = !{!131}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "fourth_function", linkageName: "_Cfourth_function.Main.4da09b2222fc2513", scope: null, file: !16, line: 49, type: !27, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !210)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !25)
// CHECK:STDOUT: !201 = !DILocation(line: 50, column: 3, scope: !198)
// CHECK:STDOUT: !203 = !DILocation(line: 51, column: 3, scope: !198)
// CHECK:STDOUT: !205 = !DILocation(line: 53, column: 18, scope: !198)
// CHECK:STDOUT: !206 = !DILocation(line: 53, column: 3, scope: !198)
// CHECK:STDOUT: !207 = !DILocation(line: 54, column: 18, scope: !198)
// CHECK:STDOUT: !208 = !DILocation(line: 54, column: 3, scope: !198)
// CHECK:STDOUT: !209 = !DILocation(line: 49, column: 1, scope: !198)
// CHECK:STDOUT: !210 = !{!199}
// CHECK:STDOUT: !249 = distinct !DISubprogram(name: "first_function", linkageName: "_Cfirst_function.Main.ee57965601b4d646", scope: null, file: !16, line: 57, type: !27, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !261)
// CHECK:STDOUT: !250 = !DILocalVariable(arg: 1, scope: !249, type: !25)
// CHECK:STDOUT: !252 = !DILocation(line: 58, column: 3, scope: !249)
// CHECK:STDOUT: !254 = !DILocation(line: 59, column: 3, scope: !249)
// CHECK:STDOUT: !256 = !DILocation(line: 61, column: 19, scope: !249)
// CHECK:STDOUT: !257 = !DILocation(line: 61, column: 27, scope: !249)
// CHECK:STDOUT: !259 = !DILocation(line: 61, column: 3, scope: !249)
// CHECK:STDOUT: !260 = !DILocation(line: 57, column: 1, scope: !249)
// CHECK:STDOUT: !261 = !{!250}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2708f44bb6994785", scope: null, file: !6, line: 30, type: !27, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !281)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !25)
// CHECK:STDOUT: !280 = !DILocation(line: 30, column: 1, scope: !275)
// CHECK:STDOUT: !281 = !{!276}
// CHECK:STDOUT: !282 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.25557136f08dae8c", scope: null, file: !6, line: 30, type: !27, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !288)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 1, scope: !282, type: !25)
// CHECK:STDOUT: !287 = !DILocation(line: 30, column: 1, scope: !282)
// CHECK:STDOUT: !288 = !{!283}
// CHECK:STDOUT: !289 = distinct !DISubprogram(name: "second_function", linkageName: "_Csecond_function.Main.efd9fcdf727c9de5", scope: null, file: !16, line: 70, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !311)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 1, scope: !289, type: !25)
// CHECK:STDOUT: !291 = !DILocalVariable(arg: 2, scope: !289, type: !25)
// CHECK:STDOUT: !292 = !DILocalVariable(arg: 3, scope: !289, type: !25)
// CHECK:STDOUT: !296 = !DILocation(line: 71, column: 3, scope: !289)
// CHECK:STDOUT: !298 = !DILocation(line: 72, column: 3, scope: !289)
// CHECK:STDOUT: !300 = !DILocation(line: 73, column: 3, scope: !289)
// CHECK:STDOUT: !303 = !DILocation(line: 75, column: 25, scope: !289)
// CHECK:STDOUT: !304 = !DILocation(line: 75, column: 33, scope: !289)
// CHECK:STDOUT: !305 = !DILocation(line: 75, column: 3, scope: !289)
// CHECK:STDOUT: !306 = !DILocation(line: 76, column: 18, scope: !289)
// CHECK:STDOUT: !309 = !DILocation(line: 76, column: 3, scope: !289)
// CHECK:STDOUT: !310 = !DILocation(line: 70, column: 1, scope: !289)
// CHECK:STDOUT: !311 = !{!290, !291, !292}
// CHECK:STDOUT: !358 = distinct !DISubprogram(name: "third_function", linkageName: "_Cthird_function.Main.1c39a3b349bf66b6", scope: null, file: !16, line: 35, type: !41, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !376)
// CHECK:STDOUT: !359 = !DILocalVariable(arg: 1, scope: !358, type: !25)
// CHECK:STDOUT: !360 = !DILocalVariable(arg: 2, scope: !358, type: !25)
// CHECK:STDOUT: !361 = !DILocalVariable(arg: 3, scope: !358, type: !25)
// CHECK:STDOUT: !365 = !DILocation(line: 36, column: 3, scope: !358)
// CHECK:STDOUT: !367 = !DILocation(line: 37, column: 3, scope: !358)
// CHECK:STDOUT: !371 = !DILocation(line: 39, column: 30, scope: !358)
// CHECK:STDOUT: !372 = !DILocation(line: 39, column: 3, scope: !358)
// CHECK:STDOUT: !373 = !DILocation(line: 40, column: 19, scope: !358)
// CHECK:STDOUT: !374 = !DILocation(line: 40, column: 3, scope: !358)
// CHECK:STDOUT: !375 = !DILocation(line: 35, column: 1, scope: !358)
// CHECK:STDOUT: !376 = !{!359, !360, !361}
// CHECK:STDOUT:
