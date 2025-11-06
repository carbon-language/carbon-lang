// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
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

fn third_function[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V);

fn fourth_function[T:! type](arg: T);

fn first_function[T:! type](arg: T);

fn second_function[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V);

fn third_function[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V) {
  var local1: class_name2*;
  var local2: class_name2*;

  third_function(arg1, arg2, local2);
  fourth_function(local1);
}

fn unused_1[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V) {
  var local1: class_name6*;

  third_function(arg1, arg3, local1);
}

fn fourth_function[T:! type](arg: T) {
  var local1: class_name5**;
  var local2: class_name6*;

  first_function(local1);
  first_function(local2);
}

fn first_function[T:! type](arg: T) {
  var local1: class_name4*;
  var local2: class_name3*;

  second_function(local2, local1, arg);
}

fn unused_2[T:! type, U:! type](arg1: T, arg2: U) {
  var local1: class_name6*;

  second_function(arg1, arg2, local1);
}

fn second_function[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V) {
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

// CHECK:STDOUT: ; ModuleID = 'self_canonical.carbon'
// CHECK:STDOUT: source_filename = "self_canonical.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain_method_no_template.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !7
// CHECK:STDOUT:   %.loc82 = load ptr, ptr %local1.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @_Cfirst_function.Main.308f7be7cd8574f7(ptr %.loc82), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth_function.Main.a95d5636f2d020e8(ptr %arg) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !13
// CHECK:STDOUT:   %.loc53 = load ptr, ptr %local1.var, align 8, !dbg !14
// CHECK:STDOUT:   call void @_Cfirst_function.Main.308f7be7cd8574f7(ptr %.loc53), !dbg !15
// CHECK:STDOUT:   %.loc54 = load ptr, ptr %local2.var, align 8, !dbg !16
// CHECK:STDOUT:   call void @_Cfirst_function.Main.308f7be7cd8574f7(ptr %.loc54), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst_function.Main.308f7be7cd8574f7(ptr %arg) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !20
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !21
// CHECK:STDOUT:   %.loc61_19 = load ptr, ptr %local2.var, align 8, !dbg !22
// CHECK:STDOUT:   %.loc61_27 = load ptr, ptr %local1.var, align 8, !dbg !23
// CHECK:STDOUT:   call void @_Csecond_function.Main.6e67709fb5f3d893(ptr %.loc61_19, ptr %.loc61_27, ptr %arg), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond_function.Main.6e67709fb5f3d893(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !27
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !28
// CHECK:STDOUT:   %local3.var = alloca ptr, align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local3.var), !dbg !29
// CHECK:STDOUT:   %.loc75_25 = load ptr, ptr %local3.var, align 8, !dbg !30
// CHECK:STDOUT:   %.loc75_33 = load ptr, ptr %local2.var, align 8, !dbg !31
// CHECK:STDOUT:   call void @_Csecond_function.Main.6e67709fb5f3d893(ptr %arg3, ptr %.loc75_25, ptr %.loc75_33), !dbg !32
// CHECK:STDOUT:   %.loc76 = load ptr, ptr %local1.var, align 8, !dbg !33
// CHECK:STDOUT:   call void @_Cthird_function.Main.bd8cd068af0bdf76(ptr %.loc76, ptr %arg3, ptr %arg1), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird_function.Main.bd8cd068af0bdf76(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !37
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !38
// CHECK:STDOUT:   %.loc39 = load ptr, ptr %local2.var, align 8, !dbg !39
// CHECK:STDOUT:   call void @_Cthird_function.Main.bd8cd068af0bdf76(ptr %arg1, ptr %arg2, ptr %.loc39), !dbg !40
// CHECK:STDOUT:   %.loc40 = load ptr, ptr %local1.var, align 8, !dbg !41
// CHECK:STDOUT:   call void @_Cfourth_function.Main.a95d5636f2d020e8(ptr %.loc40), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cfirst_function.Main.308f7be7cd8574f7, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "self_canonical.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Main_method_no_template", linkageName: "_CMain_method_no_template.Main", scope: null, file: !3, line: 79, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 80, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 82, column: 18, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 82, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 79, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "fourth_function", linkageName: "_Cfourth_function.Main.a95d5636f2d020e8", scope: null, file: !3, line: 49, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DILocation(line: 50, column: 3, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 51, column: 3, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 53, column: 18, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 53, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 54, column: 18, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 54, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 49, column: 1, scope: !11)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "first_function", linkageName: "_Cfirst_function.Main.308f7be7cd8574f7", scope: null, file: !3, line: 57, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DILocation(line: 58, column: 3, scope: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 59, column: 3, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 61, column: 19, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 61, column: 27, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 61, column: 3, scope: !19)
// CHECK:STDOUT: !25 = !DILocation(line: 57, column: 1, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "second_function", linkageName: "_Csecond_function.Main.6e67709fb5f3d893", scope: null, file: !3, line: 70, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !27 = !DILocation(line: 71, column: 3, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 72, column: 3, scope: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 73, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 75, column: 25, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 75, column: 33, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 75, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 76, column: 18, scope: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 76, column: 3, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 70, column: 1, scope: !26)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "third_function", linkageName: "_Cthird_function.Main.bd8cd068af0bdf76", scope: null, file: !3, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 36, column: 3, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 37, column: 3, scope: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 39, column: 30, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 39, column: 3, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 40, column: 19, scope: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 40, column: 3, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 35, column: 1, scope: !36)
