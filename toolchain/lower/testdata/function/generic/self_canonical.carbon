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
// CHECK:STDOUT: define void @_CMain_method_no_template.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !7
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !7
// CHECK:STDOUT:   %.loc82 = load ptr, ptr %local1.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc82), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.d62da17adf4f9e43(ptr %local1.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.d62da17adf4f9e43(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0292bd1a0554e55c(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.8ae7ae7320c86eb3(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.41b892a8f89d0c66(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth_function.Main.4da09b2222fc2513(ptr %arg) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !34
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !34
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !35
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !35
// CHECK:STDOUT:   %.loc53 = load ptr, ptr %local1.var, align 8, !dbg !36
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc53), !dbg !37
// CHECK:STDOUT:   %.loc54 = load ptr, ptr %local2.var, align 8, !dbg !38
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc54), !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.68d4072289407214(ptr %local2.var), !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.7a15f2f04d824f56(ptr %local1.var), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst_function.Main.ee57965601b4d646(ptr %arg) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !44
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !44
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !45
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !45
// CHECK:STDOUT:   %.loc61_19 = load ptr, ptr %local2.var, align 8, !dbg !46
// CHECK:STDOUT:   %.loc61_27 = load ptr, ptr %local1.var, align 8, !dbg !47
// CHECK:STDOUT:   call void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %.loc61_19, ptr %.loc61_27, ptr %arg), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0292bd1a0554e55c(ptr %local2.var), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.8ae7ae7320c86eb3(ptr %local1.var), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.68d4072289407214(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.7a15f2f04d824f56(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !65
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !66
// CHECK:STDOUT:   %local3.var = alloca ptr, align 8, !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !65
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !66
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local3.var), !dbg !67
// CHECK:STDOUT:   store ptr poison, ptr %local3.var, align 8, !dbg !67
// CHECK:STDOUT:   %.loc75_25 = load ptr, ptr %local3.var, align 8, !dbg !68
// CHECK:STDOUT:   %.loc75_33 = load ptr, ptr %local2.var, align 8, !dbg !69
// CHECK:STDOUT:   call void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %arg3, ptr %.loc75_25, ptr %.loc75_33), !dbg !70
// CHECK:STDOUT:   %.loc76 = load ptr, ptr %local1.var, align 8, !dbg !71
// CHECK:STDOUT:   call void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %.loc76, ptr %arg3, ptr %arg1), !dbg !72
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.41b892a8f89d0c66(ptr %local3.var), !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.8ae7ae7320c86eb3(ptr %local2.var), !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0292bd1a0554e55c(ptr %local1.var), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !79
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !79
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !80
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !80
// CHECK:STDOUT:   %.loc39 = load ptr, ptr %local2.var, align 8, !dbg !81
// CHECK:STDOUT:   call void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %arg1, ptr %arg2, ptr %.loc39), !dbg !82
// CHECK:STDOUT:   %.loc40 = load ptr, ptr %local1.var, align 8, !dbg !83
// CHECK:STDOUT:   call void @_Cfourth_function.Main.4da09b2222fc2513(ptr %.loc40), !dbg !84
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.41b892a8f89d0c66(ptr %local2.var), !dbg !80
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.41b892a8f89d0c66(ptr %local1.var), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cfirst_function.Main.ee57965601b4d646, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "self_canonical.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Main_method_no_template", linkageName: "_CMain_method_no_template.Main", scope: null, file: !1, line: 79, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 80, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 82, column: 18, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 82, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 79, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.d62da17adf4f9e43", scope: null, file: !12, line: 24, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !12 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !11, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 24, column: 3, scope: !11)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0292bd1a0554e55c", scope: null, file: !12, line: 24, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 24, column: 3, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.8ae7ae7320c86eb3", scope: null, file: !12, line: 24, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !15)
// CHECK:STDOUT: !26 = !DILocation(line: 24, column: 3, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.41b892a8f89d0c66", scope: null, file: !12, line: 24, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !15)
// CHECK:STDOUT: !30 = !DILocation(line: 24, column: 3, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "fourth_function", linkageName: "_Cfourth_function.Main.4da09b2222fc2513", scope: null, file: !1, line: 49, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !32)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !31, type: !15)
// CHECK:STDOUT: !34 = !DILocation(line: 50, column: 3, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 51, column: 3, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 53, column: 18, scope: !31)
// CHECK:STDOUT: !37 = !DILocation(line: 53, column: 3, scope: !31)
// CHECK:STDOUT: !38 = !DILocation(line: 54, column: 18, scope: !31)
// CHECK:STDOUT: !39 = !DILocation(line: 54, column: 3, scope: !31)
// CHECK:STDOUT: !40 = !DILocation(line: 49, column: 1, scope: !31)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "first_function", linkageName: "_Cfirst_function.Main.ee57965601b4d646", scope: null, file: !1, line: 57, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !15)
// CHECK:STDOUT: !44 = !DILocation(line: 58, column: 3, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 59, column: 3, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 61, column: 19, scope: !41)
// CHECK:STDOUT: !47 = !DILocation(line: 61, column: 27, scope: !41)
// CHECK:STDOUT: !48 = !DILocation(line: 61, column: 3, scope: !41)
// CHECK:STDOUT: !49 = !DILocation(line: 57, column: 1, scope: !41)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.68d4072289407214", scope: null, file: !12, line: 24, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !51)
// CHECK:STDOUT: !51 = !{!52}
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !50, type: !15)
// CHECK:STDOUT: !53 = !DILocation(line: 24, column: 3, scope: !50)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.7a15f2f04d824f56", scope: null, file: !12, line: 24, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !55)
// CHECK:STDOUT: !55 = !{!56}
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !54, type: !15)
// CHECK:STDOUT: !57 = !DILocation(line: 24, column: 3, scope: !54)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "second_function", linkageName: "_Csecond_function.Main.efd9fcdf727c9de5", scope: null, file: !1, line: 70, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !60 = !{null, !15, !15, !15}
// CHECK:STDOUT: !61 = !{!62, !63, !64}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !58, type: !15)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 2, scope: !58, type: !15)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 3, scope: !58, type: !15)
// CHECK:STDOUT: !65 = !DILocation(line: 71, column: 3, scope: !58)
// CHECK:STDOUT: !66 = !DILocation(line: 72, column: 3, scope: !58)
// CHECK:STDOUT: !67 = !DILocation(line: 73, column: 3, scope: !58)
// CHECK:STDOUT: !68 = !DILocation(line: 75, column: 25, scope: !58)
// CHECK:STDOUT: !69 = !DILocation(line: 75, column: 33, scope: !58)
// CHECK:STDOUT: !70 = !DILocation(line: 75, column: 3, scope: !58)
// CHECK:STDOUT: !71 = !DILocation(line: 76, column: 18, scope: !58)
// CHECK:STDOUT: !72 = !DILocation(line: 76, column: 3, scope: !58)
// CHECK:STDOUT: !73 = !DILocation(line: 70, column: 1, scope: !58)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "third_function", linkageName: "_Cthird_function.Main.1c39a3b349bf66b6", scope: null, file: !1, line: 35, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !75)
// CHECK:STDOUT: !75 = !{!76, !77, !78}
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !74, type: !15)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 2, scope: !74, type: !15)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 3, scope: !74, type: !15)
// CHECK:STDOUT: !79 = !DILocation(line: 36, column: 3, scope: !74)
// CHECK:STDOUT: !80 = !DILocation(line: 37, column: 3, scope: !74)
// CHECK:STDOUT: !81 = !DILocation(line: 39, column: 30, scope: !74)
// CHECK:STDOUT: !82 = !DILocation(line: 39, column: 3, scope: !74)
// CHECK:STDOUT: !83 = !DILocation(line: 40, column: 19, scope: !74)
// CHECK:STDOUT: !84 = !DILocation(line: 40, column: 3, scope: !74)
// CHECK:STDOUT: !85 = !DILocation(line: 35, column: 1, scope: !74)
// CHECK:STDOUT:
