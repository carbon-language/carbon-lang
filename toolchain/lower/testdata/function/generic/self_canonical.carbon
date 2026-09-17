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
// CHECK:STDOUT: define void @_CMain_method_no_template.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !19
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !19
// CHECK:STDOUT:   %.loc82 = load ptr, ptr %local1.var, align 8, !dbg !21
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc82), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.d62da17adf4f9e43(ptr %local1.var), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.d62da17adf4f9e43(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0292bd1a0554e55c(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.8ae7ae7320c86eb3(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.41b892a8f89d0c66(ptr %self) #0 !dbg !128 {
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth_function.Main.4da09b2222fc2513(ptr %arg) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !199
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !201
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !199
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !199
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !201
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !201
// CHECK:STDOUT:   %.loc53 = load ptr, ptr %local1.var, align 8, !dbg !203
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc53), !dbg !204
// CHECK:STDOUT:   %.loc54 = load ptr, ptr %local2.var, align 8, !dbg !205
// CHECK:STDOUT:   call void @_Cfirst_function.Main.ee57965601b4d646(ptr %.loc54), !dbg !206
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.68d4072289407214(ptr %local2.var), !dbg !201
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.7a15f2f04d824f56(ptr %local1.var), !dbg !199
// CHECK:STDOUT:   ret void, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst_function.Main.ee57965601b4d646(ptr %arg) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !250
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !252
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !250
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !250
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !252
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !252
// CHECK:STDOUT:   %.loc61_19 = load ptr, ptr %local2.var, align 8, !dbg !254
// CHECK:STDOUT:   %.loc61_27 = load ptr, ptr %local1.var, align 8, !dbg !255
// CHECK:STDOUT:   call void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %.loc61_19, ptr %.loc61_27, ptr %arg), !dbg !257
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0292bd1a0554e55c(ptr %local2.var), !dbg !252
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.8ae7ae7320c86eb3(ptr %local1.var), !dbg !250
// CHECK:STDOUT:   ret void, !dbg !258
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.68d4072289407214(ptr %self) #0 !dbg !273 {
// CHECK:STDOUT:   ret void, !dbg !278
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.7a15f2f04d824f56(ptr %self) #0 !dbg !280 {
// CHECK:STDOUT:   ret void, !dbg !285
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !287 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !294
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !296
// CHECK:STDOUT:   %local3.var = alloca ptr, align 8, !dbg !298
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !294
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !294
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !296
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !296
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local3.var), !dbg !298
// CHECK:STDOUT:   store ptr poison, ptr %local3.var, align 8, !dbg !298
// CHECK:STDOUT:   %.loc75_25 = load ptr, ptr %local3.var, align 8, !dbg !301
// CHECK:STDOUT:   %.loc75_33 = load ptr, ptr %local2.var, align 8, !dbg !302
// CHECK:STDOUT:   call void @_Csecond_function.Main.efd9fcdf727c9de5(ptr %arg3, ptr %.loc75_25, ptr %.loc75_33), !dbg !303
// CHECK:STDOUT:   %.loc76 = load ptr, ptr %local1.var, align 8, !dbg !304
// CHECK:STDOUT:   call void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %.loc76, ptr %arg3, ptr %arg1), !dbg !307
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.41b892a8f89d0c66(ptr %local3.var), !dbg !298
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.8ae7ae7320c86eb3(ptr %local2.var), !dbg !296
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0292bd1a0554e55c(ptr %local1.var), !dbg !294
// CHECK:STDOUT:   ret void, !dbg !308
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !356 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local1.var = alloca ptr, align 8, !dbg !363
// CHECK:STDOUT:   %local2.var = alloca ptr, align 8, !dbg !365
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local1.var), !dbg !363
// CHECK:STDOUT:   store ptr poison, ptr %local1.var, align 8, !dbg !363
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local2.var), !dbg !365
// CHECK:STDOUT:   store ptr poison, ptr %local2.var, align 8, !dbg !365
// CHECK:STDOUT:   %.loc39 = load ptr, ptr %local2.var, align 8, !dbg !369
// CHECK:STDOUT:   call void @_Cthird_function.Main.1c39a3b349bf66b6(ptr %arg1, ptr %arg2, ptr %.loc39), !dbg !370
// CHECK:STDOUT:   %.loc40 = load ptr, ptr %local1.var, align 8, !dbg !371
// CHECK:STDOUT:   call void @_Cfourth_function.Main.4da09b2222fc2513(ptr %.loc40), !dbg !372
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.41b892a8f89d0c66(ptr %local2.var), !dbg !365
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.41b892a8f89d0c66(ptr %local1.var), !dbg !363
// CHECK:STDOUT:   ret void, !dbg !373
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cfirst_function.Main.ee57965601b4d646, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "self_canonical.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Main_method_no_template", linkageName: "_CMain_method_no_template.Main", scope: null, file: !14, line: 79, type: !17, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 80, column: 3, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 82, column: 18, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 82, column: 3, scope: !18)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 79, column: 1, scope: !18)
// CHECK:STDOUT: !38 = !{null, !23, !23, !23}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.d62da17adf4f9e43", scope: null, file: !10, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !48)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !23)
// CHECK:STDOUT: !47 = !DILocation(line: 24, column: 3, scope: !42)
// CHECK:STDOUT: !48 = !{!43}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0292bd1a0554e55c", scope: null, file: !10, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !78)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !23)
// CHECK:STDOUT: !77 = !DILocation(line: 24, column: 3, scope: !72)
// CHECK:STDOUT: !78 = !{!73}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.8ae7ae7320c86eb3", scope: null, file: !10, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !85)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !23)
// CHECK:STDOUT: !84 = !DILocation(line: 24, column: 3, scope: !79)
// CHECK:STDOUT: !85 = !{!80}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.41b892a8f89d0c66", scope: null, file: !10, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !134)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !23)
// CHECK:STDOUT: !133 = !DILocation(line: 24, column: 3, scope: !128)
// CHECK:STDOUT: !134 = !{!129}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "fourth_function", linkageName: "_Cfourth_function.Main.4da09b2222fc2513", scope: null, file: !14, line: 49, type: !25, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !208)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !23)
// CHECK:STDOUT: !199 = !DILocation(line: 50, column: 3, scope: !196)
// CHECK:STDOUT: !201 = !DILocation(line: 51, column: 3, scope: !196)
// CHECK:STDOUT: !203 = !DILocation(line: 53, column: 18, scope: !196)
// CHECK:STDOUT: !204 = !DILocation(line: 53, column: 3, scope: !196)
// CHECK:STDOUT: !205 = !DILocation(line: 54, column: 18, scope: !196)
// CHECK:STDOUT: !206 = !DILocation(line: 54, column: 3, scope: !196)
// CHECK:STDOUT: !207 = !DILocation(line: 49, column: 1, scope: !196)
// CHECK:STDOUT: !208 = !{!197}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "first_function", linkageName: "_Cfirst_function.Main.ee57965601b4d646", scope: null, file: !14, line: 57, type: !25, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !259)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !23)
// CHECK:STDOUT: !250 = !DILocation(line: 58, column: 3, scope: !247)
// CHECK:STDOUT: !252 = !DILocation(line: 59, column: 3, scope: !247)
// CHECK:STDOUT: !254 = !DILocation(line: 61, column: 19, scope: !247)
// CHECK:STDOUT: !255 = !DILocation(line: 61, column: 27, scope: !247)
// CHECK:STDOUT: !257 = !DILocation(line: 61, column: 3, scope: !247)
// CHECK:STDOUT: !258 = !DILocation(line: 57, column: 1, scope: !247)
// CHECK:STDOUT: !259 = !{!248}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.68d4072289407214", scope: null, file: !10, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !279)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !23)
// CHECK:STDOUT: !278 = !DILocation(line: 24, column: 3, scope: !273)
// CHECK:STDOUT: !279 = !{!274}
// CHECK:STDOUT: !280 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.7a15f2f04d824f56", scope: null, file: !10, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !286)
// CHECK:STDOUT: !281 = !DILocalVariable(arg: 1, scope: !280, type: !23)
// CHECK:STDOUT: !285 = !DILocation(line: 24, column: 3, scope: !280)
// CHECK:STDOUT: !286 = !{!281}
// CHECK:STDOUT: !287 = distinct !DISubprogram(name: "second_function", linkageName: "_Csecond_function.Main.efd9fcdf727c9de5", scope: null, file: !14, line: 70, type: !39, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !309)
// CHECK:STDOUT: !288 = !DILocalVariable(arg: 1, scope: !287, type: !23)
// CHECK:STDOUT: !289 = !DILocalVariable(arg: 2, scope: !287, type: !23)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 3, scope: !287, type: !23)
// CHECK:STDOUT: !294 = !DILocation(line: 71, column: 3, scope: !287)
// CHECK:STDOUT: !296 = !DILocation(line: 72, column: 3, scope: !287)
// CHECK:STDOUT: !298 = !DILocation(line: 73, column: 3, scope: !287)
// CHECK:STDOUT: !301 = !DILocation(line: 75, column: 25, scope: !287)
// CHECK:STDOUT: !302 = !DILocation(line: 75, column: 33, scope: !287)
// CHECK:STDOUT: !303 = !DILocation(line: 75, column: 3, scope: !287)
// CHECK:STDOUT: !304 = !DILocation(line: 76, column: 18, scope: !287)
// CHECK:STDOUT: !307 = !DILocation(line: 76, column: 3, scope: !287)
// CHECK:STDOUT: !308 = !DILocation(line: 70, column: 1, scope: !287)
// CHECK:STDOUT: !309 = !{!288, !289, !290}
// CHECK:STDOUT: !356 = distinct !DISubprogram(name: "third_function", linkageName: "_Cthird_function.Main.1c39a3b349bf66b6", scope: null, file: !14, line: 35, type: !39, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !374)
// CHECK:STDOUT: !357 = !DILocalVariable(arg: 1, scope: !356, type: !23)
// CHECK:STDOUT: !358 = !DILocalVariable(arg: 2, scope: !356, type: !23)
// CHECK:STDOUT: !359 = !DILocalVariable(arg: 3, scope: !356, type: !23)
// CHECK:STDOUT: !363 = !DILocation(line: 36, column: 3, scope: !356)
// CHECK:STDOUT: !365 = !DILocation(line: 37, column: 3, scope: !356)
// CHECK:STDOUT: !369 = !DILocation(line: 39, column: 30, scope: !356)
// CHECK:STDOUT: !370 = !DILocation(line: 39, column: 3, scope: !356)
// CHECK:STDOUT: !371 = !DILocation(line: 40, column: 19, scope: !356)
// CHECK:STDOUT: !372 = !DILocation(line: 40, column: 3, scope: !356)
// CHECK:STDOUT: !373 = !DILocation(line: 35, column: 1, scope: !356)
// CHECK:STDOUT: !374 = !{!357, !358, !359}
// CHECK:STDOUT:
