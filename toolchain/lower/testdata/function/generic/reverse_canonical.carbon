// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/reverse_canonical.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/reverse_canonical.carbon

fn first[U:! type, T:! type, V:! type](arg1 : U, arg2 : T, arg3 : V);
fn second[U:! type, T:! type, V:! type](arg1 : U, arg2 : T, arg3 : V);
fn third[U:! type, T:! type, V:! type](arg1 : U, arg2 : T, arg3 : V);
fn fourth[U:! type, T:! type, V:! type](arg1 : U, arg2 : T, arg3 : V);

fn first[U:! type, T:! type, V:! type](arg1 : U, arg2 : T, arg3 : V) {
  var local_name : i64*;

  second(arg2, arg2, local_name);
}

fn second[U:! type, T:! type, V:! type](arg1 : U, arg2 : T, arg3 : V) {
  var local_name : i64*;

  first(arg3, arg3, arg1);
  third(arg2, arg2, local_name);
}

fn third[U:! type, T:! type, V:! type](arg1 : U, arg2 : T, arg3 : V) {
  var local_name : i64**;

  fourth(local_name, local_name, arg1);
  fourth(arg3, arg3, arg2);
}

fn fourth[U:! type, T:! type, V:! type](arg1 : U, arg2 : T, arg3 : V) {
}

fn Main() {
  var x : bool;
  var y : bool*;

  first( x, x, y);
}

// CHECK:STDOUT: ; ModuleID = 'reverse_canonical.carbon'
// CHECK:STDOUT: source_filename = "reverse_canonical.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMain.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i1, align 1, !dbg !7
// CHECK:STDOUT:   %y.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !8
// CHECK:STDOUT:   %.loc45_10 = load i8, ptr %x.var, align 1, !dbg !9
// CHECK:STDOUT:   %.loc45_101 = trunc i8 %.loc45_10 to i1, !dbg !9
// CHECK:STDOUT:   %.loc45_13 = load i8, ptr %x.var, align 1, !dbg !10
// CHECK:STDOUT:   %.loc45_132 = trunc i8 %.loc45_13 to i1, !dbg !10
// CHECK:STDOUT:   %.loc45_16 = load ptr, ptr %y.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %.loc45_101, i1 %.loc45_132, ptr %.loc45_16), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %arg1, i1 %arg2, ptr %arg3) !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !15
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !16
// CHECK:STDOUT:   call void @_Csecond.Main.3084177407dea1a2(i1 %arg2, i1 %arg2, ptr %.loc21), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.3084177407dea1a2(i1 %arg1, i1 %arg2, ptr %arg3) !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !20
// CHECK:STDOUT:   call void @_Cfirst.Main.b7d9b71c41f2cb83(ptr %arg3, ptr %arg3, i1 %arg1), !dbg !21
// CHECK:STDOUT:   %.loc28 = load ptr, ptr %local_name.var, align 8, !dbg !22
// CHECK:STDOUT:   call void @_Cthird.Main.3084177407dea1a2(i1 %arg2, i1 %arg2, ptr %.loc28), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.b7d9b71c41f2cb83(ptr %arg1, ptr %arg2, i1 %arg3) !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !26
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !27
// CHECK:STDOUT:   call void @_Csecond.Main.43323db63663b108(ptr %arg2, ptr %arg2, ptr %.loc21), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.3084177407dea1a2(i1 %arg1, i1 %arg2, ptr %arg3) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !31
// CHECK:STDOUT:   %.loc34_10 = load ptr, ptr %local_name.var, align 8, !dbg !32
// CHECK:STDOUT:   %.loc34_22 = load ptr, ptr %local_name.var, align 8, !dbg !33
// CHECK:STDOUT:   call void @_Cfourth.Main.0b2b125e0238cb18(ptr %.loc34_10, ptr %.loc34_22, i1 %arg1), !dbg !34
// CHECK:STDOUT:   call void @_Cfourth.Main.0b2b125e0238cb18(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.43323db63663b108(ptr %arg1, ptr %arg2, ptr %arg3) !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !38
// CHECK:STDOUT:   call void @_Cfirst.Main.43323db63663b108(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !39
// CHECK:STDOUT:   %.loc28 = load ptr, ptr %local_name.var, align 8, !dbg !40
// CHECK:STDOUT:   call void @_Cthird.Main.43323db63663b108(ptr %arg2, ptr %arg2, ptr %.loc28), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.0b2b125e0238cb18(ptr %arg1, ptr %arg2, i1 %arg3) !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.43323db63663b108(ptr %arg1, ptr %arg2, ptr %arg3) !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !46
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %local_name.var, align 8, !dbg !47
// CHECK:STDOUT:   call void @_Csecond.Main.43323db63663b108(ptr %arg2, ptr %arg2, ptr %.loc21), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.43323db63663b108(ptr %arg1, ptr %arg2, ptr %arg3) !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !51
// CHECK:STDOUT:   %.loc34_10 = load ptr, ptr %local_name.var, align 8, !dbg !52
// CHECK:STDOUT:   %.loc34_22 = load ptr, ptr %local_name.var, align 8, !dbg !53
// CHECK:STDOUT:   call void @_Cfourth.Main.a79789f6d6156576(ptr %.loc34_10, ptr %.loc34_22, ptr %arg1), !dbg !54
// CHECK:STDOUT:   call void @_Cfourth.Main.a79789f6d6156576(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.a79789f6d6156576(ptr %arg1, ptr %arg2, ptr %arg3) !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 8, 7 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.0b2b125e0238cb18, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.a79789f6d6156576, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "reverse_canonical.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 41, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 42, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 43, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 45, column: 10, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 45, column: 13, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 45, column: 16, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 45, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 41, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 19, column: 3, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 21, column: 22, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 21, column: 3, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 1, scope: !14)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.3084177407dea1a2", scope: null, file: !3, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DILocation(line: 25, column: 3, scope: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 27, column: 3, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 28, column: 21, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 28, column: 3, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 24, column: 1, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.b7d9b71c41f2cb83", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 3, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 21, column: 22, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 21, column: 3, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 18, column: 1, scope: !25)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.3084177407dea1a2", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 32, column: 3, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 34, column: 10, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 34, column: 22, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 34, column: 3, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 35, column: 3, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 31, column: 1, scope: !30)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.43323db63663b108", scope: null, file: !3, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !38 = !DILocation(line: 25, column: 3, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 27, column: 3, scope: !37)
// CHECK:STDOUT: !40 = !DILocation(line: 28, column: 21, scope: !37)
// CHECK:STDOUT: !41 = !DILocation(line: 28, column: 3, scope: !37)
// CHECK:STDOUT: !42 = !DILocation(line: 24, column: 1, scope: !37)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.0b2b125e0238cb18", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !44 = !DILocation(line: 38, column: 1, scope: !43)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.43323db63663b108", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 3, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 21, column: 22, scope: !45)
// CHECK:STDOUT: !48 = !DILocation(line: 21, column: 3, scope: !45)
// CHECK:STDOUT: !49 = !DILocation(line: 18, column: 1, scope: !45)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.43323db63663b108", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !51 = !DILocation(line: 32, column: 3, scope: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 34, column: 10, scope: !50)
// CHECK:STDOUT: !53 = !DILocation(line: 34, column: 22, scope: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 34, column: 3, scope: !50)
// CHECK:STDOUT: !55 = !DILocation(line: 35, column: 3, scope: !50)
// CHECK:STDOUT: !56 = !DILocation(line: 31, column: 1, scope: !50)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.a79789f6d6156576", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !58 = !DILocation(line: 38, column: 1, scope: !57)
