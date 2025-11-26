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

fn first[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V);
fn second[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V);
fn third[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V);
fn fourth[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V);

fn first[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V) {
  var local_name: C1*;

  second(arg2, arg2, local_name);
}

fn second[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V) {
  var local_name: C1*;

  first(arg3, arg3, arg1);
  third(arg2, arg2, local_name);
}

fn third[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V) {
  var local_name: C1**;

  fourth(local_name, local_name, arg1);
  fourth(arg3, arg3, arg2);
}

fn fourth[T:! type, U:! type, V:! type](arg1: T, arg2: U, arg3: V) {
}

fn Main() {
  var x: bool;
  var y: bool*;

  first(x, x, y);
}

// CHECK:STDOUT: ; ModuleID = 'reverse_canonical.carbon'
// CHECK:STDOUT: source_filename = "reverse_canonical.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i1, align 1, !dbg !7
// CHECK:STDOUT:   %y.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !8
// CHECK:STDOUT:   %.loc47_9 = load i8, ptr %x.var, align 1, !dbg !9
// CHECK:STDOUT:   %.loc47_91 = trunc i8 %.loc47_9 to i1, !dbg !9
// CHECK:STDOUT:   %.loc47_12 = load i8, ptr %x.var, align 1, !dbg !10
// CHECK:STDOUT:   %.loc47_122 = trunc i8 %.loc47_12 to i1, !dbg !10
// CHECK:STDOUT:   %.loc47_15 = load ptr, ptr %y.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %.loc47_91, i1 %.loc47_122, ptr %.loc47_15), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !22
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !23
// CHECK:STDOUT:   call void @_Csecond.Main.e20e3db21e34eb80(i1 %arg2, i1 %arg2, ptr %.loc23), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.e20e3db21e34eb80(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !31
// CHECK:STDOUT:   call void @_Cfirst.Main.140560e2430fc2f7(ptr %arg3, ptr %arg3, i1 %arg1), !dbg !32
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !33
// CHECK:STDOUT:   call void @_Cthird.Main.e20e3db21e34eb80(i1 %arg2, i1 %arg2, ptr %.loc30), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.140560e2430fc2f7(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !41
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !42
// CHECK:STDOUT:   call void @_Csecond.Main.f9ad2587bfb58bac(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.e20e3db21e34eb80(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !50
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !51
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !52
// CHECK:STDOUT:   call void @_Cfourth.Main.9260a3b095cb39ed(ptr %.loc36_10, ptr %.loc36_22, i1 %arg1), !dbg !53
// CHECK:STDOUT:   call void @_Cfourth.Main.9260a3b095cb39ed(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.f9ad2587bfb58bac(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !61
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !61
// CHECK:STDOUT:   call void @_Cfirst.Main.f9ad2587bfb58bac(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !62
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !63
// CHECK:STDOUT:   call void @_Cthird.Main.f9ad2587bfb58bac(ptr %arg2, ptr %arg2, ptr %.loc30), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.9260a3b095cb39ed(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.f9ad2587bfb58bac(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !77
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !78
// CHECK:STDOUT:   call void @_Csecond.Main.f9ad2587bfb58bac(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.f9ad2587bfb58bac(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !86
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !87
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !88
// CHECK:STDOUT:   call void @_Cfourth.Main.880241312efeb323(ptr %.loc36_10, ptr %.loc36_22, ptr %arg1), !dbg !89
// CHECK:STDOUT:   call void @_Cfourth.Main.880241312efeb323(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.880241312efeb323(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 8, 7 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.9260a3b095cb39ed, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_Cfourth.Main.880241312efeb323, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "reverse_canonical.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 43, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 44, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 45, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 47, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 47, column: 12, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 47, column: 15, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 47, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 43, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !3, line: 20, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17, !17, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !18 = !{!19, !20, !21}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 2, scope: !14, type: !17)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 3, scope: !14, type: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 21, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 23, column: 22, scope: !14)
// CHECK:STDOUT: !24 = !DILocation(line: 23, column: 3, scope: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 20, column: 1, scope: !14)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.e20e3db21e34eb80", scope: null, file: !3, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28, !29, !30}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !17)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !26, type: !17)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 3, scope: !26, type: !17)
// CHECK:STDOUT: !31 = !DILocation(line: 27, column: 3, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 29, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 30, column: 21, scope: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 30, column: 3, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 26, column: 1, scope: !26)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.140560e2430fc2f7", scope: null, file: !3, line: 20, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38, !39, !40}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !17)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !36, type: !17)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 3, scope: !36, type: !17)
// CHECK:STDOUT: !41 = !DILocation(line: 21, column: 3, scope: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 23, column: 22, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 23, column: 3, scope: !36)
// CHECK:STDOUT: !44 = !DILocation(line: 20, column: 1, scope: !36)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.e20e3db21e34eb80", scope: null, file: !3, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47, !48, !49}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !17)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 2, scope: !45, type: !17)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 3, scope: !45, type: !17)
// CHECK:STDOUT: !50 = !DILocation(line: 34, column: 3, scope: !45)
// CHECK:STDOUT: !51 = !DILocation(line: 36, column: 10, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 36, column: 22, scope: !45)
// CHECK:STDOUT: !53 = !DILocation(line: 36, column: 3, scope: !45)
// CHECK:STDOUT: !54 = !DILocation(line: 37, column: 3, scope: !45)
// CHECK:STDOUT: !55 = !DILocation(line: 33, column: 1, scope: !45)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.f9ad2587bfb58bac", scope: null, file: !3, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !57)
// CHECK:STDOUT: !57 = !{!58, !59, !60}
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !56, type: !17)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 2, scope: !56, type: !17)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 3, scope: !56, type: !17)
// CHECK:STDOUT: !61 = !DILocation(line: 27, column: 3, scope: !56)
// CHECK:STDOUT: !62 = !DILocation(line: 29, column: 3, scope: !56)
// CHECK:STDOUT: !63 = !DILocation(line: 30, column: 21, scope: !56)
// CHECK:STDOUT: !64 = !DILocation(line: 30, column: 3, scope: !56)
// CHECK:STDOUT: !65 = !DILocation(line: 26, column: 1, scope: !56)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.9260a3b095cb39ed", scope: null, file: !3, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !67)
// CHECK:STDOUT: !67 = !{!68, !69, !70}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !66, type: !17)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 2, scope: !66, type: !17)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 3, scope: !66, type: !17)
// CHECK:STDOUT: !71 = !DILocation(line: 40, column: 1, scope: !66)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.f9ad2587bfb58bac", scope: null, file: !3, line: 20, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !73)
// CHECK:STDOUT: !73 = !{!74, !75, !76}
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !72, type: !17)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 2, scope: !72, type: !17)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 3, scope: !72, type: !17)
// CHECK:STDOUT: !77 = !DILocation(line: 21, column: 3, scope: !72)
// CHECK:STDOUT: !78 = !DILocation(line: 23, column: 22, scope: !72)
// CHECK:STDOUT: !79 = !DILocation(line: 23, column: 3, scope: !72)
// CHECK:STDOUT: !80 = !DILocation(line: 20, column: 1, scope: !72)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.f9ad2587bfb58bac", scope: null, file: !3, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83, !84, !85}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !17)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !81, type: !17)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 3, scope: !81, type: !17)
// CHECK:STDOUT: !86 = !DILocation(line: 34, column: 3, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 36, column: 10, scope: !81)
// CHECK:STDOUT: !88 = !DILocation(line: 36, column: 22, scope: !81)
// CHECK:STDOUT: !89 = !DILocation(line: 36, column: 3, scope: !81)
// CHECK:STDOUT: !90 = !DILocation(line: 37, column: 3, scope: !81)
// CHECK:STDOUT: !91 = !DILocation(line: 33, column: 1, scope: !81)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.880241312efeb323", scope: null, file: !3, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94, !95, !96}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !17)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 2, scope: !92, type: !17)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 3, scope: !92, type: !17)
// CHECK:STDOUT: !97 = !DILocation(line: 40, column: 1, scope: !92)
