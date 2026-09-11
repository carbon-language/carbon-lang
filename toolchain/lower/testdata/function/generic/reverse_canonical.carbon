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
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i1, align 1, !dbg !7
// CHECK:STDOUT:   %y.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !7
// CHECK:STDOUT:   store i8 poison, ptr %x.var, align 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %y.var), !dbg !8
// CHECK:STDOUT:   store ptr poison, ptr %y.var, align 8, !dbg !8
// CHECK:STDOUT:   %.loc47_9 = load i8, ptr %x.var, align 1, !dbg !9
// CHECK:STDOUT:   %.loc47_91 = trunc i8 %.loc47_9 to i1, !dbg !9
// CHECK:STDOUT:   %.loc47_12 = load i8, ptr %x.var, align 1, !dbg !10
// CHECK:STDOUT:   %.loc47_122 = trunc i8 %.loc47_12 to i1, !dbg !10
// CHECK:STDOUT:   %.loc47_15 = load ptr, ptr %y.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @_Cfirst.Main.36f5e0b0ce09cb9a(i1 %.loc47_91, i1 %.loc47_122, ptr %.loc47_15), !dbg !12
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.4d8205a9809188d0(ptr %y.var), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad5376756201bea4(ptr %x.var), !dbg !7
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
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !22
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !23
// CHECK:STDOUT:   call void @_Csecond.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc23), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.4d8205a9809188d0(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ad5376756201bea4(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !42
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !42
// CHECK:STDOUT:   call void @_Cfirst.Main.81f1c9de445b895c(ptr %arg3, ptr %arg3, i1 %arg1), !dbg !43
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !44
// CHECK:STDOUT:   call void @_Cthird.Main.0f29a29bf45af723(i1 %arg2, i1 %arg2, ptr %.loc30), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.81f1c9de445b895c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !56
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !56
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !57
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !58
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.0f29a29bf45af723(i1 %arg1, i1 %arg2, ptr %arg3) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !65
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !65
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !66
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !67
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %.loc36_10, ptr %.loc36_22, i1 %arg1), !dbg !68
// CHECK:STDOUT:   call void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg3, ptr %arg3, i1 %arg2), !dbg !69
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.30c57425aa4cd11c(ptr %local_name.var), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !76
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !76
// CHECK:STDOUT:   call void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg3, ptr %arg3, ptr %arg1), !dbg !77
// CHECK:STDOUT:   %.loc30 = load ptr, ptr %local_name.var, align 8, !dbg !78
// CHECK:STDOUT:   call void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc30), !dbg !79
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.c6787c0e2572c86c(ptr %arg1, ptr %arg2, i1 %arg3) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.30c57425aa4cd11c(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfirst.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !96
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !96
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %local_name.var, align 8, !dbg !97
// CHECK:STDOUT:   call void @_Csecond.Main.e1ce67f3c23ab61f(ptr %arg2, ptr %arg2, ptr %.loc23), !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.00ee02cee901e22f(ptr %local_name.var), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cthird.Main.e1ce67f3c23ab61f(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %local_name.var = alloca ptr, align 8, !dbg !105
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %local_name.var), !dbg !105
// CHECK:STDOUT:   store ptr poison, ptr %local_name.var, align 8, !dbg !105
// CHECK:STDOUT:   %.loc36_10 = load ptr, ptr %local_name.var, align 8, !dbg !106
// CHECK:STDOUT:   %.loc36_22 = load ptr, ptr %local_name.var, align 8, !dbg !107
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %.loc36_10, ptr %.loc36_22, ptr %arg1), !dbg !108
// CHECK:STDOUT:   call void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg3, ptr %arg3, ptr %arg2), !dbg !109
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.30c57425aa4cd11c(ptr %local_name.var), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_Cfourth.Main.901b24db5c2cfe42(ptr %arg1, ptr %arg2, ptr %arg3) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !116
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "reverse_canonical.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !1, line: 43, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 44, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 45, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 47, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 47, column: 12, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 47, column: 15, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 47, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 43, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.36f5e0b0ce09cb9a", scope: null, file: !1, line: 20, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17, !17, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !18 = !{!19, !20, !21}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 2, scope: !14, type: !17)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 3, scope: !14, type: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 21, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 23, column: 22, scope: !14)
// CHECK:STDOUT: !24 = !DILocation(line: 23, column: 3, scope: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 20, column: 1, scope: !14)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.4d8205a9809188d0", scope: null, file: !27, line: 24, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !27 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !17}
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !26, type: !17)
// CHECK:STDOUT: !32 = !DILocation(line: 24, column: 3, scope: !26)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ad5376756201bea4", scope: null, file: !27, line: 24, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !17)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 3, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.0f29a29bf45af723", scope: null, file: !1, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39, !40, !41}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !17)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 2, scope: !37, type: !17)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 3, scope: !37, type: !17)
// CHECK:STDOUT: !42 = !DILocation(line: 27, column: 3, scope: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 29, column: 3, scope: !37)
// CHECK:STDOUT: !44 = !DILocation(line: 30, column: 21, scope: !37)
// CHECK:STDOUT: !45 = !DILocation(line: 30, column: 3, scope: !37)
// CHECK:STDOUT: !46 = !DILocation(line: 26, column: 1, scope: !37)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.00ee02cee901e22f", scope: null, file: !27, line: 24, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !48)
// CHECK:STDOUT: !48 = !{!49}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !47, type: !17)
// CHECK:STDOUT: !50 = !DILocation(line: 24, column: 3, scope: !47)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.81f1c9de445b895c", scope: null, file: !1, line: 20, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !52)
// CHECK:STDOUT: !52 = !{!53, !54, !55}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !51, type: !17)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 2, scope: !51, type: !17)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 3, scope: !51, type: !17)
// CHECK:STDOUT: !56 = !DILocation(line: 21, column: 3, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 23, column: 22, scope: !51)
// CHECK:STDOUT: !58 = !DILocation(line: 23, column: 3, scope: !51)
// CHECK:STDOUT: !59 = !DILocation(line: 20, column: 1, scope: !51)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.0f29a29bf45af723", scope: null, file: !1, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !61)
// CHECK:STDOUT: !61 = !{!62, !63, !64}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !60, type: !17)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 2, scope: !60, type: !17)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 3, scope: !60, type: !17)
// CHECK:STDOUT: !65 = !DILocation(line: 34, column: 3, scope: !60)
// CHECK:STDOUT: !66 = !DILocation(line: 36, column: 10, scope: !60)
// CHECK:STDOUT: !67 = !DILocation(line: 36, column: 22, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 36, column: 3, scope: !60)
// CHECK:STDOUT: !69 = !DILocation(line: 37, column: 3, scope: !60)
// CHECK:STDOUT: !70 = !DILocation(line: 33, column: 1, scope: !60)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "second", linkageName: "_Csecond.Main.e1ce67f3c23ab61f", scope: null, file: !1, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !72)
// CHECK:STDOUT: !72 = !{!73, !74, !75}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !71, type: !17)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 2, scope: !71, type: !17)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 3, scope: !71, type: !17)
// CHECK:STDOUT: !76 = !DILocation(line: 27, column: 3, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 29, column: 3, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 30, column: 21, scope: !71)
// CHECK:STDOUT: !79 = !DILocation(line: 30, column: 3, scope: !71)
// CHECK:STDOUT: !80 = !DILocation(line: 26, column: 1, scope: !71)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.c6787c0e2572c86c", scope: null, file: !1, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83, !84, !85}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !17)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !81, type: !17)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 3, scope: !81, type: !17)
// CHECK:STDOUT: !86 = !DILocation(line: 40, column: 1, scope: !81)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.30c57425aa4cd11c", scope: null, file: !27, line: 24, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !17)
// CHECK:STDOUT: !90 = !DILocation(line: 24, column: 3, scope: !87)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "first", linkageName: "_Cfirst.Main.e1ce67f3c23ab61f", scope: null, file: !1, line: 20, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !92)
// CHECK:STDOUT: !92 = !{!93, !94, !95}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !91, type: !17)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 2, scope: !91, type: !17)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 3, scope: !91, type: !17)
// CHECK:STDOUT: !96 = !DILocation(line: 21, column: 3, scope: !91)
// CHECK:STDOUT: !97 = !DILocation(line: 23, column: 22, scope: !91)
// CHECK:STDOUT: !98 = !DILocation(line: 23, column: 3, scope: !91)
// CHECK:STDOUT: !99 = !DILocation(line: 20, column: 1, scope: !91)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "third", linkageName: "_Cthird.Main.e1ce67f3c23ab61f", scope: null, file: !1, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102, !103, !104}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !17)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 2, scope: !100, type: !17)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 3, scope: !100, type: !17)
// CHECK:STDOUT: !105 = !DILocation(line: 34, column: 3, scope: !100)
// CHECK:STDOUT: !106 = !DILocation(line: 36, column: 10, scope: !100)
// CHECK:STDOUT: !107 = !DILocation(line: 36, column: 22, scope: !100)
// CHECK:STDOUT: !108 = !DILocation(line: 36, column: 3, scope: !100)
// CHECK:STDOUT: !109 = !DILocation(line: 37, column: 3, scope: !100)
// CHECK:STDOUT: !110 = !DILocation(line: 33, column: 1, scope: !100)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "fourth", linkageName: "_Cfourth.Main.901b24db5c2cfe42", scope: null, file: !1, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !112)
// CHECK:STDOUT: !112 = !{!113, !114, !115}
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !111, type: !17)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 2, scope: !111, type: !17)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 3, scope: !111, type: !17)
// CHECK:STDOUT: !116 = !DILocation(line: 40, column: 1, scope: !111)
// CHECK:STDOUT:
