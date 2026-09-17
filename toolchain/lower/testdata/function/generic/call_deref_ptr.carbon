// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_deref_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_deref_ptr.carbon

// Builds on call_basic.carbon and call_impl_function.carbon.
// Test that when equivalence is found for the specific fingerprint, we don't
// conclude "equivalent" for all the call graph. This is to account for
// switching to a work-queue from recursion.

fn A[U: Core.Copy](x: U) -> U {
  return x;
}

fn D[U: Core.Copy](x: U) -> U {
  return x;
}

fn B[U: Core.Copy & Core.Destroy](x: U*) {
  D(*x);
}

fn F[T: Core.Copy & Core.Destroy](x: T*) {
  A(i32);
  B(x);
}

fn M() {
  var ptr_i32 : i32*;
  var ptr_f64 : f64*;

  F(ptr_i32);
  F(ptr_f64);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_deref_ptr.carbon'
// CHECK:STDOUT: source_filename = "call_deref_ptr.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type.26 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !43
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !43
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !45
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !45
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !47
// CHECK:STDOUT:   call void @_CF.Main.5216072513bfc692(ptr %.loc39_5), !dbg !48
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !49
// CHECK:STDOUT:   call void @_CF.Main.f34906dc59cae110(ptr %.loc40_5), !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %ptr_f64.var), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %ptr_i32.var), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.5216072513bfc692(ptr %x) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.26 @_CA.Main.71d6c598da8ca542(%type.26 zeroinitializer), !dbg !63
// CHECK:STDOUT:   call void @_CB.Main.5216072513bfc692(ptr %x), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.f34906dc59cae110(ptr %x) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.26 @_CA.Main.71d6c598da8ca542(%type.26 zeroinitializer), !dbg !73
// CHECK:STDOUT:   call void @_CB.Main.f34906dc59cae110(ptr %x), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.9f1044688d586372(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f427c19f24d378eb(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.26 @_CA.Main.71d6c598da8ca542(%type.26 %x) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.26 %x, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.5216072513bfc692(ptr %x) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_7.3.temp = alloca i32, align 4, !dbg !105
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_7.3.temp), !dbg !105
// CHECK:STDOUT:   %.loc27_5.2 = load i32, ptr %x, align 4, !dbg !104
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %.loc27_5.2), !dbg !105
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc27_7.3.temp, align 4, !dbg !105
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %.loc27_7.3.temp), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.f34906dc59cae110(ptr %x) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_7.3.temp = alloca double, align 8, !dbg !115
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_7.3.temp), !dbg !115
// CHECK:STDOUT:   %.loc27_5.2 = load double, ptr %x, align 8, !dbg !114
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %.loc27_5.2), !dbg !115
// CHECK:STDOUT:   store double %D.call, ptr %.loc27_7.3.temp, align 8, !dbg !115
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %.loc27_7.3.temp), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !126 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.23895ff6626cffe1(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core"(ptr %self), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.71d6c598da8ca542, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !32 = !DIFile(filename: "call_deref_ptr.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{null}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !34}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !32, line: 35, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !43 = !DILocation(line: 36, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !DILocation(line: 37, column: 3, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 39, column: 5, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 39, column: 3, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 40, column: 5, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 40, column: 3, scope: !42)
// CHECK:STDOUT: !51 = !DILocation(line: 35, column: 1, scope: !42)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !32, line: 39, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !34)
// CHECK:STDOUT: !54 = !DILocation(line: 39, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !32, line: 40, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !39)
// CHECK:STDOUT: !58 = !DILocation(line: 40, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5216072513bfc692", scope: null, file: !32, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !69)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !39)
// CHECK:STDOUT: !63 = !DILocation(line: 31, column: 3, scope: !60)
// CHECK:STDOUT: !64 = !{!39, !39}
// CHECK:STDOUT: !65 = !DISubroutineType(types: !64)
// CHECK:STDOUT: !67 = !DILocation(line: 32, column: 3, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 30, column: 1, scope: !60)
// CHECK:STDOUT: !69 = !{!61}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f34906dc59cae110", scope: null, file: !32, line: 30, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !77)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !39)
// CHECK:STDOUT: !73 = !DILocation(line: 31, column: 3, scope: !70)
// CHECK:STDOUT: !75 = !DILocation(line: 32, column: 3, scope: !70)
// CHECK:STDOUT: !76 = !DILocation(line: 30, column: 1, scope: !70)
// CHECK:STDOUT: !77 = !{!71}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.9f1044688d586372", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !84)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !39)
// CHECK:STDOUT: !83 = !DILocation(line: 24, column: 3, scope: !78)
// CHECK:STDOUT: !84 = !{!79}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f427c19f24d378eb", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !91)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !39)
// CHECK:STDOUT: !90 = !DILocation(line: 24, column: 3, scope: !85)
// CHECK:STDOUT: !91 = !{!86}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.71d6c598da8ca542", scope: null, file: !32, line: 18, type: !65, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !99)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !39)
// CHECK:STDOUT: !98 = !DILocation(line: 19, column: 3, scope: !92)
// CHECK:STDOUT: !99 = !{!93}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.5216072513bfc692", scope: null, file: !32, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !109)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !39)
// CHECK:STDOUT: !104 = !DILocation(line: 27, column: 5, scope: !100)
// CHECK:STDOUT: !105 = !DILocation(line: 27, column: 3, scope: !100)
// CHECK:STDOUT: !106 = !{!34, !34}
// CHECK:STDOUT: !107 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !108 = !DILocation(line: 26, column: 1, scope: !100)
// CHECK:STDOUT: !109 = !{!101}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.f34906dc59cae110", scope: null, file: !32, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !117)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !39)
// CHECK:STDOUT: !114 = !DILocation(line: 27, column: 5, scope: !110)
// CHECK:STDOUT: !115 = !DILocation(line: 27, column: 3, scope: !110)
// CHECK:STDOUT: !116 = !DILocation(line: 26, column: 1, scope: !110)
// CHECK:STDOUT: !117 = !{!111}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !32, line: 22, type: !107, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !125)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !34)
// CHECK:STDOUT: !124 = !DILocation(line: 23, column: 3, scope: !118)
// CHECK:STDOUT: !125 = !{!119}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !28, line: 24, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !132)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !34)
// CHECK:STDOUT: !130 = !DILocation(line: 26, column: 5, scope: !126)
// CHECK:STDOUT: !131 = !DILocation(line: 24, column: 3, scope: !126)
// CHECK:STDOUT: !132 = !{!127}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !32, line: 22, type: !65, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !140)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !39)
// CHECK:STDOUT: !139 = !DILocation(line: 23, column: 3, scope: !133)
// CHECK:STDOUT: !140 = !{!134}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.23895ff6626cffe1", scope: null, file: !28, line: 24, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !147)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !39)
// CHECK:STDOUT: !145 = !DILocation(line: 26, column: 5, scope: !141)
// CHECK:STDOUT: !146 = !DILocation(line: 24, column: 3, scope: !141)
// CHECK:STDOUT: !147 = !{!142}
// CHECK:STDOUT:
