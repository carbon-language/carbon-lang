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
// CHECK:STDOUT: %type.28 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !45
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !45
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !47
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !47
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !49
// CHECK:STDOUT:   call void @_CF.Main.64609c610134ef75(ptr %.loc39_5), !dbg !50
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !51
// CHECK:STDOUT:   call void @_CF.Main.862050cdacea1015(ptr %.loc40_5), !dbg !52
// CHECK:STDOUT:   call void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !47
// CHECK:STDOUT:   call void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.64609c610134ef75(ptr %x) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.28 @_CA.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !73
// CHECK:STDOUT:   call void @_CB.Main.64609c610134ef75(ptr %x), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.862050cdacea1015(ptr %x) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.28 @_CA.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !83
// CHECK:STDOUT:   call void @_CB.Main.862050cdacea1015(ptr %x), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CA.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.28 %x, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.64609c610134ef75(ptr %x) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_7.3.temp = alloca i32, align 4, !dbg !101
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_7.3.temp), !dbg !101
// CHECK:STDOUT:   %.loc27_5.2 = load i32, ptr %x, align 4, !dbg !100
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.3094c9a3a4d84642(i32 %.loc27_5.2), !dbg !101
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc27_7.3.temp, align 4, !dbg !101
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc27_7.3.temp), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.862050cdacea1015(ptr %x) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_7.3.temp = alloca double, align 8, !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_7.3.temp), !dbg !111
// CHECK:STDOUT:   %.loc27_5.2 = load double, ptr %x, align 8, !dbg !110
// CHECK:STDOUT:   %D.call = call double @_CD.Main.d30a9b8c414024b6(double %.loc27_5.2), !dbg !111
// CHECK:STDOUT:   store double %D.call, ptr %.loc27_7.3.temp, align 8, !dbg !111
// CHECK:STDOUT:   call void @"_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core"(ptr %.loc27_7.3.temp), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.3094c9a3a4d84642(i32 %x) #0 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.d30a9b8c414024b6(double %x) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.c3359c2c217ad593, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "call_deref_ptr.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{null, !36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{null, !41}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !34, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !45 = !DILocation(line: 36, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 37, column: 3, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 39, column: 5, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 39, column: 3, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 40, column: 5, scope: !44)
// CHECK:STDOUT: !52 = !DILocation(line: 40, column: 3, scope: !44)
// CHECK:STDOUT: !53 = !DILocation(line: 35, column: 1, scope: !44)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !34, line: 39, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !36)
// CHECK:STDOUT: !56 = !DILocation(line: 39, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.84c40a423a4f9e08:core.Destroy.Core", scope: null, file: !34, line: 40, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !41)
// CHECK:STDOUT: !60 = !DILocation(line: 40, column: 3, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core", scope: null, file: !34, line: 37, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !41)
// CHECK:STDOUT: !64 = !DILocation(line: 37, column: 3, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1feff81df232a20a:core.Destroy.Core", scope: null, file: !34, line: 36, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !41)
// CHECK:STDOUT: !68 = !DILocation(line: 36, column: 3, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.64609c610134ef75", scope: null, file: !34, line: 30, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !79)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !41)
// CHECK:STDOUT: !73 = !DILocation(line: 31, column: 3, scope: !70)
// CHECK:STDOUT: !74 = !{!41, !41}
// CHECK:STDOUT: !75 = !DISubroutineType(types: !74)
// CHECK:STDOUT: !77 = !DILocation(line: 32, column: 3, scope: !70)
// CHECK:STDOUT: !78 = !DILocation(line: 30, column: 1, scope: !70)
// CHECK:STDOUT: !79 = !{!71}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.862050cdacea1015", scope: null, file: !34, line: 30, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !87)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !41)
// CHECK:STDOUT: !83 = !DILocation(line: 31, column: 3, scope: !80)
// CHECK:STDOUT: !85 = !DILocation(line: 32, column: 3, scope: !80)
// CHECK:STDOUT: !86 = !DILocation(line: 30, column: 1, scope: !80)
// CHECK:STDOUT: !87 = !{!81}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c3359c2c217ad593", scope: null, file: !34, line: 18, type: !75, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !95)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !41)
// CHECK:STDOUT: !94 = !DILocation(line: 19, column: 3, scope: !88)
// CHECK:STDOUT: !95 = !{!89}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.64609c610134ef75", scope: null, file: !34, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !105)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !41)
// CHECK:STDOUT: !100 = !DILocation(line: 27, column: 5, scope: !96)
// CHECK:STDOUT: !101 = !DILocation(line: 27, column: 3, scope: !96)
// CHECK:STDOUT: !102 = !{!36, !36}
// CHECK:STDOUT: !103 = !DISubroutineType(types: !102)
// CHECK:STDOUT: !104 = !DILocation(line: 26, column: 1, scope: !96)
// CHECK:STDOUT: !105 = !{!97}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.862050cdacea1015", scope: null, file: !34, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !113)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !41)
// CHECK:STDOUT: !110 = !DILocation(line: 27, column: 5, scope: !106)
// CHECK:STDOUT: !111 = !DILocation(line: 27, column: 3, scope: !106)
// CHECK:STDOUT: !112 = !DILocation(line: 26, column: 1, scope: !106)
// CHECK:STDOUT: !113 = !{!107}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.3094c9a3a4d84642", scope: null, file: !34, line: 22, type: !103, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !121)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !36)
// CHECK:STDOUT: !120 = !DILocation(line: 23, column: 3, scope: !114)
// CHECK:STDOUT: !121 = !{!115}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.d30a9b8c414024b6", scope: null, file: !34, line: 22, type: !75, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !129)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !41)
// CHECK:STDOUT: !128 = !DILocation(line: 23, column: 3, scope: !122)
// CHECK:STDOUT: !129 = !{!123}
// CHECK:STDOUT:
