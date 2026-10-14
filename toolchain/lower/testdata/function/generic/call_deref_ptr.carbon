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
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !40
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !40
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !42
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !42
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !44
// CHECK:STDOUT:   call void @_CF.Main.8fbe4c345ab91e37(ptr %.loc39_5), !dbg !45
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !49
// CHECK:STDOUT:   call void @_CF.Main.40206c9f9f46679a(ptr %.loc40_5), !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.81b4e30e2d9fb9f0(ptr %ptr_f64.var), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.39f93332b5bec0ba(ptr %ptr_i32.var), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.8fbe4c345ab91e37(ptr %x) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.28 @_CA.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !55
// CHECK:STDOUT:   call void @_CB.Main.8fbe4c345ab91e37(ptr %x), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.40206c9f9f46679a(ptr %x) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.28 @_CA.Main.c3359c2c217ad593(%type.28 zeroinitializer), !dbg !65
// CHECK:STDOUT:   call void @_CB.Main.40206c9f9f46679a(ptr %x), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.81b4e30e2d9fb9f0(ptr %t) #0 !dbg !70 {
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.39f93332b5bec0ba(ptr %t) #0 !dbg !77 {
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.28 @_CA.Main.c3359c2c217ad593(%type.28 %x) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.28 %x, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.8fbe4c345ab91e37(ptr %x) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_7.3.temp = alloca i32, align 4, !dbg !97
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_7.3.temp), !dbg !97
// CHECK:STDOUT:   %.loc27_5.2 = load i32, ptr %x, align 4, !dbg !96
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2aef1d52a6e1f957(i32 %.loc27_5.2), !dbg !97
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc27_7.3.temp, align 4, !dbg !97
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %.loc27_7.3.temp), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.40206c9f9f46679a(ptr %x) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_7.3.temp = alloca double, align 8, !dbg !109
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_7.3.temp), !dbg !109
// CHECK:STDOUT:   %.loc27_5.2 = load double, ptr %x, align 8, !dbg !108
// CHECK:STDOUT:   %D.call = call double @_CD.Main.cff5c31fb9f8fd40(double %.loc27_5.2), !dbg !109
// CHECK:STDOUT:   store double %D.call, ptr %.loc27_7.3.temp, align 8, !dbg !109
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %.loc27_7.3.temp), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.2aef1d52a6e1f957(i32 %x) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %t) #0 !dbg !120 {
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.cff5c31fb9f8fd40(double %x) #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.236ebccb4c02fafd(ptr %t) #0 !dbg !135 {
// CHECK:STDOUT:   ret void, !dbg !140
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
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "call_deref_ptr.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !34, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 36, column: 3, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 37, column: 3, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 39, column: 5, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 39, column: 3, scope: !39)
// CHECK:STDOUT: !46 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !47 = !{null, !46}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 40, column: 5, scope: !39)
// CHECK:STDOUT: !50 = !DILocation(line: 40, column: 3, scope: !39)
// CHECK:STDOUT: !51 = !DILocation(line: 35, column: 1, scope: !39)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.8fbe4c345ab91e37", scope: null, file: !34, line: 30, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !61)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !46)
// CHECK:STDOUT: !55 = !DILocation(line: 31, column: 3, scope: !52)
// CHECK:STDOUT: !56 = !{!46, !46}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !59 = !DILocation(line: 32, column: 3, scope: !52)
// CHECK:STDOUT: !60 = !DILocation(line: 30, column: 1, scope: !52)
// CHECK:STDOUT: !61 = !{!53}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.40206c9f9f46679a", scope: null, file: !34, line: 30, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !69)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !46)
// CHECK:STDOUT: !65 = !DILocation(line: 31, column: 3, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 32, column: 3, scope: !62)
// CHECK:STDOUT: !68 = !DILocation(line: 30, column: 1, scope: !62)
// CHECK:STDOUT: !69 = !{!63}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.81b4e30e2d9fb9f0", scope: null, file: !26, line: 30, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !76)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !46)
// CHECK:STDOUT: !75 = !DILocation(line: 30, column: 1, scope: !70)
// CHECK:STDOUT: !76 = !{!71}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.39f93332b5bec0ba", scope: null, file: !26, line: 30, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !83)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !46)
// CHECK:STDOUT: !82 = !DILocation(line: 30, column: 1, scope: !77)
// CHECK:STDOUT: !83 = !{!78}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c3359c2c217ad593", scope: null, file: !34, line: 18, type: !57, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !91)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !46)
// CHECK:STDOUT: !90 = !DILocation(line: 19, column: 3, scope: !84)
// CHECK:STDOUT: !91 = !{!85}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.8fbe4c345ab91e37", scope: null, file: !34, line: 26, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !103)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !46)
// CHECK:STDOUT: !96 = !DILocation(line: 27, column: 5, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 27, column: 3, scope: !92)
// CHECK:STDOUT: !98 = !{!36, !36}
// CHECK:STDOUT: !99 = !DISubroutineType(types: !98)
// CHECK:STDOUT: !100 = !{null, !36}
// CHECK:STDOUT: !101 = !DISubroutineType(types: !100)
// CHECK:STDOUT: !102 = !DILocation(line: 26, column: 1, scope: !92)
// CHECK:STDOUT: !103 = !{!93}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.40206c9f9f46679a", scope: null, file: !34, line: 26, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !111)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !46)
// CHECK:STDOUT: !108 = !DILocation(line: 27, column: 5, scope: !104)
// CHECK:STDOUT: !109 = !DILocation(line: 27, column: 3, scope: !104)
// CHECK:STDOUT: !110 = !DILocation(line: 26, column: 1, scope: !104)
// CHECK:STDOUT: !111 = !{!105}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.2aef1d52a6e1f957", scope: null, file: !34, line: 22, type: !99, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !119)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !36)
// CHECK:STDOUT: !118 = !DILocation(line: 23, column: 3, scope: !112)
// CHECK:STDOUT: !119 = !{!113}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7f8dce37f94a83bf", scope: null, file: !26, line: 30, type: !101, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !126)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !36)
// CHECK:STDOUT: !125 = !DILocation(line: 30, column: 1, scope: !120)
// CHECK:STDOUT: !126 = !{!121}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.cff5c31fb9f8fd40", scope: null, file: !34, line: 22, type: !57, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !134)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !46)
// CHECK:STDOUT: !133 = !DILocation(line: 23, column: 3, scope: !127)
// CHECK:STDOUT: !134 = !{!128}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.236ebccb4c02fafd", scope: null, file: !26, line: 30, type: !48, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !141)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !46)
// CHECK:STDOUT: !140 = !DILocation(line: 30, column: 1, scope: !135)
// CHECK:STDOUT: !141 = !{!136}
// CHECK:STDOUT:
