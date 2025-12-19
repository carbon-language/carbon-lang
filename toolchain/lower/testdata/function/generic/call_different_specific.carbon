// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_different_specific.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_different_specific.carbon

// Builds on call_basic.carbon and call_impl_function.carbon.
// Test that when equivalence is found for the specific fingerprint, we don't
// conclude "equivalent" for all the call graph.

// The two calls to F have the same function type, but the corresponding
// definitions should not be coalesced. The function bodies of the two
// specifics of F look mostly identical: same type for the parameters,
// calling the same A and different specifics of B, with the same function
// type. The difference is discovered in the call graph inside the two B
// specifics, where the pointer values are different, and the calls to the
// two D specifics are to functions with different function types.

fn A[U:! Core.Copy](x: U) -> U {
  return x;
}

fn D[U:! Core.Copy](x: U) -> U {
  return x;
}

fn B[U:! Core.Copy & Core.Destroy](x: U*) {
  D(*x);
}

fn F[T:! Core.Copy & Core.Destroy](x: T*) {
  A(i32);
  B(x);
}

fn M() {
  var ptr_i32 : i32*;
  var ptr_f64 : f64*;

  F(ptr_i32);
  F(ptr_f64);
}

// CHECK:STDOUT: ; ModuleID = 'call_different_specific.carbon'
// CHECK:STDOUT: source_filename = "call_different_specific.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !8
// CHECK:STDOUT:   %.loc46_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !9
// CHECK:STDOUT:   call void @_CF.Main.329e9a7481f16207(ptr %.loc46_5), !dbg !10
// CHECK:STDOUT:   %.loc47_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @_CF.Main.7220df690d061955(ptr %.loc47_5), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.329e9a7481f16207(ptr %x) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type @_CA.Main.582d60b54baf6b63(%type zeroinitializer), !dbg !20
// CHECK:STDOUT:   call void @_CB.Main.329e9a7481f16207(ptr %x), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.7220df690d061955(ptr %x) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type @_CA.Main.582d60b54baf6b63(%type zeroinitializer), !dbg !26
// CHECK:STDOUT:   call void @_CB.Main.7220df690d061955(ptr %x), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type @_CA.Main.582d60b54baf6b63(%type %x) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.329e9a7481f16207(ptr %x) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.3.temp = alloca i32, align 4, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.3.temp), !dbg !38
// CHECK:STDOUT:   %.loc34_5.2 = load i32, ptr %x, align 4, !dbg !39
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.64ccbb8e5d9a0b8e(i32 %.loc34_5.2), !dbg !38
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc34_7.3.temp, align 4, !dbg !38
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.7220df690d061955(ptr %x) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.3.temp = alloca double, align 8, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.3.temp), !dbg !44
// CHECK:STDOUT:   %.loc34_5.2 = load double, ptr %x, align 8, !dbg !45
// CHECK:STDOUT:   %D.call = call double @_CD.Main.de5b400758c39a65(double %.loc34_5.2), !dbg !44
// CHECK:STDOUT:   store double %D.call, ptr %.loc34_7.3.temp, align 8, !dbg !44
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.64ccbb8e5d9a0b8e(i32 %x) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.de5b400758c39a65(double %x) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.582d60b54baf6b63, { 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_different_specific.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 42, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 43, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 44, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 46, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 46, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 47, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 47, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 42, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.329e9a7481f16207", scope: null, file: !3, line: 37, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 38, column: 3, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 39, column: 3, scope: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 37, column: 1, scope: !14)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.7220df690d061955", scope: null, file: !3, line: 37, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !17)
// CHECK:STDOUT: !26 = !DILocation(line: 38, column: 3, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 39, column: 3, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 37, column: 1, scope: !23)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.582d60b54baf6b63", scope: null, file: !3, line: 25, type: !30, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !32)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{!17, !17}
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !29, type: !17)
// CHECK:STDOUT: !34 = !DILocation(line: 26, column: 3, scope: !29)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.329e9a7481f16207", scope: null, file: !3, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !36)
// CHECK:STDOUT: !36 = !{!37}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !35, type: !17)
// CHECK:STDOUT: !38 = !DILocation(line: 34, column: 3, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 34, column: 5, scope: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 33, column: 1, scope: !35)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.7220df690d061955", scope: null, file: !3, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !17)
// CHECK:STDOUT: !44 = !DILocation(line: 34, column: 3, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 34, column: 5, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 33, column: 1, scope: !41)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 29, type: !48, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !51)
// CHECK:STDOUT: !48 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !49 = !{!50, !50}
// CHECK:STDOUT: !50 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !51 = !{!52}
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !47, type: !50)
// CHECK:STDOUT: !53 = !DILocation(line: 30, column: 3, scope: !47)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.de5b400758c39a65", scope: null, file: !3, line: 29, type: !30, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !55)
// CHECK:STDOUT: !55 = !{!56}
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !54, type: !17)
// CHECK:STDOUT: !57 = !DILocation(line: 30, column: 3, scope: !54)
