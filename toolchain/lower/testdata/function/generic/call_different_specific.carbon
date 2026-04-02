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

fn M(ptr_i32: i32*, ptr_f64: f64*) {
  F(ptr_i32);
  F(ptr_f64);
}

// CHECK:STDOUT: ; ModuleID = 'call_different_specific.carbon'
// CHECK:STDOUT: source_filename = "call_different_specific.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main(ptr %ptr_i32, ptr %ptr_f64) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.bfb441135f07cbe1(ptr %ptr_i32), !dbg !11
// CHECK:STDOUT:   call void @_CF.Main.29cf1dfeccef7249(ptr %ptr_f64), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.bfb441135f07cbe1(ptr %x) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type @_CA.Main.582d60b54baf6b63(%type zeroinitializer), !dbg !19
// CHECK:STDOUT:   call void @_CB.Main.bfb441135f07cbe1(ptr %x), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.29cf1dfeccef7249(ptr %x) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type @_CA.Main.582d60b54baf6b63(%type zeroinitializer), !dbg !25
// CHECK:STDOUT:   call void @_CB.Main.29cf1dfeccef7249(ptr %x), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type @_CA.Main.582d60b54baf6b63(%type %x) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.bfb441135f07cbe1(ptr %x) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.2.temp = alloca i32, align 4, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.2.temp), !dbg !37
// CHECK:STDOUT:   %.loc34_5.2 = load i32, ptr %x, align 4, !dbg !38
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.64ccbb8e5d9a0b8e(i32 %.loc34_5.2), !dbg !37
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc34_7.2.temp, align 4, !dbg !37
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.29cf1dfeccef7249(ptr %x) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.2.temp = alloca double, align 8, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.2.temp), !dbg !43
// CHECK:STDOUT:   %.loc34_5.2 = load double, ptr %x, align 8, !dbg !44
// CHECK:STDOUT:   %D.call = call double @_CD.Main.de5b400758c39a65(double %.loc34_5.2), !dbg !43
// CHECK:STDOUT:   store double %D.call, ptr %.loc34_7.2.temp, align 8, !dbg !43
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.64ccbb8e5d9a0b8e(i32 %x) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.de5b400758c39a65(double %x) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CA.Main.582d60b54baf6b63, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 42, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 43, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 44, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 42, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.bfb441135f07cbe1", scope: null, file: !3, line: 37, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !7}
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !14, type: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 38, column: 3, scope: !14)
// CHECK:STDOUT: !20 = !DILocation(line: 39, column: 3, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 37, column: 1, scope: !14)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.29cf1dfeccef7249", scope: null, file: !3, line: 37, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !7)
// CHECK:STDOUT: !25 = !DILocation(line: 38, column: 3, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 39, column: 3, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 37, column: 1, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.582d60b54baf6b63", scope: null, file: !3, line: 25, type: !29, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{!7, !7}
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !33 = !DILocation(line: 26, column: 3, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.bfb441135f07cbe1", scope: null, file: !3, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 34, column: 3, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 34, column: 5, scope: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 33, column: 1, scope: !34)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.29cf1dfeccef7249", scope: null, file: !3, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !7)
// CHECK:STDOUT: !43 = !DILocation(line: 34, column: 3, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 34, column: 5, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 33, column: 1, scope: !40)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.64ccbb8e5d9a0b8e", scope: null, file: !3, line: 29, type: !47, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!49, !49}
// CHECK:STDOUT: !49 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !46, type: !49)
// CHECK:STDOUT: !52 = !DILocation(line: 30, column: 3, scope: !46)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.de5b400758c39a65", scope: null, file: !3, line: 29, type: !29, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !7)
// CHECK:STDOUT: !56 = !DILocation(line: 30, column: 3, scope: !53)
