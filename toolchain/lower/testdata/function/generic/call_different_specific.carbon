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
// CHECK:STDOUT:   call void @_CF.Main.58016a73bff04416(ptr %ptr_i32), !dbg !11
// CHECK:STDOUT:   call void @_CF.Main.fa1f0912a5bbe922(ptr %ptr_f64), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.58016a73bff04416(ptr %x) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type @_CA.Main.ee509b43cf4eb3c5(%type zeroinitializer), !dbg !30
// CHECK:STDOUT:   call void @_CB.Main.58016a73bff04416(ptr %x), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.fa1f0912a5bbe922(ptr %x) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type @_CA.Main.ee509b43cf4eb3c5(%type zeroinitializer), !dbg !36
// CHECK:STDOUT:   call void @_CB.Main.fa1f0912a5bbe922(ptr %x), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type @_CA.Main.ee509b43cf4eb3c5(%type %x) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.58016a73bff04416(ptr %x) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.3.temp = alloca i32, align 4, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.3.temp), !dbg !48
// CHECK:STDOUT:   %.loc34_5.2 = load i32, ptr %x, align 4, !dbg !49
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.5bd8d5767580997a(i32 %.loc34_5.2), !dbg !48
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc34_7.3.temp, align 4, !dbg !48
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %.loc34_7.3.temp), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.fa1f0912a5bbe922(ptr %x) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.3.temp = alloca double, align 8, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.3.temp), !dbg !54
// CHECK:STDOUT:   %.loc34_5.2 = load double, ptr %x, align 8, !dbg !55
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c27b765f2159c9a0(double %.loc34_5.2), !dbg !54
// CHECK:STDOUT:   store double %D.call, ptr %.loc34_7.3.temp, align 8, !dbg !54
// CHECK:STDOUT:   call void @"_COp.30d8007fdcc57f83:core.Destroy.Core"(ptr %.loc34_7.3.temp), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.5bd8d5767580997a(i32 %x) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c27b765f2159c9a0(double %x) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CA.Main.ee509b43cf4eb3c5, { 1, 0 }
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
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 43, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 43, column: 3, scope: !14)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Op", linkageName: "_COp.30d8007fdcc57f83:core.Destroy.Core", scope: null, file: !3, line: 44, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{null, !7}
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !21, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 44, column: 3, scope: !21)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.58016a73bff04416", scope: null, file: !3, line: 37, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 38, column: 3, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 39, column: 3, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 37, column: 1, scope: !27)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.fa1f0912a5bbe922", scope: null, file: !3, line: 37, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !36 = !DILocation(line: 38, column: 3, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 39, column: 3, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 37, column: 1, scope: !33)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.ee509b43cf4eb3c5", scope: null, file: !3, line: 25, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!7, !7}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 3, scope: !39)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.58016a73bff04416", scope: null, file: !3, line: 33, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !7)
// CHECK:STDOUT: !48 = !DILocation(line: 34, column: 3, scope: !45)
// CHECK:STDOUT: !49 = !DILocation(line: 34, column: 5, scope: !45)
// CHECK:STDOUT: !50 = !DILocation(line: 33, column: 1, scope: !45)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.fa1f0912a5bbe922", scope: null, file: !3, line: 33, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !52)
// CHECK:STDOUT: !52 = !{!53}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !51, type: !7)
// CHECK:STDOUT: !54 = !DILocation(line: 34, column: 3, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 34, column: 5, scope: !51)
// CHECK:STDOUT: !56 = !DILocation(line: 33, column: 1, scope: !51)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.5bd8d5767580997a", scope: null, file: !3, line: 29, type: !58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !59 = !{!17, !17}
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !57, type: !17)
// CHECK:STDOUT: !62 = !DILocation(line: 30, column: 3, scope: !57)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c27b765f2159c9a0", scope: null, file: !3, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !64)
// CHECK:STDOUT: !64 = !{!65}
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !63, type: !7)
// CHECK:STDOUT: !66 = !DILocation(line: 30, column: 3, scope: !63)
