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

fn M(ptr_i32: i32*, ptr_f64: f64*) {
  F(ptr_i32);
  F(ptr_f64);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_different_specific.carbon'
// CHECK:STDOUT: source_filename = "call_different_specific.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type.0 = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main(ptr %ptr_i32, ptr %ptr_f64) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.89a3e87df6fdafe1(ptr %ptr_i32), !dbg !20
// CHECK:STDOUT:   call void @_CF.Main.38241ddc2cd9d74d(ptr %ptr_f64), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.89a3e87df6fdafe1(ptr %x) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.0 @_CA.Main.c3359c2c217ad593(%type.0 zeroinitializer), !dbg !36
// CHECK:STDOUT:   call void @_CB.Main.89a3e87df6fdafe1(ptr %x), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.38241ddc2cd9d74d(ptr %x) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.0 @_CA.Main.c3359c2c217ad593(%type.0 zeroinitializer), !dbg !46
// CHECK:STDOUT:   call void @_CB.Main.38241ddc2cd9d74d(ptr %x), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.0 @_CA.Main.c3359c2c217ad593(%type.0 %x) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.0 %x, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.89a3e87df6fdafe1(ptr %x) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.3.temp = alloca i32, align 4, !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.3.temp), !dbg !64
// CHECK:STDOUT:   %.loc34_5.2 = load i32, ptr %x, align 4, !dbg !63
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.4d6ceefeb7f772f4(i32 %.loc34_5.2), !dbg !64
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc34_7.3.temp, align 4, !dbg !64
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %.loc34_7.3.temp), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.38241ddc2cd9d74d(ptr %x) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.3.temp = alloca double, align 8, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.3.temp), !dbg !74
// CHECK:STDOUT:   %.loc34_5.2 = load double, ptr %x, align 8, !dbg !73
// CHECK:STDOUT:   %D.call = call double @_CD.Main.c5e4ba542a56b71f(double %.loc34_5.2), !dbg !74
// CHECK:STDOUT:   store double %D.call, ptr %.loc34_7.3.temp, align 8, !dbg !74
// CHECK:STDOUT:   call void @"_COp.dc14d4dd21801791:core.Destroy.Core"(ptr %.loc34_7.3.temp), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.4d6ceefeb7f772f4(i32 %x) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.c5e4ba542a56b71f(double %x) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CA.Main.c3359c2c217ad593, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_different_specific.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 42, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !24)
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !14, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !14, type: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 43, column: 3, scope: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 44, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 42, column: 1, scope: !14)
// CHECK:STDOUT: !24 = !{!15, !16}
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 43, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !28)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !6)
// CHECK:STDOUT: !27 = !DILocation(line: 43, column: 3, scope: !25)
// CHECK:STDOUT: !28 = !{!26}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 44, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !32)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !7)
// CHECK:STDOUT: !31 = !DILocation(line: 44, column: 3, scope: !29)
// CHECK:STDOUT: !32 = !{!30}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.89a3e87df6fdafe1", scope: null, file: !4, line: 37, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !36 = !DILocation(line: 38, column: 3, scope: !33)
// CHECK:STDOUT: !37 = !{!7, !7}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !40 = !DILocation(line: 39, column: 3, scope: !33)
// CHECK:STDOUT: !41 = !DILocation(line: 37, column: 1, scope: !33)
// CHECK:STDOUT: !42 = !{!34}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.38241ddc2cd9d74d", scope: null, file: !4, line: 37, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !46 = !DILocation(line: 38, column: 3, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 39, column: 3, scope: !43)
// CHECK:STDOUT: !49 = !DILocation(line: 37, column: 1, scope: !43)
// CHECK:STDOUT: !50 = !{!44}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c3359c2c217ad593", scope: null, file: !4, line: 25, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !7)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 3, scope: !51)
// CHECK:STDOUT: !58 = !{!52}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.89a3e87df6fdafe1", scope: null, file: !4, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !7)
// CHECK:STDOUT: !63 = !DILocation(line: 34, column: 5, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 34, column: 3, scope: !59)
// CHECK:STDOUT: !65 = !{!6, !6}
// CHECK:STDOUT: !66 = !DISubroutineType(types: !65)
// CHECK:STDOUT: !67 = !DILocation(line: 33, column: 1, scope: !59)
// CHECK:STDOUT: !68 = !{!60}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.38241ddc2cd9d74d", scope: null, file: !4, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !7)
// CHECK:STDOUT: !73 = !DILocation(line: 34, column: 5, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 34, column: 3, scope: !69)
// CHECK:STDOUT: !75 = !DILocation(line: 33, column: 1, scope: !69)
// CHECK:STDOUT: !76 = !{!70}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.4d6ceefeb7f772f4", scope: null, file: !4, line: 29, type: !66, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !6)
// CHECK:STDOUT: !83 = !DILocation(line: 30, column: 3, scope: !77)
// CHECK:STDOUT: !84 = !{!78}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.c5e4ba542a56b71f", scope: null, file: !4, line: 29, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !92)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !7)
// CHECK:STDOUT: !91 = !DILocation(line: 30, column: 3, scope: !85)
// CHECK:STDOUT: !92 = !{!86}
// CHECK:STDOUT:
