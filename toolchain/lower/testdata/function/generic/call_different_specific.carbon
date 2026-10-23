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
// CHECK:STDOUT: define void @_CM.Main(ptr %ptr_i32, ptr %ptr_f64) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.091764f9509e9823(ptr %ptr_i32), !dbg !16
// CHECK:STDOUT:   call void @_CF.Main.b5920d783ed6ed2a(ptr %ptr_f64), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.091764f9509e9823(ptr %x) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.0 @_CA.Main.c3359c2c217ad593(%type.0 zeroinitializer), !dbg !26
// CHECK:STDOUT:   call void @_CB.Main.091764f9509e9823(ptr %x), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.b5920d783ed6ed2a(ptr %x) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type.0 @_CA.Main.c3359c2c217ad593(%type.0 zeroinitializer), !dbg !36
// CHECK:STDOUT:   call void @_CB.Main.b5920d783ed6ed2a(ptr %x), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr %type.0 @_CA.Main.c3359c2c217ad593(%type.0 %x) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type.0 %x, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.091764f9509e9823(ptr %x) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.3.temp = alloca i32, align 4, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.3.temp), !dbg !54
// CHECK:STDOUT:   %.loc34_5.2 = load i32, ptr %x, align 4, !dbg !53
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.2fad9574e050607e(i32 %.loc34_5.2), !dbg !54
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc34_7.3.temp, align 4, !dbg !54
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc34_7.3.temp), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.b5920d783ed6ed2a(ptr %x) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc34_7.3.temp = alloca double, align 8, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc34_7.3.temp), !dbg !66
// CHECK:STDOUT:   %.loc34_5.2 = load double, ptr %x, align 8, !dbg !65
// CHECK:STDOUT:   %D.call = call double @_CD.Main.21aefb7dfdebb8e4(double %.loc34_5.2), !dbg !66
// CHECK:STDOUT:   store double %D.call, ptr %.loc34_7.3.temp, align 8, !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %.loc34_7.3.temp), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.2fad9574e050607e(i32 %x) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !78 {
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.21aefb7dfdebb8e4(double %x) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !93 {
// CHECK:STDOUT:   ret void, !dbg !98
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
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 42, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !11 = !DILocalVariable(arg: 1, scope: !10, type: !7)
// CHECK:STDOUT: !12 = !DILocalVariable(arg: 2, scope: !10, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 43, column: 3, scope: !10)
// CHECK:STDOUT: !17 = !{null, !7}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 44, column: 3, scope: !10)
// CHECK:STDOUT: !21 = !DILocation(line: 42, column: 1, scope: !10)
// CHECK:STDOUT: !22 = !{!11, !12}
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.091764f9509e9823", scope: null, file: !4, line: 37, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !32)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 38, column: 3, scope: !23)
// CHECK:STDOUT: !27 = !{!7, !7}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !30 = !DILocation(line: 39, column: 3, scope: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 37, column: 1, scope: !23)
// CHECK:STDOUT: !32 = !{!24}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b5920d783ed6ed2a", scope: null, file: !4, line: 37, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !40)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !36 = !DILocation(line: 38, column: 3, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 39, column: 3, scope: !33)
// CHECK:STDOUT: !39 = !DILocation(line: 37, column: 1, scope: !33)
// CHECK:STDOUT: !40 = !{!34}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.c3359c2c217ad593", scope: null, file: !4, line: 25, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !48)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 26, column: 3, scope: !41)
// CHECK:STDOUT: !48 = !{!42}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.091764f9509e9823", scope: null, file: !4, line: 33, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !60)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !53 = !DILocation(line: 34, column: 5, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 34, column: 3, scope: !49)
// CHECK:STDOUT: !55 = !{!6, !6}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !57 = !{null, !6}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 33, column: 1, scope: !49)
// CHECK:STDOUT: !60 = !{!50}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.b5920d783ed6ed2a", scope: null, file: !4, line: 33, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !7)
// CHECK:STDOUT: !65 = !DILocation(line: 34, column: 5, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 34, column: 3, scope: !61)
// CHECK:STDOUT: !67 = !DILocation(line: 33, column: 1, scope: !61)
// CHECK:STDOUT: !68 = !{!62}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.2fad9574e050607e", scope: null, file: !4, line: 29, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !6)
// CHECK:STDOUT: !75 = !DILocation(line: 30, column: 3, scope: !69)
// CHECK:STDOUT: !76 = !{!70}
// CHECK:STDOUT: !77 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !77, line: 35, type: !58, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !6)
// CHECK:STDOUT: !83 = !DILocation(line: 35, column: 1, scope: !78)
// CHECK:STDOUT: !84 = !{!79}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.21aefb7dfdebb8e4", scope: null, file: !4, line: 29, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !92)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !7)
// CHECK:STDOUT: !91 = !DILocation(line: 30, column: 3, scope: !85)
// CHECK:STDOUT: !92 = !{!86}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !77, line: 35, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !99)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !7)
// CHECK:STDOUT: !98 = !DILocation(line: 35, column: 1, scope: !93)
// CHECK:STDOUT: !99 = !{!94}
// CHECK:STDOUT:
