// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/float.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/float.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

fn Negate(a: f64) -> f64 = "float.negate";
fn TestNegate(a: f64) -> f64 { return Negate(a); }

fn Add(a: f64, b: f64) -> f64 = "float.add";
fn TestAdd(a: f64, b: f64) -> f64 { return Add(a, b); }

fn Sub(a: f64, b: f64) -> f64 = "float.sub";
fn TestSub(a: f64, b: f64) -> f64 { return Sub(a, b); }

fn Mul(a: f64, b: f64) -> f64 = "float.mul";
fn TestMul(a: f64, b: f64) -> f64 { return Mul(a, b); }

fn Div(a: f64, b: f64) -> f64 = "float.div";
fn TestDiv(a: f64, b: f64) -> f64 { return Div(a, b); }

fn Eq(a: f64, b: f64) -> bool = "float.eq";
fn TestEq(a: f64, b: f64) -> bool { return Eq(a, b); }

fn Neq(a: f64, b: f64) -> bool = "float.neq";
fn TestNeq(a: f64, b: f64) -> bool { return Neq(a, b); }

fn Less(a: f64, b: f64) -> bool = "float.less";
fn TestLess(a: f64, b: f64) -> bool { return Less(a, b); }

fn LessEq(a: f64, b: f64) -> bool = "float.less_eq";
fn TestLessEq(a: f64, b: f64) -> bool { return LessEq(a, b); }

fn Greater(a: f64, b: f64) -> bool = "float.greater";
fn TestGreater(a: f64, b: f64) -> bool { return Greater(a, b); }

fn GreaterEq(a: f64, b: f64) -> bool = "float.greater_eq";
fn TestGreaterEq(a: f64, b: f64) -> bool { return GreaterEq(a, b); }

// --- compound_assign.carbon

library "[[@TEST_NAME]]";

fn Add(ref a: f64, b: f64) = "float.add_assign";
fn TestAdd(ref a: f64, b: f64) { Add(ref a, b); }

fn Sub(ref a: f64, b: f64) = "float.sub_assign";
fn TestSub(ref a: f64, b: f64) { Sub(ref a, b); }

fn Mul(ref a: f64, b: f64) = "float.mul_assign";
fn TestMul(ref a: f64, b: f64) { Mul(ref a, b); }

fn Div(ref a: f64, b: f64) = "float.div_assign";
fn TestDiv(ref a: f64, b: f64) { Div(ref a, b); }

// --- types.carbon

library "[[@TEST_NAME]]";

fn F16(a: f16, b: f16) -> f16 = "float.add";
fn TestF16(a: f16, b: f16) -> f16 { return F16(a, b); }

fn F32(a: f32, b: f32) -> f32 = "float.add";
fn TestF32(a: f32, b: f32) -> f32 { return F32(a, b); }

fn F64(a: f64, b: f64) -> f64 = "float.add";
fn TestF64(a: f64, b: f64) -> f64 { return F64(a, b); }

fn F128(a: f128, b: f128) -> f128 = "float.add";
fn TestF128(a: f128, b: f128) -> f128 { return F128(a, b); }

// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestNegate.Main(double %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = fneg double %a, !dbg !10
// CHECK:STDOUT:   ret double %Negate.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestAdd.Main(double %a, double %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = fadd double %a, %b, !dbg !18
// CHECK:STDOUT:   ret double %Add.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestSub.Main(double %a, double %b) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = fsub double %a, %b, !dbg !24
// CHECK:STDOUT:   ret double %Sub.call, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestMul.Main(double %a, double %b) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = fmul double %a, %b, !dbg !30
// CHECK:STDOUT:   ret double %Mul.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestDiv.Main(double %a, double %b) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = fdiv double %a, %b, !dbg !36
// CHECK:STDOUT:   ret double %Div.call, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(double %a, double %b) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = fcmp oeq double %a, %b, !dbg !42
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(double %a, double %b) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = fcmp one double %a, %b, !dbg !48
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(double %a, double %b) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = fcmp olt double %a, %b, !dbg !54
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(double %a, double %b) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = fcmp ole double %a, %b, !dbg !60
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(double %a, double %b) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = fcmp ogt double %a, %b, !dbg !66
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(double %a, double %b) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = fcmp oge double %a, %b, !dbg !72
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 5, column: 39, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 5, column: 32, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !1, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!7, !7, !7}
// CHECK:STDOUT: !15 = !{!16, !17}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 2, scope: !12, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 8, column: 44, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 8, column: 37, scope: !12)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !1, line: 11, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
// CHECK:STDOUT: !21 = !{!22, !23}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !20, type: !7)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 2, scope: !20, type: !7)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 44, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 11, column: 37, scope: !20)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !1, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28, !29}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !7)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !26, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 44, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 37, scope: !26)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34, !35}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !32, type: !7)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 44, scope: !32)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 37, scope: !32)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !1, line: 20, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40, !41}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 2, scope: !38, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 20, column: 44, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 20, column: 37, scope: !38)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !1, line: 23, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46, !47}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !7)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 2, scope: !44, type: !7)
// CHECK:STDOUT: !48 = !DILocation(line: 23, column: 45, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 23, column: 38, scope: !44)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !1, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !51)
// CHECK:STDOUT: !51 = !{!52, !53}
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !50, type: !7)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 2, scope: !50, type: !7)
// CHECK:STDOUT: !54 = !DILocation(line: 26, column: 46, scope: !50)
// CHECK:STDOUT: !55 = !DILocation(line: 26, column: 39, scope: !50)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !1, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !57)
// CHECK:STDOUT: !57 = !{!58, !59}
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !56, type: !7)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 2, scope: !56, type: !7)
// CHECK:STDOUT: !60 = !DILocation(line: 29, column: 48, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 29, column: 41, scope: !56)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !1, line: 32, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64, !65}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !7)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 2, scope: !62, type: !7)
// CHECK:STDOUT: !66 = !DILocation(line: 32, column: 49, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 32, column: 42, scope: !62)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !1, line: 35, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !69)
// CHECK:STDOUT: !69 = !{!70, !71}
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !68, type: !7)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 2, scope: !68, type: !7)
// CHECK:STDOUT: !72 = !DILocation(line: 35, column: 51, scope: !68)
// CHECK:STDOUT: !73 = !DILocation(line: 35, column: 44, scope: !68)
// CHECK:STDOUT: ; ModuleID = 'compound_assign.carbon'
// CHECK:STDOUT: source_filename = "compound_assign.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestAdd.Main(ptr %a, double %b) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = load double, ptr %a, align 8, !dbg !11
// CHECK:STDOUT:   %Add.call1 = fadd double %Add.call, %b, !dbg !11
// CHECK:STDOUT:   store double %Add.call1, ptr %a, align 8, !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSub.Main(ptr %a, double %b) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = load double, ptr %a, align 8, !dbg !17
// CHECK:STDOUT:   %Sub.call1 = fsub double %Sub.call, %b, !dbg !17
// CHECK:STDOUT:   store double %Sub.call1, ptr %a, align 8, !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestMul.Main(ptr %a, double %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = load double, ptr %a, align 8, !dbg !23
// CHECK:STDOUT:   %Mul.call1 = fmul double %Mul.call, %b, !dbg !23
// CHECK:STDOUT:   store double %Mul.call1, ptr %a, align 8, !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestDiv.Main(ptr %a, double %b) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = load double, ptr %a, align 8, !dbg !29
// CHECK:STDOUT:   %Div.call1 = fdiv double %Div.call, %b, !dbg !29
// CHECK:STDOUT:   store double %Div.call1, ptr %a, align 8, !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "compound_assign.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 5, column: 34, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !1, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !13, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 8, column: 34, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 8, column: 1, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !1, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21, !22}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 2, scope: !19, type: !7)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 34, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 1, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !1, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27, !28}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !7)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 2, scope: !25, type: !7)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 34, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 1, scope: !25)
// CHECK:STDOUT: ; ModuleID = 'types.carbon'
// CHECK:STDOUT: source_filename = "types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define half @_CTestF16.Main(half %a, half %b) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F16.call = fadd half %a, %b, !dbg !11
// CHECK:STDOUT:   ret half %F16.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestF32.Main(float %a, float %b) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F32.call = fadd float %a, %b, !dbg !17
// CHECK:STDOUT:   ret float %F32.call, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestF64.Main(double %a, double %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F64.call = fadd double %a, %b, !dbg !23
// CHECK:STDOUT:   ret double %F64.call, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestF128.Main(fp128 %a, fp128 %b) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F128.call = fadd fp128 %a, %b, !dbg !29
// CHECK:STDOUT:   ret fp128 %F128.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "types.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestF16", linkageName: "_CTestF16.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 5, column: 44, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 37, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "TestF32", linkageName: "_CTestF32.Main", scope: null, file: !1, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !13, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 8, column: 44, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 8, column: 37, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "TestF64", linkageName: "_CTestF64.Main", scope: null, file: !1, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21, !22}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 2, scope: !19, type: !7)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 44, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 37, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "TestF128", linkageName: "_CTestF128.Main", scope: null, file: !1, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27, !28}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !7)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 2, scope: !25, type: !7)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 48, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 41, scope: !25)
