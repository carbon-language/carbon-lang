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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestNegate.Main(double %a) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = fneg double %a, !dbg !47
// CHECK:STDOUT:   ret double %Negate.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestAdd.Main(double %a, double %b) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = fadd double %a, %b, !dbg !59
// CHECK:STDOUT:   ret double %Add.call, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestSub.Main(double %a, double %b) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = fsub double %a, %b, !dbg !71
// CHECK:STDOUT:   ret double %Sub.call, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestMul.Main(double %a, double %b) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = fmul double %a, %b, !dbg !83
// CHECK:STDOUT:   ret double %Mul.call, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestDiv.Main(double %a, double %b) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = fdiv double %a, %b, !dbg !95
// CHECK:STDOUT:   ret double %Div.call, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(double %a, double %b) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = fcmp oeq double %a, %b, !dbg !107
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(double %a, double %b) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = fcmp one double %a, %b, !dbg !119
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(double %a, double %b) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = fcmp olt double %a, %b, !dbg !131
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(double %a, double %b) #0 !dbg !134 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = fcmp ole double %a, %b, !dbg !143
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(double %a, double %b) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = fcmp ogt double %a, %b, !dbg !155
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(double %a, double %b) #0 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = fcmp oge double %a, %b, !dbg !167
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{!36, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{!36, !36, !36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !34, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !49)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !36)
// CHECK:STDOUT: !47 = !DILocation(line: 5, column: 39, scope: !41)
// CHECK:STDOUT: !48 = !DILocation(line: 5, column: 32, scope: !41)
// CHECK:STDOUT: !49 = !{!42}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !34, line: 8, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !61)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !36)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !50, type: !36)
// CHECK:STDOUT: !59 = !DILocation(line: 8, column: 44, scope: !50)
// CHECK:STDOUT: !60 = !DILocation(line: 8, column: 37, scope: !50)
// CHECK:STDOUT: !61 = !{!51, !52}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !34, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !73)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !36)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !62, type: !36)
// CHECK:STDOUT: !71 = !DILocation(line: 11, column: 44, scope: !62)
// CHECK:STDOUT: !72 = !DILocation(line: 11, column: 37, scope: !62)
// CHECK:STDOUT: !73 = !{!63, !64}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !34, line: 14, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !85)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !36)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 2, scope: !74, type: !36)
// CHECK:STDOUT: !83 = !DILocation(line: 14, column: 44, scope: !74)
// CHECK:STDOUT: !84 = !DILocation(line: 14, column: 37, scope: !74)
// CHECK:STDOUT: !85 = !{!75, !76}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !34, line: 17, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !97)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !36)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 2, scope: !86, type: !36)
// CHECK:STDOUT: !95 = !DILocation(line: 17, column: 44, scope: !86)
// CHECK:STDOUT: !96 = !DILocation(line: 17, column: 37, scope: !86)
// CHECK:STDOUT: !97 = !{!87, !88}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !34, line: 20, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !109)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !36)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !98, type: !36)
// CHECK:STDOUT: !107 = !DILocation(line: 20, column: 44, scope: !98)
// CHECK:STDOUT: !108 = !DILocation(line: 20, column: 37, scope: !98)
// CHECK:STDOUT: !109 = !{!99, !100}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !34, line: 23, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !121)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !36)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 2, scope: !110, type: !36)
// CHECK:STDOUT: !119 = !DILocation(line: 23, column: 45, scope: !110)
// CHECK:STDOUT: !120 = !DILocation(line: 23, column: 38, scope: !110)
// CHECK:STDOUT: !121 = !{!111, !112}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !34, line: 26, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !133)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !36)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !122, type: !36)
// CHECK:STDOUT: !131 = !DILocation(line: 26, column: 46, scope: !122)
// CHECK:STDOUT: !132 = !DILocation(line: 26, column: 39, scope: !122)
// CHECK:STDOUT: !133 = !{!123, !124}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !34, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !145)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !36)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 2, scope: !134, type: !36)
// CHECK:STDOUT: !143 = !DILocation(line: 29, column: 48, scope: !134)
// CHECK:STDOUT: !144 = !DILocation(line: 29, column: 41, scope: !134)
// CHECK:STDOUT: !145 = !{!135, !136}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !34, line: 32, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !157)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !36)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 2, scope: !146, type: !36)
// CHECK:STDOUT: !155 = !DILocation(line: 32, column: 49, scope: !146)
// CHECK:STDOUT: !156 = !DILocation(line: 32, column: 42, scope: !146)
// CHECK:STDOUT: !157 = !{!147, !148}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !34, line: 35, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !169)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !36)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 2, scope: !158, type: !36)
// CHECK:STDOUT: !167 = !DILocation(line: 35, column: 51, scope: !158)
// CHECK:STDOUT: !168 = !DILocation(line: 35, column: 44, scope: !158)
// CHECK:STDOUT: !169 = !{!159, !160}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compound_assign.carbon'
// CHECK:STDOUT: source_filename = "compound_assign.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestAdd.Main(ptr %a, double %b) #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = load double, ptr %a, align 8, !dbg !182
// CHECK:STDOUT:   %Add.call1 = fadd double %Add.call, %b, !dbg !182
// CHECK:STDOUT:   store double %Add.call1, ptr %a, align 8, !dbg !182
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSub.Main(ptr %a, double %b) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = load double, ptr %a, align 8, !dbg !193
// CHECK:STDOUT:   %Sub.call1 = fsub double %Sub.call, %b, !dbg !193
// CHECK:STDOUT:   store double %Sub.call1, ptr %a, align 8, !dbg !193
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestMul.Main(ptr %a, double %b) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = load double, ptr %a, align 8, !dbg !204
// CHECK:STDOUT:   %Mul.call1 = fmul double %Mul.call, %b, !dbg !204
// CHECK:STDOUT:   store double %Mul.call1, ptr %a, align 8, !dbg !204
// CHECK:STDOUT:   ret void, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestDiv.Main(ptr %a, double %b) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = load double, ptr %a, align 8, !dbg !215
// CHECK:STDOUT:   %Div.call1 = fdiv double %Div.call, %b, !dbg !215
// CHECK:STDOUT:   store double %Div.call1, ptr %a, align 8, !dbg !215
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!171}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !170 = !DIFile(filename: "compound_assign.carbon", directory: "")
// CHECK:STDOUT: !171 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !170, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !172 = !{null, !36, !36}
// CHECK:STDOUT: !173 = !DISubroutineType(types: !172)
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !170, line: 5, type: !173, spFlags: DISPFlagDefinition, unit: !171, retainedNodes: !184)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !36)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 2, scope: !174, type: !36)
// CHECK:STDOUT: !182 = !DILocation(line: 5, column: 34, scope: !174)
// CHECK:STDOUT: !183 = !DILocation(line: 5, column: 1, scope: !174)
// CHECK:STDOUT: !184 = !{!175, !176}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !170, line: 8, type: !173, spFlags: DISPFlagDefinition, unit: !171, retainedNodes: !195)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !36)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 2, scope: !185, type: !36)
// CHECK:STDOUT: !193 = !DILocation(line: 8, column: 34, scope: !185)
// CHECK:STDOUT: !194 = !DILocation(line: 8, column: 1, scope: !185)
// CHECK:STDOUT: !195 = !{!186, !187}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !170, line: 11, type: !173, spFlags: DISPFlagDefinition, unit: !171, retainedNodes: !206)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !36)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 2, scope: !196, type: !36)
// CHECK:STDOUT: !204 = !DILocation(line: 11, column: 34, scope: !196)
// CHECK:STDOUT: !205 = !DILocation(line: 11, column: 1, scope: !196)
// CHECK:STDOUT: !206 = !{!197, !198}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !170, line: 14, type: !173, spFlags: DISPFlagDefinition, unit: !171, retainedNodes: !217)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !36)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 2, scope: !207, type: !36)
// CHECK:STDOUT: !215 = !DILocation(line: 14, column: 34, scope: !207)
// CHECK:STDOUT: !216 = !DILocation(line: 14, column: 1, scope: !207)
// CHECK:STDOUT: !217 = !{!208, !209}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'types.carbon'
// CHECK:STDOUT: source_filename = "types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define half @_CTestF16.Main(half %a, half %b) #0 !dbg !220 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F16.call = fadd half %a, %b, !dbg !229
// CHECK:STDOUT:   ret half %F16.call, !dbg !230
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestF32.Main(float %a, float %b) #0 !dbg !232 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F32.call = fadd float %a, %b, !dbg !241
// CHECK:STDOUT:   ret float %F32.call, !dbg !242
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestF64.Main(double %a, double %b) #0 !dbg !244 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F64.call = fadd double %a, %b, !dbg !253
// CHECK:STDOUT:   ret double %F64.call, !dbg !254
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestF128.Main(fp128 %a, fp128 %b) #0 !dbg !256 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F128.call = fadd fp128 %a, %b, !dbg !265
// CHECK:STDOUT:   ret fp128 %F128.call, !dbg !266
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!219}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !39 = !{!36, !36, !36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !218 = !DIFile(filename: "types.carbon", directory: "")
// CHECK:STDOUT: !219 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !218, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !220 = distinct !DISubprogram(name: "TestF16", linkageName: "_CTestF16.Main", scope: null, file: !218, line: 5, type: !40, spFlags: DISPFlagDefinition, unit: !219, retainedNodes: !231)
// CHECK:STDOUT: !221 = !DILocalVariable(arg: 1, scope: !220, type: !36)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 2, scope: !220, type: !36)
// CHECK:STDOUT: !229 = !DILocation(line: 5, column: 44, scope: !220)
// CHECK:STDOUT: !230 = !DILocation(line: 5, column: 37, scope: !220)
// CHECK:STDOUT: !231 = !{!221, !222}
// CHECK:STDOUT: !232 = distinct !DISubprogram(name: "TestF32", linkageName: "_CTestF32.Main", scope: null, file: !218, line: 8, type: !40, spFlags: DISPFlagDefinition, unit: !219, retainedNodes: !243)
// CHECK:STDOUT: !233 = !DILocalVariable(arg: 1, scope: !232, type: !36)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 2, scope: !232, type: !36)
// CHECK:STDOUT: !241 = !DILocation(line: 8, column: 44, scope: !232)
// CHECK:STDOUT: !242 = !DILocation(line: 8, column: 37, scope: !232)
// CHECK:STDOUT: !243 = !{!233, !234}
// CHECK:STDOUT: !244 = distinct !DISubprogram(name: "TestF64", linkageName: "_CTestF64.Main", scope: null, file: !218, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !219, retainedNodes: !255)
// CHECK:STDOUT: !245 = !DILocalVariable(arg: 1, scope: !244, type: !36)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 2, scope: !244, type: !36)
// CHECK:STDOUT: !253 = !DILocation(line: 11, column: 44, scope: !244)
// CHECK:STDOUT: !254 = !DILocation(line: 11, column: 37, scope: !244)
// CHECK:STDOUT: !255 = !{!245, !246}
// CHECK:STDOUT: !256 = distinct !DISubprogram(name: "TestF128", linkageName: "_CTestF128.Main", scope: null, file: !218, line: 14, type: !40, spFlags: DISPFlagDefinition, unit: !219, retainedNodes: !267)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 1, scope: !256, type: !36)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 2, scope: !256, type: !36)
// CHECK:STDOUT: !265 = !DILocation(line: 14, column: 48, scope: !256)
// CHECK:STDOUT: !266 = !DILocation(line: 14, column: 41, scope: !256)
// CHECK:STDOUT: !267 = !{!257, !258}
// CHECK:STDOUT:
