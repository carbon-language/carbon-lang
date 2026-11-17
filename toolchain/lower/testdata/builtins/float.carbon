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
// CHECK:STDOUT: define double @_CTestNegate.Main(double %a) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = fneg double %a, !dbg !48
// CHECK:STDOUT:   ret double %Negate.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestAdd.Main(double %a, double %b) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = fadd double %a, %b, !dbg !60
// CHECK:STDOUT:   ret double %Add.call, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestSub.Main(double %a, double %b) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = fsub double %a, %b, !dbg !72
// CHECK:STDOUT:   ret double %Sub.call, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestMul.Main(double %a, double %b) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = fmul double %a, %b, !dbg !84
// CHECK:STDOUT:   ret double %Mul.call, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestDiv.Main(double %a, double %b) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = fdiv double %a, %b, !dbg !96
// CHECK:STDOUT:   ret double %Div.call, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(double %a, double %b) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = fcmp oeq double %a, %b, !dbg !108
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(double %a, double %b) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = fcmp one double %a, %b, !dbg !120
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(double %a, double %b) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = fcmp olt double %a, %b, !dbg !132
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(double %a, double %b) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = fcmp ole double %a, %b, !dbg !144
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(double %a, double %b) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = fcmp ogt double %a, %b, !dbg !156
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(double %a, double %b) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = fcmp oge double %a, %b, !dbg !168
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!37, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{!37, !37, !37}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !35, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !50)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !37)
// CHECK:STDOUT: !48 = !DILocation(line: 5, column: 39, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 5, column: 32, scope: !42)
// CHECK:STDOUT: !50 = !{!43}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !35, line: 8, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !62)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !37)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 2, scope: !51, type: !37)
// CHECK:STDOUT: !60 = !DILocation(line: 8, column: 44, scope: !51)
// CHECK:STDOUT: !61 = !DILocation(line: 8, column: 37, scope: !51)
// CHECK:STDOUT: !62 = !{!52, !53}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !35, line: 11, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !74)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !37)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 2, scope: !63, type: !37)
// CHECK:STDOUT: !72 = !DILocation(line: 11, column: 44, scope: !63)
// CHECK:STDOUT: !73 = !DILocation(line: 11, column: 37, scope: !63)
// CHECK:STDOUT: !74 = !{!64, !65}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !35, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !86)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !37)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 2, scope: !75, type: !37)
// CHECK:STDOUT: !84 = !DILocation(line: 14, column: 44, scope: !75)
// CHECK:STDOUT: !85 = !DILocation(line: 14, column: 37, scope: !75)
// CHECK:STDOUT: !86 = !{!76, !77}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !35, line: 17, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !98)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !37)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 2, scope: !87, type: !37)
// CHECK:STDOUT: !96 = !DILocation(line: 17, column: 44, scope: !87)
// CHECK:STDOUT: !97 = !DILocation(line: 17, column: 37, scope: !87)
// CHECK:STDOUT: !98 = !{!88, !89}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !35, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !110)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !37)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 2, scope: !99, type: !37)
// CHECK:STDOUT: !108 = !DILocation(line: 20, column: 44, scope: !99)
// CHECK:STDOUT: !109 = !DILocation(line: 20, column: 37, scope: !99)
// CHECK:STDOUT: !110 = !{!100, !101}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !35, line: 23, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !122)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !37)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 2, scope: !111, type: !37)
// CHECK:STDOUT: !120 = !DILocation(line: 23, column: 45, scope: !111)
// CHECK:STDOUT: !121 = !DILocation(line: 23, column: 38, scope: !111)
// CHECK:STDOUT: !122 = !{!112, !113}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !35, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !134)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !37)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 2, scope: !123, type: !37)
// CHECK:STDOUT: !132 = !DILocation(line: 26, column: 46, scope: !123)
// CHECK:STDOUT: !133 = !DILocation(line: 26, column: 39, scope: !123)
// CHECK:STDOUT: !134 = !{!124, !125}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !35, line: 29, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !146)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !37)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 2, scope: !135, type: !37)
// CHECK:STDOUT: !144 = !DILocation(line: 29, column: 48, scope: !135)
// CHECK:STDOUT: !145 = !DILocation(line: 29, column: 41, scope: !135)
// CHECK:STDOUT: !146 = !{!136, !137}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !35, line: 32, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !158)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !37)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 2, scope: !147, type: !37)
// CHECK:STDOUT: !156 = !DILocation(line: 32, column: 49, scope: !147)
// CHECK:STDOUT: !157 = !DILocation(line: 32, column: 42, scope: !147)
// CHECK:STDOUT: !158 = !{!148, !149}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !35, line: 35, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !170)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !37)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 2, scope: !159, type: !37)
// CHECK:STDOUT: !168 = !DILocation(line: 35, column: 51, scope: !159)
// CHECK:STDOUT: !169 = !DILocation(line: 35, column: 44, scope: !159)
// CHECK:STDOUT: !170 = !{!160, !161}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compound_assign.carbon'
// CHECK:STDOUT: source_filename = "compound_assign.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestAdd.Main(ptr %a, double %b) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = load double, ptr %a, align 8, !dbg !183
// CHECK:STDOUT:   %Add.call1 = fadd double %Add.call, %b, !dbg !183
// CHECK:STDOUT:   store double %Add.call1, ptr %a, align 8, !dbg !183
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSub.Main(ptr %a, double %b) #0 !dbg !186 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = load double, ptr %a, align 8, !dbg !194
// CHECK:STDOUT:   %Sub.call1 = fsub double %Sub.call, %b, !dbg !194
// CHECK:STDOUT:   store double %Sub.call1, ptr %a, align 8, !dbg !194
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestMul.Main(ptr %a, double %b) #0 !dbg !197 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = load double, ptr %a, align 8, !dbg !205
// CHECK:STDOUT:   %Mul.call1 = fmul double %Mul.call, %b, !dbg !205
// CHECK:STDOUT:   store double %Mul.call1, ptr %a, align 8, !dbg !205
// CHECK:STDOUT:   ret void, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestDiv.Main(ptr %a, double %b) #0 !dbg !208 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = load double, ptr %a, align 8, !dbg !216
// CHECK:STDOUT:   %Div.call1 = fdiv double %Div.call, %b, !dbg !216
// CHECK:STDOUT:   store double %Div.call1, ptr %a, align 8, !dbg !216
// CHECK:STDOUT:   ret void, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!172}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !171 = !DIFile(filename: "compound_assign.carbon", directory: "")
// CHECK:STDOUT: !172 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !171, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !173 = !{null, !37, !37}
// CHECK:STDOUT: !174 = !DISubroutineType(types: !173)
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !171, line: 5, type: !174, spFlags: DISPFlagDefinition, unit: !172, retainedNodes: !185)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !37)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 2, scope: !175, type: !37)
// CHECK:STDOUT: !183 = !DILocation(line: 5, column: 34, scope: !175)
// CHECK:STDOUT: !184 = !DILocation(line: 5, column: 1, scope: !175)
// CHECK:STDOUT: !185 = !{!176, !177}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !171, line: 8, type: !174, spFlags: DISPFlagDefinition, unit: !172, retainedNodes: !196)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !37)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 2, scope: !186, type: !37)
// CHECK:STDOUT: !194 = !DILocation(line: 8, column: 34, scope: !186)
// CHECK:STDOUT: !195 = !DILocation(line: 8, column: 1, scope: !186)
// CHECK:STDOUT: !196 = !{!187, !188}
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !171, line: 11, type: !174, spFlags: DISPFlagDefinition, unit: !172, retainedNodes: !207)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !37)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 2, scope: !197, type: !37)
// CHECK:STDOUT: !205 = !DILocation(line: 11, column: 34, scope: !197)
// CHECK:STDOUT: !206 = !DILocation(line: 11, column: 1, scope: !197)
// CHECK:STDOUT: !207 = !{!198, !199}
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !171, line: 14, type: !174, spFlags: DISPFlagDefinition, unit: !172, retainedNodes: !218)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 1, scope: !208, type: !37)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 2, scope: !208, type: !37)
// CHECK:STDOUT: !216 = !DILocation(line: 14, column: 34, scope: !208)
// CHECK:STDOUT: !217 = !DILocation(line: 14, column: 1, scope: !208)
// CHECK:STDOUT: !218 = !{!209, !210}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'types.carbon'
// CHECK:STDOUT: source_filename = "types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define half @_CTestF16.Main(half %a, half %b) #0 !dbg !221 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F16.call = fadd half %a, %b, !dbg !230
// CHECK:STDOUT:   ret half %F16.call, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestF32.Main(float %a, float %b) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F32.call = fadd float %a, %b, !dbg !242
// CHECK:STDOUT:   ret float %F32.call, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestF64.Main(double %a, double %b) #0 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F64.call = fadd double %a, %b, !dbg !254
// CHECK:STDOUT:   ret double %F64.call, !dbg !255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestF128.Main(fp128 %a, fp128 %b) #0 !dbg !257 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F128.call = fadd fp128 %a, %b, !dbg !266
// CHECK:STDOUT:   ret fp128 %F128.call, !dbg !267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!220}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!37, !37, !37}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !219 = !DIFile(filename: "types.carbon", directory: "")
// CHECK:STDOUT: !220 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !219, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "TestF16", linkageName: "_CTestF16.Main", scope: null, file: !219, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !220, retainedNodes: !232)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !37)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 2, scope: !221, type: !37)
// CHECK:STDOUT: !230 = !DILocation(line: 5, column: 44, scope: !221)
// CHECK:STDOUT: !231 = !DILocation(line: 5, column: 37, scope: !221)
// CHECK:STDOUT: !232 = !{!222, !223}
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "TestF32", linkageName: "_CTestF32.Main", scope: null, file: !219, line: 8, type: !41, spFlags: DISPFlagDefinition, unit: !220, retainedNodes: !244)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !37)
// CHECK:STDOUT: !235 = !DILocalVariable(arg: 2, scope: !233, type: !37)
// CHECK:STDOUT: !242 = !DILocation(line: 8, column: 44, scope: !233)
// CHECK:STDOUT: !243 = !DILocation(line: 8, column: 37, scope: !233)
// CHECK:STDOUT: !244 = !{!234, !235}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "TestF64", linkageName: "_CTestF64.Main", scope: null, file: !219, line: 11, type: !41, spFlags: DISPFlagDefinition, unit: !220, retainedNodes: !256)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !37)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 2, scope: !245, type: !37)
// CHECK:STDOUT: !254 = !DILocation(line: 11, column: 44, scope: !245)
// CHECK:STDOUT: !255 = !DILocation(line: 11, column: 37, scope: !245)
// CHECK:STDOUT: !256 = !{!246, !247}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "TestF128", linkageName: "_CTestF128.Main", scope: null, file: !219, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !220, retainedNodes: !268)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !37)
// CHECK:STDOUT: !259 = !DILocalVariable(arg: 2, scope: !257, type: !37)
// CHECK:STDOUT: !266 = !DILocation(line: 14, column: 48, scope: !257)
// CHECK:STDOUT: !267 = !DILocation(line: 14, column: 41, scope: !257)
// CHECK:STDOUT: !268 = !{!258, !259}
// CHECK:STDOUT:
