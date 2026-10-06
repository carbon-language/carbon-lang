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
// CHECK:STDOUT: define double @_CTestNegate.Main(double %a) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = fneg double %a, !dbg !45
// CHECK:STDOUT:   ret double %Negate.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestAdd.Main(double %a, double %b) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = fadd double %a, %b, !dbg !57
// CHECK:STDOUT:   ret double %Add.call, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestSub.Main(double %a, double %b) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = fsub double %a, %b, !dbg !69
// CHECK:STDOUT:   ret double %Sub.call, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestMul.Main(double %a, double %b) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = fmul double %a, %b, !dbg !81
// CHECK:STDOUT:   ret double %Mul.call, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestDiv.Main(double %a, double %b) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = fdiv double %a, %b, !dbg !93
// CHECK:STDOUT:   ret double %Div.call, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(double %a, double %b) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = fcmp oeq double %a, %b, !dbg !105
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(double %a, double %b) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = fcmp one double %a, %b, !dbg !117
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(double %a, double %b) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = fcmp olt double %a, %b, !dbg !129
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(double %a, double %b) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = fcmp ole double %a, %b, !dbg !141
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(double %a, double %b) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = fcmp ogt double %a, %b, !dbg !153
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(double %a, double %b) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = fcmp oge double %a, %b, !dbg !165
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{!34, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{!34, !34, !34}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !32, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !47)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !34)
// CHECK:STDOUT: !45 = !DILocation(line: 5, column: 39, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 5, column: 32, scope: !39)
// CHECK:STDOUT: !47 = !{!40}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !32, line: 8, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !59)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !34)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !48, type: !34)
// CHECK:STDOUT: !57 = !DILocation(line: 8, column: 44, scope: !48)
// CHECK:STDOUT: !58 = !DILocation(line: 8, column: 37, scope: !48)
// CHECK:STDOUT: !59 = !{!49, !50}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !32, line: 11, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !71)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !34)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 2, scope: !60, type: !34)
// CHECK:STDOUT: !69 = !DILocation(line: 11, column: 44, scope: !60)
// CHECK:STDOUT: !70 = !DILocation(line: 11, column: 37, scope: !60)
// CHECK:STDOUT: !71 = !{!61, !62}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !32, line: 14, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !83)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !34)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 2, scope: !72, type: !34)
// CHECK:STDOUT: !81 = !DILocation(line: 14, column: 44, scope: !72)
// CHECK:STDOUT: !82 = !DILocation(line: 14, column: 37, scope: !72)
// CHECK:STDOUT: !83 = !{!73, !74}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !32, line: 17, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !95)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !34)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 2, scope: !84, type: !34)
// CHECK:STDOUT: !93 = !DILocation(line: 17, column: 44, scope: !84)
// CHECK:STDOUT: !94 = !DILocation(line: 17, column: 37, scope: !84)
// CHECK:STDOUT: !95 = !{!85, !86}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !32, line: 20, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !107)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !34)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 2, scope: !96, type: !34)
// CHECK:STDOUT: !105 = !DILocation(line: 20, column: 44, scope: !96)
// CHECK:STDOUT: !106 = !DILocation(line: 20, column: 37, scope: !96)
// CHECK:STDOUT: !107 = !{!97, !98}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !32, line: 23, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !119)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !34)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !108, type: !34)
// CHECK:STDOUT: !117 = !DILocation(line: 23, column: 45, scope: !108)
// CHECK:STDOUT: !118 = !DILocation(line: 23, column: 38, scope: !108)
// CHECK:STDOUT: !119 = !{!109, !110}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !32, line: 26, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !131)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !34)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !120, type: !34)
// CHECK:STDOUT: !129 = !DILocation(line: 26, column: 46, scope: !120)
// CHECK:STDOUT: !130 = !DILocation(line: 26, column: 39, scope: !120)
// CHECK:STDOUT: !131 = !{!121, !122}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !32, line: 29, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !143)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !34)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 2, scope: !132, type: !34)
// CHECK:STDOUT: !141 = !DILocation(line: 29, column: 48, scope: !132)
// CHECK:STDOUT: !142 = !DILocation(line: 29, column: 41, scope: !132)
// CHECK:STDOUT: !143 = !{!133, !134}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !32, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !155)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !34)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 2, scope: !144, type: !34)
// CHECK:STDOUT: !153 = !DILocation(line: 32, column: 49, scope: !144)
// CHECK:STDOUT: !154 = !DILocation(line: 32, column: 42, scope: !144)
// CHECK:STDOUT: !155 = !{!145, !146}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !32, line: 35, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !167)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !34)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 2, scope: !156, type: !34)
// CHECK:STDOUT: !165 = !DILocation(line: 35, column: 51, scope: !156)
// CHECK:STDOUT: !166 = !DILocation(line: 35, column: 44, scope: !156)
// CHECK:STDOUT: !167 = !{!157, !158}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compound_assign.carbon'
// CHECK:STDOUT: source_filename = "compound_assign.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestAdd.Main(ptr %a, double %b) #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = load double, ptr %a, align 8, !dbg !180
// CHECK:STDOUT:   %Add.call1 = fadd double %Add.call, %b, !dbg !180
// CHECK:STDOUT:   store double %Add.call1, ptr %a, align 8, !dbg !180
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSub.Main(ptr %a, double %b) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = load double, ptr %a, align 8, !dbg !191
// CHECK:STDOUT:   %Sub.call1 = fsub double %Sub.call, %b, !dbg !191
// CHECK:STDOUT:   store double %Sub.call1, ptr %a, align 8, !dbg !191
// CHECK:STDOUT:   ret void, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestMul.Main(ptr %a, double %b) #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = load double, ptr %a, align 8, !dbg !202
// CHECK:STDOUT:   %Mul.call1 = fmul double %Mul.call, %b, !dbg !202
// CHECK:STDOUT:   store double %Mul.call1, ptr %a, align 8, !dbg !202
// CHECK:STDOUT:   ret void, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestDiv.Main(ptr %a, double %b) #0 !dbg !205 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = load double, ptr %a, align 8, !dbg !213
// CHECK:STDOUT:   %Div.call1 = fdiv double %Div.call, %b, !dbg !213
// CHECK:STDOUT:   store double %Div.call1, ptr %a, align 8, !dbg !213
// CHECK:STDOUT:   ret void, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!169}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !168 = !DIFile(filename: "compound_assign.carbon", directory: "")
// CHECK:STDOUT: !169 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !168, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !170 = !{null, !34, !34}
// CHECK:STDOUT: !171 = !DISubroutineType(types: !170)
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !168, line: 5, type: !171, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !182)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !34)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 2, scope: !172, type: !34)
// CHECK:STDOUT: !180 = !DILocation(line: 5, column: 34, scope: !172)
// CHECK:STDOUT: !181 = !DILocation(line: 5, column: 1, scope: !172)
// CHECK:STDOUT: !182 = !{!173, !174}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !168, line: 8, type: !171, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !193)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !34)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 2, scope: !183, type: !34)
// CHECK:STDOUT: !191 = !DILocation(line: 8, column: 34, scope: !183)
// CHECK:STDOUT: !192 = !DILocation(line: 8, column: 1, scope: !183)
// CHECK:STDOUT: !193 = !{!184, !185}
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !168, line: 11, type: !171, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !204)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !34)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 2, scope: !194, type: !34)
// CHECK:STDOUT: !202 = !DILocation(line: 11, column: 34, scope: !194)
// CHECK:STDOUT: !203 = !DILocation(line: 11, column: 1, scope: !194)
// CHECK:STDOUT: !204 = !{!195, !196}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !168, line: 14, type: !171, spFlags: DISPFlagDefinition, unit: !169, retainedNodes: !215)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !34)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 2, scope: !205, type: !34)
// CHECK:STDOUT: !213 = !DILocation(line: 14, column: 34, scope: !205)
// CHECK:STDOUT: !214 = !DILocation(line: 14, column: 1, scope: !205)
// CHECK:STDOUT: !215 = !{!206, !207}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'types.carbon'
// CHECK:STDOUT: source_filename = "types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define half @_CTestF16.Main(half %a, half %b) #0 !dbg !218 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F16.call = fadd half %a, %b, !dbg !227
// CHECK:STDOUT:   ret half %F16.call, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestF32.Main(float %a, float %b) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F32.call = fadd float %a, %b, !dbg !239
// CHECK:STDOUT:   ret float %F32.call, !dbg !240
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestF64.Main(double %a, double %b) #0 !dbg !242 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F64.call = fadd double %a, %b, !dbg !251
// CHECK:STDOUT:   ret double %F64.call, !dbg !252
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestF128.Main(fp128 %a, fp128 %b) #0 !dbg !254 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F128.call = fadd fp128 %a, %b, !dbg !263
// CHECK:STDOUT:   ret fp128 %F128.call, !dbg !264
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!217}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{!34, !34, !34}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !216 = !DIFile(filename: "types.carbon", directory: "")
// CHECK:STDOUT: !217 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !216, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "TestF16", linkageName: "_CTestF16.Main", scope: null, file: !216, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !217, retainedNodes: !229)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !34)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 2, scope: !218, type: !34)
// CHECK:STDOUT: !227 = !DILocation(line: 5, column: 44, scope: !218)
// CHECK:STDOUT: !228 = !DILocation(line: 5, column: 37, scope: !218)
// CHECK:STDOUT: !229 = !{!219, !220}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "TestF32", linkageName: "_CTestF32.Main", scope: null, file: !216, line: 8, type: !38, spFlags: DISPFlagDefinition, unit: !217, retainedNodes: !241)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !34)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 2, scope: !230, type: !34)
// CHECK:STDOUT: !239 = !DILocation(line: 8, column: 44, scope: !230)
// CHECK:STDOUT: !240 = !DILocation(line: 8, column: 37, scope: !230)
// CHECK:STDOUT: !241 = !{!231, !232}
// CHECK:STDOUT: !242 = distinct !DISubprogram(name: "TestF64", linkageName: "_CTestF64.Main", scope: null, file: !216, line: 11, type: !38, spFlags: DISPFlagDefinition, unit: !217, retainedNodes: !253)
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 1, scope: !242, type: !34)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 2, scope: !242, type: !34)
// CHECK:STDOUT: !251 = !DILocation(line: 11, column: 44, scope: !242)
// CHECK:STDOUT: !252 = !DILocation(line: 11, column: 37, scope: !242)
// CHECK:STDOUT: !253 = !{!243, !244}
// CHECK:STDOUT: !254 = distinct !DISubprogram(name: "TestF128", linkageName: "_CTestF128.Main", scope: null, file: !216, line: 14, type: !38, spFlags: DISPFlagDefinition, unit: !217, retainedNodes: !265)
// CHECK:STDOUT: !255 = !DILocalVariable(arg: 1, scope: !254, type: !34)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 2, scope: !254, type: !34)
// CHECK:STDOUT: !263 = !DILocation(line: 14, column: 48, scope: !254)
// CHECK:STDOUT: !264 = !DILocation(line: 14, column: 41, scope: !254)
// CHECK:STDOUT: !265 = !{!255, !256}
// CHECK:STDOUT:
