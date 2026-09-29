// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/template/operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/template/operator.carbon

// --- unary.carbon
library "[[@TEST_NAME]]";

fn Negate[template T: type](a: T) -> T {
  return -a;
}

fn CallNegate(a: i32) -> i32 { return Negate(a); }

// --- binary.carbon
library "[[@TEST_NAME]]";

fn Add[template T: type](a: T, b: T) -> T {
  return a + b;
}

fn CallAdd(a: i32, b: i32) -> i32 { return Add(a, b); }

// --- chained.carbon
library "[[@TEST_NAME]]";

fn Polynomial[template T: type](n: T) -> T {
  return 3 * n * n + 5 * n - 2;
}

fn CallPolynomial(n: i32) -> i32 { return Polynomial(n); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'unary.carbon'
// CHECK:STDOUT: source_filename = "unary.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallNegate.Main(i32 %a) #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = call i32 @_CNegate.Main.84588f41d61dafba(i32 %a), !dbg !15
// CHECK:STDOUT:   ret i32 %Negate.call, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CNegate.Main.84588f41d61dafba(i32 %a) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !24
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "unary.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "CallNegate", linkageName: "_CCallNegate.Main", scope: null, file: !4, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !17)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !9, type: !6)
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 39, scope: !9)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 32, scope: !9)
// CHECK:STDOUT: !17 = !{!10}
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Negate", linkageName: "_CNegate.Main.84588f41d61dafba", scope: null, file: !4, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !18, type: !6)
// CHECK:STDOUT: !24 = !DILocation(line: 4, column: 10, scope: !18)
// CHECK:STDOUT: !25 = !DILocation(line: 4, column: 3, scope: !18)
// CHECK:STDOUT: !26 = !{!19}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'binary.carbon'
// CHECK:STDOUT: source_filename = "binary.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallAdd.Main(i32 %a, i32 %b) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = call i32 @_CAdd.Main.84588f41d61dafba(i32 %a, i32 %b), !dbg !40
// CHECK:STDOUT:   ret i32 %Add.call, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAdd.Main.84588f41d61dafba(i32 %a, i32 %b) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !50
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!28}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !27 = !DIFile(filename: "binary.carbon", directory: "")
// CHECK:STDOUT: !28 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !27, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !29 = !{!6, !6, !6}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "CallAdd", linkageName: "_CCallAdd.Main", scope: null, file: !27, line: 7, type: !30, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !42)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !6)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !31, type: !6)
// CHECK:STDOUT: !40 = !DILocation(line: 7, column: 44, scope: !31)
// CHECK:STDOUT: !41 = !DILocation(line: 7, column: 37, scope: !31)
// CHECK:STDOUT: !42 = !{!32, !33}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Add", linkageName: "_CAdd.Main.84588f41d61dafba", scope: null, file: !27, line: 3, type: !30, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !53)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !6)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 2, scope: !43, type: !6)
// CHECK:STDOUT: !50 = !DILocation(line: 4, column: 10, scope: !43)
// CHECK:STDOUT: !52 = !DILocation(line: 4, column: 3, scope: !43)
// CHECK:STDOUT: !53 = !{!44, !45}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'chained.carbon'
// CHECK:STDOUT: source_filename = "chained.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallPolynomial.Main(i32 %n) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Polynomial.call = call i32 @_CPolynomial.Main.84588f41d61dafba(i32 %n), !dbg !62
// CHECK:STDOUT:   ret i32 %Polynomial.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CPolynomial.Main.84588f41d61dafba(i32 %n) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call.83a = mul i32 3, %n, !dbg !71
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.call = mul i32 %T.as_type.as.MulWith.impl.Op.call.83a, %n, !dbg !71
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call.958 = mul i32 5, %n, !dbg !74
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Int.as.MulWith.impl.Op.call, %T.as_type.as.MulWith.impl.Op.call.958, !dbg !71
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.call = sub i32 %Int.as.AddWith.impl.Op.call, 2, !dbg !71
// CHECK:STDOUT:   ret i32 %Int.as.SubWith.impl.Op.call, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!55}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !54 = !DIFile(filename: "chained.carbon", directory: "")
// CHECK:STDOUT: !55 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !54, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "CallPolynomial", linkageName: "_CCallPolynomial.Main", scope: null, file: !54, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !55, retainedNodes: !64)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !6)
// CHECK:STDOUT: !62 = !DILocation(line: 7, column: 43, scope: !56)
// CHECK:STDOUT: !63 = !DILocation(line: 7, column: 36, scope: !56)
// CHECK:STDOUT: !64 = !{!57}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Polynomial", linkageName: "_CPolynomial.Main.84588f41d61dafba", scope: null, file: !54, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !55, retainedNodes: !76)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !6)
// CHECK:STDOUT: !71 = !DILocation(line: 4, column: 10, scope: !65)
// CHECK:STDOUT: !74 = !DILocation(line: 4, column: 22, scope: !65)
// CHECK:STDOUT: !75 = !DILocation(line: 4, column: 3, scope: !65)
// CHECK:STDOUT: !76 = !{!66}
// CHECK:STDOUT:
