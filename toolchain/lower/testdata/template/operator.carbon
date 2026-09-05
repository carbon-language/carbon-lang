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
// CHECK:STDOUT: define i32 @_CCallNegate.Main(i32 %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = call i32 @_CNegate.Main.84588f41d61dafba(i32 %a), !dbg !10
// CHECK:STDOUT:   ret i32 %Negate.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CNegate.Main.84588f41d61dafba(i32 %a) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !15
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %Int.as.Negate.impl.Op.call), !dbg !16
// CHECK:STDOUT:   ret i32 %From.as_type.as.ImplicitAs.impl.Convert.call, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !17 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !21
// CHECK:STDOUT:   ret i32 %1, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !23 {
// CHECK:STDOUT:   ret i32 %self, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "unary.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CallNegate", linkageName: "_CCallNegate.Main", scope: null, file: !1, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 39, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 32, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Negate", linkageName: "_CNegate.Main.84588f41d61dafba", scope: null, file: !1, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 4, column: 10, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 4, column: 3, scope: !12)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !18, line: 64, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !18 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 66, column: 17, scope: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 66, column: 5, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !18, line: 59, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 59, column: 28, scope: !23)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'binary.carbon'
// CHECK:STDOUT: source_filename = "binary.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallAdd.Main(i32 %a, i32 %b) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = call i32 @_CAdd.Main.84588f41d61dafba(i32 %a, i32 %b), !dbg !11
// CHECK:STDOUT:   ret i32 %Add.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAdd.Main.84588f41d61dafba(i32 %a, i32 %b) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !17
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %Int.as.AddWith.impl.Op.call), !dbg !18
// CHECK:STDOUT:   ret i32 %From.as_type.as.ImplicitAs.impl.Convert.call, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !19 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !25
// CHECK:STDOUT:   ret i32 %1, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !27 {
// CHECK:STDOUT:   ret i32 %self, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "binary.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CallAdd", linkageName: "_CCallAdd.Main", scope: null, file: !1, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 44, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 7, column: 37, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Add", linkageName: "_CAdd.Main.84588f41d61dafba", scope: null, file: !1, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !13, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 4, column: 10, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 4, column: 3, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !20, line: 64, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!7, !7}
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !25 = !DILocation(line: 66, column: 17, scope: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 66, column: 5, scope: !19)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !20, line: 59, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 59, column: 28, scope: !27)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'chained.carbon'
// CHECK:STDOUT: source_filename = "chained.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallPolynomial.Main(i32 %n) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Polynomial.call = call i32 @_CPolynomial.Main.84588f41d61dafba(i32 %n), !dbg !10
// CHECK:STDOUT:   ret i32 %Polynomial.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CPolynomial.Main.84588f41d61dafba(i32 %n) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call.83a = mul i32 3, %n, !dbg !15
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.call = mul i32 %T.as_type.as.MulWith.impl.Op.call.83a, %n, !dbg !15
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call.958 = mul i32 5, %n, !dbg !16
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Int.as.MulWith.impl.Op.call, %T.as_type.as.MulWith.impl.Op.call.958, !dbg !15
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.call = sub i32 %Int.as.AddWith.impl.Op.call, 2, !dbg !15
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %Int.as.SubWith.impl.Op.call), !dbg !17
// CHECK:STDOUT:   ret i32 %From.as_type.as.ImplicitAs.impl.Convert.call, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !18 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !22
// CHECK:STDOUT:   ret i32 %1, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !24 {
// CHECK:STDOUT:   ret i32 %self, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "chained.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CallPolynomial", linkageName: "_CCallPolynomial.Main", scope: null, file: !1, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 43, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 36, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Polynomial", linkageName: "_CPolynomial.Main.84588f41d61dafba", scope: null, file: !1, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 4, column: 10, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 4, column: 22, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 4, column: 3, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !19, line: 64, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 66, column: 17, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 66, column: 5, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !19, line: 59, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 59, column: 28, scope: !24)
// CHECK:STDOUT:
