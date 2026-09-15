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
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %Int.as.Negate.impl.Op.call), !dbg !25
// CHECK:STDOUT:   ret i32 %From.as_type.as.ImplicitAs.impl.Convert.call, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !28 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !33
// CHECK:STDOUT:   ret i32 %1, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !37 {
// CHECK:STDOUT:   ret i32 %self, !dbg !43
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
// CHECK:STDOUT: !27 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !27, line: 64, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !36)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !6)
// CHECK:STDOUT: !33 = !DILocation(line: 66, column: 17, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 66, column: 5, scope: !28)
// CHECK:STDOUT: !36 = !{!29}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !27, line: 59, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !44)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !6)
// CHECK:STDOUT: !43 = !DILocation(line: 59, column: 28, scope: !37)
// CHECK:STDOUT: !44 = !{!38}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'binary.carbon'
// CHECK:STDOUT: source_filename = "binary.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallAdd.Main(i32 %a, i32 %b) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = call i32 @_CAdd.Main.84588f41d61dafba(i32 %a, i32 %b), !dbg !58
// CHECK:STDOUT:   ret i32 %Add.call, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CAdd.Main.84588f41d61dafba(i32 %a, i32 %b) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !68
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %Int.as.AddWith.impl.Op.call), !dbg !70
// CHECK:STDOUT:   ret i32 %From.as_type.as.ImplicitAs.impl.Convert.call, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !72 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !77
// CHECK:STDOUT:   ret i32 %1, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !81 {
// CHECK:STDOUT:   ret i32 %self, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!46}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !27 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !45 = !DIFile(filename: "binary.carbon", directory: "")
// CHECK:STDOUT: !46 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !45, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !47 = !{!6, !6, !6}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "CallAdd", linkageName: "_CCallAdd.Main", scope: null, file: !45, line: 7, type: !48, spFlags: DISPFlagDefinition, unit: !46, retainedNodes: !60)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !6)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 2, scope: !49, type: !6)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 44, scope: !49)
// CHECK:STDOUT: !59 = !DILocation(line: 7, column: 37, scope: !49)
// CHECK:STDOUT: !60 = !{!50, !51}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Add", linkageName: "_CAdd.Main.84588f41d61dafba", scope: null, file: !45, line: 3, type: !48, spFlags: DISPFlagDefinition, unit: !46, retainedNodes: !71)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !6)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 2, scope: !61, type: !6)
// CHECK:STDOUT: !68 = !DILocation(line: 4, column: 10, scope: !61)
// CHECK:STDOUT: !70 = !DILocation(line: 4, column: 3, scope: !61)
// CHECK:STDOUT: !71 = !{!62, !63}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !27, line: 64, type: !8, spFlags: DISPFlagDefinition, unit: !46, retainedNodes: !80)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !6)
// CHECK:STDOUT: !77 = !DILocation(line: 66, column: 17, scope: !72)
// CHECK:STDOUT: !79 = !DILocation(line: 66, column: 5, scope: !72)
// CHECK:STDOUT: !80 = !{!73}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !27, line: 59, type: !8, spFlags: DISPFlagDefinition, unit: !46, retainedNodes: !88)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !6)
// CHECK:STDOUT: !87 = !DILocation(line: 59, column: 28, scope: !81)
// CHECK:STDOUT: !88 = !{!82}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'chained.carbon'
// CHECK:STDOUT: source_filename = "chained.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallPolynomial.Main(i32 %n) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Polynomial.call = call i32 @_CPolynomial.Main.84588f41d61dafba(i32 %n), !dbg !97
// CHECK:STDOUT:   ret i32 %Polynomial.call, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CPolynomial.Main.84588f41d61dafba(i32 %n) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call.83a = mul i32 3, %n, !dbg !106
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.call = mul i32 %T.as_type.as.MulWith.impl.Op.call.83a, %n, !dbg !106
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call.958 = mul i32 5, %n, !dbg !109
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Int.as.MulWith.impl.Op.call, %T.as_type.as.MulWith.impl.Op.call.958, !dbg !106
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.call = sub i32 %Int.as.AddWith.impl.Op.call, 2, !dbg !106
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %Int.as.SubWith.impl.Op.call), !dbg !110
// CHECK:STDOUT:   ret i32 %From.as_type.as.ImplicitAs.impl.Convert.call, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !112 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !117
// CHECK:STDOUT:   ret i32 %1, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !121 {
// CHECK:STDOUT:   ret i32 %self, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!90}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !27 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !89 = !DIFile(filename: "chained.carbon", directory: "")
// CHECK:STDOUT: !90 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !89, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "CallPolynomial", linkageName: "_CCallPolynomial.Main", scope: null, file: !89, line: 7, type: !8, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !99)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !6)
// CHECK:STDOUT: !97 = !DILocation(line: 7, column: 43, scope: !91)
// CHECK:STDOUT: !98 = !DILocation(line: 7, column: 36, scope: !91)
// CHECK:STDOUT: !99 = !{!92}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Polynomial", linkageName: "_CPolynomial.Main.84588f41d61dafba", scope: null, file: !89, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !111)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !6)
// CHECK:STDOUT: !106 = !DILocation(line: 4, column: 10, scope: !100)
// CHECK:STDOUT: !109 = !DILocation(line: 4, column: 22, scope: !100)
// CHECK:STDOUT: !110 = !DILocation(line: 4, column: 3, scope: !100)
// CHECK:STDOUT: !111 = !{!101}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !27, line: 64, type: !8, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !120)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !6)
// CHECK:STDOUT: !117 = !DILocation(line: 66, column: 17, scope: !112)
// CHECK:STDOUT: !119 = !DILocation(line: 66, column: 5, scope: !112)
// CHECK:STDOUT: !120 = !{!113}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !27, line: 59, type: !8, spFlags: DISPFlagDefinition, unit: !90, retainedNodes: !128)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !6)
// CHECK:STDOUT: !127 = !DILocation(line: 59, column: 28, scope: !121)
// CHECK:STDOUT: !128 = !{!122}
// CHECK:STDOUT:
