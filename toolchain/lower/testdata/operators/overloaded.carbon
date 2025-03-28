// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/overloaded.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/overloaded.carbon

class Number {
  var is_positive: bool;
}

impl Number as Core.Negate {
  fn Op[self: Number]() -> Number {
    return {.is_positive = not self.is_positive};
  }
}
impl Number as Core.Mul {
  fn Op[self: Number](other: Number) -> Number {
    return {.is_positive = (self.is_positive and other.is_positive) or
                            (not self.is_positive and not other.is_positive)};
  }
}

fn Calculate(a: Number, b: Number) -> Number {
  return -a * b;
}

// CHECK:STDOUT: ; ModuleID = 'overloaded.carbon'
// CHECK:STDOUT: source_filename = "overloaded.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_COp.Number.Main:Negate.Core"(ptr sret({ i1 }) %return, ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_36.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !7
// CHECK:STDOUT:   %.loc17_36.2 = load i1, ptr %.loc17_36.1.is_positive, align 1, !dbg !7
// CHECK:STDOUT:   %.loc17_28 = xor i1 %.loc17_36.2, true, !dbg !8
// CHECK:STDOUT:   %.loc17_48.2.is_positive = getelementptr inbounds nuw { i1 }, ptr %return, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   store i1 %.loc17_28, ptr %.loc17_48.2.is_positive, align 1, !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_COp.Number.Main:Mul.Core"(ptr sret({ i1 }) %return, ptr %self, ptr %other) !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_33.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc22_33.2 = load i1, ptr %.loc22_33.1.is_positive, align 1, !dbg !12
// CHECK:STDOUT:   br i1 %.loc22_33.2, label %and.rhs.loc22, label %and.result.loc22, !dbg !12
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   %.loc22_55.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   %.loc22_55.2 = load i1, ptr %.loc22_55.1.is_positive, align 1, !dbg !13
// CHECK:STDOUT:   br label %and.result.loc22, !dbg !12
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc22:                                 ; preds = %and.rhs.loc22, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %.loc22_55.2, %and.rhs.loc22 ]
// CHECK:STDOUT:   %.loc22_69.1 = xor i1 %0, true, !dbg !14
// CHECK:STDOUT:   br i1 %.loc22_69.1, label %or.rhs, label %or.result, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %and.result.loc22
// CHECK:STDOUT:   %.loc23_38.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   %.loc23_38.2 = load i1, ptr %.loc23_38.1.is_positive, align 1, !dbg !15
// CHECK:STDOUT:   %.loc23_30 = xor i1 %.loc23_38.2, true, !dbg !16
// CHECK:STDOUT:   br i1 %.loc23_30, label %and.rhs.loc23, label %and.result.loc23, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc23:                                    ; preds = %or.rhs
// CHECK:STDOUT:   %.loc23_64.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   %.loc23_64.2 = load i1, ptr %.loc23_64.1.is_positive, align 1, !dbg !17
// CHECK:STDOUT:   %.loc23_55 = xor i1 %.loc23_64.2, true, !dbg !18
// CHECK:STDOUT:   br label %and.result.loc23, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc23:                                 ; preds = %and.rhs.loc23, %or.rhs
// CHECK:STDOUT:   %1 = phi i1 [ false, %or.rhs ], [ %.loc23_55, %and.rhs.loc23 ]
// CHECK:STDOUT:   br label %or.result, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %and.result.loc23, %and.result.loc22
// CHECK:STDOUT:   %2 = phi i1 [ true, %and.result.loc22 ], [ %1, %and.result.loc23 ]
// CHECK:STDOUT:   %.loc23_77.2.is_positive = getelementptr inbounds nuw { i1 }, ptr %return, i32 0, i32 0, !dbg !19
// CHECK:STDOUT:   store i1 %2, ptr %.loc23_77.2.is_positive, align 1, !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCalculate.Main(ptr sret({ i1 }) %return, ptr %a, ptr %b) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_10.1.temp = alloca { i1 }, align 8, !dbg !22
// CHECK:STDOUT:   call void @"_COp.Number.Main:Negate.Core"(ptr %.loc28_10.1.temp, ptr %a), !dbg !22
// CHECK:STDOUT:   call void @"_COp.Number.Main:Mul.Core"(ptr %return, ptr %.loc28_10.1.temp, ptr %b), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i1 true, { 2, 0, 1, 3, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "overloaded.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Number.Main:Negate.Core", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 17, column: 32, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 28, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 17, column: 12, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 5, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Number.Main:Mul.Core", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 29, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 22, column: 50, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 22, column: 28, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 34, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 23, column: 30, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 23, column: 59, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 23, column: 55, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 12, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 22, column: 5, scope: !11)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Calculate", linkageName: "_CCalculate.Main", scope: null, file: !3, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 28, column: 10, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 28, column: 3, scope: !21)
