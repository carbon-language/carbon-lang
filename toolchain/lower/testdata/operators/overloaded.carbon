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
// CHECK:STDOUT: define void @Op(ptr sret({ i1 }) %return, ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_36.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %.loc17_36.2 = load i1, ptr %.loc17_36.1.is_positive, align 1
// CHECK:STDOUT:   %.loc17_28 = xor i1 %.loc17_36.2, true
// CHECK:STDOUT:   %.loc17_48.2.is_positive = getelementptr inbounds nuw { i1 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i1 %.loc17_28, ptr %.loc17_48.2.is_positive, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.1(ptr sret({ i1 }) %return, ptr %self, ptr %other) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_33.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %.loc22_33.2 = load i1, ptr %.loc22_33.1.is_positive, align 1
// CHECK:STDOUT:   br i1 %.loc22_33.2, label %and.rhs.loc22, label %and.result.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc22:                                    ; preds = %entry
// CHECK:STDOUT:   %.loc22_55.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0
// CHECK:STDOUT:   %.loc22_55.2 = load i1, ptr %.loc22_55.1.is_positive, align 1
// CHECK:STDOUT:   br label %and.result.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc22:                                 ; preds = %and.rhs.loc22, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %.loc22_55.2, %and.rhs.loc22 ]
// CHECK:STDOUT:   %.loc22_69.1 = xor i1 %0, true
// CHECK:STDOUT:   br i1 %.loc22_69.1, label %or.rhs, label %or.result
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %and.result.loc22
// CHECK:STDOUT:   %.loc23_38.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %.loc23_38.2 = load i1, ptr %.loc23_38.1.is_positive, align 1
// CHECK:STDOUT:   %.loc23_30 = xor i1 %.loc23_38.2, true
// CHECK:STDOUT:   br i1 %.loc23_30, label %and.rhs.loc23, label %and.result.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc23:                                    ; preds = %or.rhs
// CHECK:STDOUT:   %.loc23_64.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0
// CHECK:STDOUT:   %.loc23_64.2 = load i1, ptr %.loc23_64.1.is_positive, align 1
// CHECK:STDOUT:   %.loc23_55 = xor i1 %.loc23_64.2, true
// CHECK:STDOUT:   br label %and.result.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc23:                                 ; preds = %and.rhs.loc23, %or.rhs
// CHECK:STDOUT:   %1 = phi i1 [ false, %or.rhs ], [ %.loc23_55, %and.rhs.loc23 ]
// CHECK:STDOUT:   br label %or.result
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %and.result.loc23, %and.result.loc22
// CHECK:STDOUT:   %2 = phi i1 [ true, %and.result.loc22 ], [ %1, %and.result.loc23 ]
// CHECK:STDOUT:   %.loc23_77.2.is_positive = getelementptr inbounds nuw { i1 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i1 %2, ptr %.loc23_77.2.is_positive, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Calculate(ptr sret({ i1 }) %return, ptr %a, ptr %b) !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_10.2.temp = alloca { i1 }, align 8
// CHECK:STDOUT:   call void @Op(ptr %.loc28_10.2.temp, ptr %a)
// CHECK:STDOUT:   call void @Op.1(ptr %return, ptr %.loc28_10.2.temp, ptr %b)
// CHECK:STDOUT:   ret void
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Op.1", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "Calculate", scope: null, file: !3, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !2)
