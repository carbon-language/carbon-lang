// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/overloaded.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/overloaded.carbon

class Number {
  var is_positive: bool;
}

impl Number as Core.Negate where .Result = Number {
  fn Op(self: Number) -> Number {
    return {.is_positive = not self.is_positive};
  }
}
impl Number as Core.MulWith(Number) where .Result = Number {
  fn Op(self: Number, other: Number) -> Number {
    return {.is_positive = (self.is_positive and other.is_positive) or
                            (not self.is_positive and not other.is_positive)};
  }
}

fn Calculate(a: Number, b: Number) -> Number {
  return -a * b;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'overloaded.carbon'
// CHECK:STDOUT: source_filename = "overloaded.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.Number.Main:Negate.Core"(ptr sret({ i1 }) %return, ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_36.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   %.loc19_36.2 = load i8, ptr %.loc19_36.1.is_positive, align 1, !dbg !18
// CHECK:STDOUT:   %.loc19_36.21 = trunc i8 %.loc19_36.2 to i1, !dbg !18
// CHECK:STDOUT:   %.loc19_28 = xor i1 %.loc19_36.21, true, !dbg !19
// CHECK:STDOUT:   %.loc19_48.2.is_positive = getelementptr inbounds nuw { i1 }, ptr %return, i32 0, i32 0, !dbg !20
// CHECK:STDOUT:   %.loc19_48.3 = zext i1 %.loc19_28 to i8, !dbg !20
// CHECK:STDOUT:   store i8 %.loc19_48.3, ptr %.loc19_48.2.is_positive, align 1, !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.Number.Main:MulWith.124acaad20680602.Core"(ptr sret({ i1 }) %return, ptr %self, ptr %other) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_33.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   %.loc24_33.2 = load i8, ptr %.loc24_33.1.is_positive, align 1, !dbg !30
// CHECK:STDOUT:   %.loc24_33.21 = trunc i8 %.loc24_33.2 to i1, !dbg !30
// CHECK:STDOUT:   br i1 %.loc24_33.21, label %and.rhs.loc24, label %and.result.loc24, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc24:                                    ; preds = %entry
// CHECK:STDOUT:   %.loc24_55.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   %.loc24_55.2 = load i8, ptr %.loc24_55.1.is_positive, align 1, !dbg !31
// CHECK:STDOUT:   %.loc24_55.22 = trunc i8 %.loc24_55.2 to i1, !dbg !31
// CHECK:STDOUT:   br label %and.result.loc24, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc24:                                 ; preds = %and.rhs.loc24, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %.loc24_55.22, %and.rhs.loc24 ]
// CHECK:STDOUT:   %.loc24_69.1 = xor i1 %0, true, !dbg !32
// CHECK:STDOUT:   br i1 %.loc24_69.1, label %or.rhs, label %or.result, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %and.result.loc24
// CHECK:STDOUT:   %.loc25_38.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc25_38.2 = load i8, ptr %.loc25_38.1.is_positive, align 1, !dbg !33
// CHECK:STDOUT:   %.loc25_38.23 = trunc i8 %.loc25_38.2 to i1, !dbg !33
// CHECK:STDOUT:   %.loc25_30 = xor i1 %.loc25_38.23, true, !dbg !34
// CHECK:STDOUT:   br i1 %.loc25_30, label %and.rhs.loc25, label %and.result.loc25, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc25:                                    ; preds = %or.rhs
// CHECK:STDOUT:   %.loc25_64.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc25_64.2 = load i8, ptr %.loc25_64.1.is_positive, align 1, !dbg !35
// CHECK:STDOUT:   %.loc25_64.24 = trunc i8 %.loc25_64.2 to i1, !dbg !35
// CHECK:STDOUT:   %.loc25_55 = xor i1 %.loc25_64.24, true, !dbg !36
// CHECK:STDOUT:   br label %and.result.loc25, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc25:                                 ; preds = %and.rhs.loc25, %or.rhs
// CHECK:STDOUT:   %1 = phi i1 [ false, %or.rhs ], [ %.loc25_55, %and.rhs.loc25 ]
// CHECK:STDOUT:   br label %or.result, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %and.result.loc25, %and.result.loc24
// CHECK:STDOUT:   %2 = phi i1 [ true, %and.result.loc24 ], [ %1, %and.result.loc25 ]
// CHECK:STDOUT:   %.loc25_77.2.is_positive = getelementptr inbounds nuw { i1 }, ptr %return, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   %.loc25_77.3 = zext i1 %2 to i8, !dbg !37
// CHECK:STDOUT:   store i8 %.loc25_77.3, ptr %.loc25_77.2.is_positive, align 1, !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCalculate.Main(ptr sret({ i1 }) %return, ptr %a, ptr %b) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc30_10.1.temp = alloca { i1 }, align 1, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc30_10.1.temp), !dbg !48
// CHECK:STDOUT:   call void @"_COp.Number.Main:Negate.Core"(ptr %.loc30_10.1.temp, ptr %a), !dbg !48
// CHECK:STDOUT:   call void @"_COp.Number.Main:MulWith.124acaad20680602.Core"(ptr %return, ptr %.loc30_10.1.temp, ptr %b), !dbg !48
// CHECK:STDOUT:   call void @"_COp.3e6115156c640ef2:core.Destroy.Core"(ptr %.loc30_10.1.temp), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4fdb30b3bf76b447:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e6115156c640ef2:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "overloaded.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !7 = !{!6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{!6, !6, !6}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !{null, !6}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Number.Main:Negate.Core", scope: null, file: !4, line: 18, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !13, type: !6)
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 32, scope: !13)
// CHECK:STDOUT: !19 = !DILocation(line: 19, column: 28, scope: !13)
// CHECK:STDOUT: !20 = !DILocation(line: 19, column: 12, scope: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 19, column: 5, scope: !13)
// CHECK:STDOUT: !22 = !{!14}
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Number.Main:MulWith.124acaad20680602.Core", scope: null, file: !4, line: 23, type: !10, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !39)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !6)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !23, type: !6)
// CHECK:STDOUT: !30 = !DILocation(line: 24, column: 29, scope: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 24, column: 50, scope: !23)
// CHECK:STDOUT: !32 = !DILocation(line: 24, column: 28, scope: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 25, column: 34, scope: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 25, column: 30, scope: !23)
// CHECK:STDOUT: !35 = !DILocation(line: 25, column: 59, scope: !23)
// CHECK:STDOUT: !36 = !DILocation(line: 25, column: 55, scope: !23)
// CHECK:STDOUT: !37 = !DILocation(line: 24, column: 12, scope: !23)
// CHECK:STDOUT: !38 = !DILocation(line: 24, column: 5, scope: !23)
// CHECK:STDOUT: !39 = !{!24, !25}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Calculate", linkageName: "_CCalculate.Main", scope: null, file: !4, line: 29, type: !10, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !51)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !6)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !40, type: !6)
// CHECK:STDOUT: !48 = !DILocation(line: 30, column: 10, scope: !40)
// CHECK:STDOUT: !50 = !DILocation(line: 30, column: 3, scope: !40)
// CHECK:STDOUT: !51 = !{!41, !42}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4fdb30b3bf76b447:core.Destroy.Core", scope: null, file: !4, line: 30, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !6)
// CHECK:STDOUT: !54 = !DILocation(line: 30, column: 10, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e6115156c640ef2:core.Destroy.Core", scope: null, file: !4, line: 30, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !6)
// CHECK:STDOUT: !58 = !DILocation(line: 30, column: 10, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT:
