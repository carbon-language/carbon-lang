// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_impl.carbon

import Core library "io";

interface I {
  fn F();
}
class X {}
impl X as I {
  fn F() { Core.Print(1); }
}
class Y {}
impl Y as I {
  fn F() { Core.Print(2); }
}

// Builds on "call_different_impls.carbon", cannot lower a single G due to
// different calls to F. Additionally, add recursion so the function
// fingerprint is dependent on itself.
fn G(T:! I, count: i32) -> i32 {
  T.F();

  if (count > 0) {
    return count;
  }
  return G(T, count + 1);
}

fn Run() {
  G(X, 0);
  G(Y, 0);
}

// CHECK:STDOUT: ; ModuleID = 'call_recursive_impl.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.Y.Main:I.Main"() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call.loc40 = call i32 @_CG.Main.4d3c36eeac175c1f(i32 0), !dbg !13
// CHECK:STDOUT:   %G.call.loc41 = call i32 @_CG.Main.7abe23e63361d139(i32 0), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.4d3c36eeac175c1f(i32 %count) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(), !dbg !22
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !23
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !24
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !25
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !26
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.4d3c36eeac175c1f(i32 %Int.as.AddWith.impl.Op.call), !dbg !27
// CHECK:STDOUT:   ret i32 %G.call, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.7abe23e63361d139(i32 %count) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.Y.Main:I.Main"(), !dbg !32
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !33
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !36
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.7abe23e63361d139(i32 %Int.as.AddWith.impl.Op.call), !dbg !37
// CHECK:STDOUT:   ret i32 %G.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @printf, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_impl.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 20, column: 12, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "F", linkageName: "_CF.Y.Main:I.Main", scope: null, file: !3, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 24, column: 12, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 24, column: 3, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 39, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 40, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 41, column: 3, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 39, column: 1, scope: !12)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.4d3c36eeac175c1f", scope: null, file: !3, line: 30, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{!19, !19}
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !16, type: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 31, column: 3, scope: !16)
// CHECK:STDOUT: !23 = !DILocation(line: 33, column: 7, scope: !16)
// CHECK:STDOUT: !24 = !DILocation(line: 33, column: 6, scope: !16)
// CHECK:STDOUT: !25 = !DILocation(line: 34, column: 5, scope: !16)
// CHECK:STDOUT: !26 = !DILocation(line: 36, column: 15, scope: !16)
// CHECK:STDOUT: !27 = !DILocation(line: 36, column: 10, scope: !16)
// CHECK:STDOUT: !28 = !DILocation(line: 36, column: 3, scope: !16)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.7abe23e63361d139", scope: null, file: !3, line: 30, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !19)
// CHECK:STDOUT: !32 = !DILocation(line: 31, column: 3, scope: !29)
// CHECK:STDOUT: !33 = !DILocation(line: 33, column: 7, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 33, column: 6, scope: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 34, column: 5, scope: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 36, column: 15, scope: !29)
// CHECK:STDOUT: !37 = !DILocation(line: 36, column: 10, scope: !29)
// CHECK:STDOUT: !38 = !DILocation(line: 36, column: 3, scope: !29)
