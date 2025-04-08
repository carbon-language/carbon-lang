// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT: @printf.int.format.1 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CF.Y.Main:I.Main"() !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format.1, i32 2), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call.loc38 = call i32 @_CG.Main.562d5cdd5cdbd5c5(i32 0), !dbg !13
// CHECK:STDOUT:   %G.call.loc39 = call i32 @_CG.Main.386ac59331346fc4(i32 0), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main.562d5cdd5cdbd5c5(i32 %count) !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(), !dbg !17
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 0, !dbg !18
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !19
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !21
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.562d5cdd5cdbd5c5(i32 %int.sadd), !dbg !22
// CHECK:STDOUT:   ret i32 %G.call, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main.386ac59331346fc4(i32 %count) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.Y.Main:I.Main"(), !dbg !25
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 0, !dbg !26
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !29
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.386ac59331346fc4(i32 %int.sadd), !dbg !30
// CHECK:STDOUT:   ret i32 %G.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @printf, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_impl.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 12, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "F", linkageName: "_CF.Y.Main:I.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 12, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 22, column: 3, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 37, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 38, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 39, column: 3, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 37, column: 1, scope: !12)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.562d5cdd5cdbd5c5", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DILocation(line: 29, column: 3, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 31, column: 7, scope: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 31, column: 6, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 32, column: 5, scope: !16)
// CHECK:STDOUT: !21 = !DILocation(line: 34, column: 15, scope: !16)
// CHECK:STDOUT: !22 = !DILocation(line: 34, column: 10, scope: !16)
// CHECK:STDOUT: !23 = !DILocation(line: 34, column: 3, scope: !16)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.386ac59331346fc4", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 29, column: 3, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 31, column: 7, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 31, column: 6, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 5, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 34, column: 15, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 34, column: 10, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 34, column: 3, scope: !24)
