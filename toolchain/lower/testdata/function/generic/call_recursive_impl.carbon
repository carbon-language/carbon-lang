// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --include-carbon-core
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
fn G(generic T: I, count: i32) -> i32 {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_impl.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.Y.Main:I.Main"() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call.loc41 = call i32 @_CG.Main.d0cc85b6c89517e3(i32 0), !dbg !18
// CHECK:STDOUT:   %G.call.loc42 = call i32 @_CG.Main.e8d4d639bae9ba40(i32 0), !dbg !21
// CHECK:STDOUT:   ret i32 0, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.d0cc85b6c89517e3(i32 %count) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(), !dbg !28
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !29
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !33
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.d0cc85b6c89517e3(i32 %Int.as.AddWith.impl.Op.call), !dbg !34
// CHECK:STDOUT:   ret i32 %G.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CG.Main.e8d4d639bae9ba40(i32 %count) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.Y.Main:I.Main"(), !dbg !42
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !43
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !44
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !46
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !47
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main.e8d4d639bae9ba40(i32 %Int.as.AddWith.impl.Op.call), !dbg !48
// CHECK:STDOUT:   ret i32 %G.call, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @printf, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_impl.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{!6}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !4, line: 21, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DILocation(line: 21, column: 12, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 21, column: 3, scope: !11)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.Y.Main:I.Main", scope: null, file: !4, line: 25, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 25, column: 12, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 25, column: 3, scope: !14)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !4, line: 40, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 41, column: 3, scope: !17)
// CHECK:STDOUT: !19 = !{!6, !6}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 42, column: 3, scope: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 40, column: 1, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.d0cc85b6c89517e3", scope: null, file: !4, line: 31, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !36)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !6)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 3, scope: !23)
// CHECK:STDOUT: !29 = !DILocation(line: 34, column: 7, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 34, column: 6, scope: !23)
// CHECK:STDOUT: !32 = !DILocation(line: 35, column: 5, scope: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 37, column: 15, scope: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 37, column: 10, scope: !23)
// CHECK:STDOUT: !35 = !DILocation(line: 37, column: 3, scope: !23)
// CHECK:STDOUT: !36 = !{!24}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e8d4d639bae9ba40", scope: null, file: !4, line: 31, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !6)
// CHECK:STDOUT: !42 = !DILocation(line: 32, column: 3, scope: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 34, column: 7, scope: !37)
// CHECK:STDOUT: !44 = !DILocation(line: 34, column: 6, scope: !37)
// CHECK:STDOUT: !46 = !DILocation(line: 35, column: 5, scope: !37)
// CHECK:STDOUT: !47 = !DILocation(line: 37, column: 15, scope: !37)
// CHECK:STDOUT: !48 = !DILocation(line: 37, column: 10, scope: !37)
// CHECK:STDOUT: !49 = !DILocation(line: 37, column: 3, scope: !37)
// CHECK:STDOUT: !50 = !{!38}
// CHECK:STDOUT:
