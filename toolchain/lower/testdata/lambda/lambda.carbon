// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/lambda/lambda.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/lambda/lambda.carbon

// --- lib.carbon
library "[[@TEST_NAME]]";

alias Imported = fn (x: i32) -> i32 { return x; };
alias ImportedTerse = fn (x: i32) => x;
alias ImportedGenericTerse = fn [T: Core.Copy](x: T) => x;

// --- main.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn Test() -> i32 {
  alias f = fn (x: i32) -> i32 {
    return x;
  };
  alias g = fn [T: Core.Copy](x: T) => x;
  return (fn [T: Core.Copy](y: T) => y)(
      g(f(ImportedGenericTerse(ImportedTerse(Imported(1))))));
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.ceee9ea6d55401eb"(i32 %x) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.869d128921d398dd"(i32 %x) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.ceee9ea6d55401eb", scope: null, file: !18, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !30)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 3, column: 39, scope: !23)
// CHECK:STDOUT: !30 = !{!24}
// CHECK:STDOUT: !31 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.869d128921d398dd", scope: null, file: !18, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !20)
// CHECK:STDOUT: !34 = !DILocation(line: 4, column: 38, scope: !31)
// CHECK:STDOUT: !35 = !{!32}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'main.carbon'
// CHECK:STDOUT: source_filename = "main.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn.call.loc11_56 = call i32 @"_Cfn:enclosed.ceee9ea6d55401eb"(i32 1), !dbg !43
// CHECK:STDOUT:   %fn.call.loc11_57 = call i32 @"_Cfn:enclosed.869d128921d398dd"(i32 %fn.call.loc11_56), !dbg !44
// CHECK:STDOUT:   %fn.call.loc11_58 = call i32 @"_Cfn:enclosed.24b0b9b2e7014b8d.2fad9574e050607e"(i32 %fn.call.loc11_57), !dbg !45
// CHECK:STDOUT:   %fn.call.loc11_59 = call i32 @"_Cfn:enclosed.78e97d08196088b9"(i32 %fn.call.loc11_58), !dbg !46
// CHECK:STDOUT:   %fn.call.loc11_60 = call i32 @"_Cfn:enclosed.2cf98fd9c300909e.2fad9574e050607e"(i32 %fn.call.loc11_59), !dbg !47
// CHECK:STDOUT:   %fn.call.loc11_61 = call i32 @"_Cfn:enclosed.5a8ed418d63f8fd2.2fad9574e050607e"(i32 %fn.call.loc11_60), !dbg !48
// CHECK:STDOUT:   ret i32 %fn.call.loc11_61, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.78e97d08196088b9"(i32 %x) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_Cfn:enclosed.869d128921d398dd"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_Cfn:enclosed.ceee9ea6d55401eb"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_Cfn:enclosed.24b0b9b2e7014b8d.2fad9574e050607e"(i32 %x) #0 !dbg !58 {
// CHECK:STDOUT:   ret i32 %x, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_Cfn:enclosed.2cf98fd9c300909e.2fad9574e050607e"(i32 %x) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_Cfn:enclosed.5a8ed418d63f8fd2.2fad9574e050607e"(i32 %y) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %y, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !36 = !DIFile(filename: "main.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = !{!20}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !36, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 11, column: 46, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 11, column: 32, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 11, column: 11, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 11, column: 9, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 11, column: 7, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 10, column: 10, scope: !40)
// CHECK:STDOUT: !49 = !DILocation(line: 10, column: 3, scope: !40)
// CHECK:STDOUT: !50 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.78e97d08196088b9", scope: null, file: !36, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !57)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !20)
// CHECK:STDOUT: !56 = !DILocation(line: 7, column: 5, scope: !50)
// CHECK:STDOUT: !57 = !{!51}
// CHECK:STDOUT: !58 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.24b0b9b2e7014b8d.2fad9574e050607e", scope: null, file: !18, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !62)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !20)
// CHECK:STDOUT: !61 = !DILocation(line: 5, column: 57, scope: !58)
// CHECK:STDOUT: !62 = !{!59}
// CHECK:STDOUT: !63 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.2cf98fd9c300909e.2fad9574e050607e", scope: null, file: !36, line: 9, type: !22, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !67)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !20)
// CHECK:STDOUT: !66 = !DILocation(line: 9, column: 40, scope: !63)
// CHECK:STDOUT: !67 = !{!64}
// CHECK:STDOUT: !68 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.5a8ed418d63f8fd2.2fad9574e050607e", scope: null, file: !36, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !72)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !20)
// CHECK:STDOUT: !71 = !DILocation(line: 10, column: 38, scope: !68)
// CHECK:STDOUT: !72 = !{!69}
// CHECK:STDOUT:
