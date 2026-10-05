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

// --- main.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn Test() -> i32 {
  alias f = fn (x: i32) -> i32 {
    return x;
  };
  alias g = fn (x: i32) => x;
  return (fn (y: i32) => y)(g(f(ImportedTerse(Imported(1)))));
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.3c14c8ca6559e7d0"(i32 %x) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.57eff363c02bd6d7"(i32 %x) #0 !dbg !31 {
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
// CHECK:STDOUT: !23 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.3c14c8ca6559e7d0", scope: null, file: !18, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !30)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 3, column: 39, scope: !23)
// CHECK:STDOUT: !30 = !{!24}
// CHECK:STDOUT: !31 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.57eff363c02bd6d7", scope: null, file: !18, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !35)
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
// CHECK:STDOUT:   %fn.call.loc10_57 = call i32 @"_Cfn:enclosed.3c14c8ca6559e7d0"(i32 1), !dbg !43
// CHECK:STDOUT:   %fn.call.loc10_58 = call i32 @"_Cfn:enclosed.57eff363c02bd6d7"(i32 %fn.call.loc10_57), !dbg !44
// CHECK:STDOUT:   %fn.call.loc10_59 = call i32 @"_Cfn:enclosed.78e97d08196088b9"(i32 %fn.call.loc10_58), !dbg !45
// CHECK:STDOUT:   %fn.call.loc10_60 = call i32 @"_Cfn:enclosed.0e7773781338b8b4"(i32 %fn.call.loc10_59), !dbg !46
// CHECK:STDOUT:   %fn.call.loc10_61 = call i32 @"_Cfn:enclosed.1d7712df3766e2e7"(i32 %fn.call.loc10_60), !dbg !47
// CHECK:STDOUT:   ret i32 %fn.call.loc10_61, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.78e97d08196088b9"(i32 %x) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.0e7773781338b8b4"(i32 %x) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.1d7712df3766e2e7"(i32 %y) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %y, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_Cfn:enclosed.57eff363c02bd6d7"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_Cfn:enclosed.3c14c8ca6559e7d0"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !36 = !DIFile(filename: "main.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = !{!20}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !36, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !43 = !DILocation(line: 10, column: 47, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 10, column: 33, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 10, column: 31, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 10, column: 29, scope: !40)
// CHECK:STDOUT: !47 = !DILocation(line: 10, column: 10, scope: !40)
// CHECK:STDOUT: !48 = !DILocation(line: 10, column: 3, scope: !40)
// CHECK:STDOUT: !49 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.78e97d08196088b9", scope: null, file: !36, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !56)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !20)
// CHECK:STDOUT: !55 = !DILocation(line: 7, column: 5, scope: !49)
// CHECK:STDOUT: !56 = !{!50}
// CHECK:STDOUT: !57 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.0e7773781338b8b4", scope: null, file: !36, line: 9, type: !22, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !61)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !20)
// CHECK:STDOUT: !60 = !DILocation(line: 9, column: 28, scope: !57)
// CHECK:STDOUT: !61 = !{!58}
// CHECK:STDOUT: !62 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.1d7712df3766e2e7", scope: null, file: !36, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !66)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 10, column: 26, scope: !62)
// CHECK:STDOUT: !66 = !{!63}
// CHECK:STDOUT:
