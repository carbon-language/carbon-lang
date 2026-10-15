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

// --- main.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn Test() -> i32 {
  alias f = fn (x: i32) -> i32 {
    return x;
  };
  return (fn (y: i32) -> i32 { return y; })(f(Imported(1)));
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
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'main.carbon'
// CHECK:STDOUT: source_filename = "main.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn.call.loc9_57 = call i32 @"_Cfn:enclosed.3c14c8ca6559e7d0"(i32 1), !dbg !38
// CHECK:STDOUT:   %fn.call.loc9_58 = call i32 @"_Cfn:enclosed.4f93545957a800ba"(i32 %fn.call.loc9_57), !dbg !39
// CHECK:STDOUT:   %fn.call.loc9_59 = call i32 @"_Cfn:enclosed.1e20c3b8ac0c9b5a"(i32 %fn.call.loc9_58), !dbg !40
// CHECK:STDOUT:   ret i32 %fn.call.loc9_59, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.4f93545957a800ba"(i32 %x) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.1e20c3b8ac0c9b5a"(i32 %y) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %y, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_Cfn:enclosed.3c14c8ca6559e7d0"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!32}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !31 = !DIFile(filename: "main.carbon", directory: "")
// CHECK:STDOUT: !32 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !31, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !33 = !{!20}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !31, line: 5, type: !34, spFlags: DISPFlagDefinition, unit: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 9, column: 47, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 9, column: 45, scope: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 9, column: 10, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 9, column: 3, scope: !35)
// CHECK:STDOUT: !42 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.4f93545957a800ba", scope: null, file: !31, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !49)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !20)
// CHECK:STDOUT: !48 = !DILocation(line: 7, column: 5, scope: !42)
// CHECK:STDOUT: !49 = !{!43}
// CHECK:STDOUT: !50 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.1e20c3b8ac0c9b5a", scope: null, file: !31, line: 9, type: !22, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !57)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !20)
// CHECK:STDOUT: !56 = !DILocation(line: 9, column: 32, scope: !50)
// CHECK:STDOUT: !57 = !{!51}
// CHECK:STDOUT:
