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
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.7b08d4bea697a676"(i32 %x) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.3617edd7cf4142da"(i32 %x) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.7b08d4bea697a676", scope: null, file: !20, line: 3, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !32)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !22)
// CHECK:STDOUT: !31 = !DILocation(line: 3, column: 39, scope: !25)
// CHECK:STDOUT: !32 = !{!26}
// CHECK:STDOUT: !33 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.3617edd7cf4142da", scope: null, file: !20, line: 4, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !37)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 4, column: 38, scope: !33)
// CHECK:STDOUT: !37 = !{!34}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'main.carbon'
// CHECK:STDOUT: source_filename = "main.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn.call.loc11_56 = call i32 @"_Cfn:enclosed.7b08d4bea697a676"(i32 1), !dbg !45
// CHECK:STDOUT:   %fn.call.loc11_57 = call i32 @"_Cfn:enclosed.3617edd7cf4142da"(i32 %fn.call.loc11_56), !dbg !46
// CHECK:STDOUT:   %fn.call.loc11_58 = call i32 @"_Cfn:enclosed.457746e0e6b66a51.2fad9574e050607e"(i32 %fn.call.loc11_57), !dbg !47
// CHECK:STDOUT:   %fn.call.loc11_59 = call i32 @"_Cfn:enclosed.7c4a94efd515b6fa"(i32 %fn.call.loc11_58), !dbg !48
// CHECK:STDOUT:   %fn.call.loc11_60 = call i32 @"_Cfn:enclosed.f7ae899d984eeaea.2fad9574e050607e"(i32 %fn.call.loc11_59), !dbg !49
// CHECK:STDOUT:   %fn.call.loc11_61 = call i32 @"_Cfn:enclosed.4410c9017de102ba.2fad9574e050607e"(i32 %fn.call.loc11_60), !dbg !50
// CHECK:STDOUT:   ret i32 %fn.call.loc11_61, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.7c4a94efd515b6fa"(i32 %x) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_Cfn:enclosed.3617edd7cf4142da"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_Cfn:enclosed.7b08d4bea697a676"(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_Cfn:enclosed.457746e0e6b66a51.2fad9574e050607e"(i32 %x) #0 !dbg !60 {
// CHECK:STDOUT:   ret i32 %x, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_Cfn:enclosed.f7ae899d984eeaea.2fad9574e050607e"(i32 %x) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_Cfn:enclosed.4410c9017de102ba.2fad9574e050607e"(i32 %y) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %y, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!39}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !38 = !DIFile(filename: "main.carbon", directory: "")
// CHECK:STDOUT: !39 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !38, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !40 = !{!22}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !38, line: 5, type: !41, spFlags: DISPFlagDefinition, unit: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 11, column: 46, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 11, column: 32, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 11, column: 11, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 11, column: 9, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 11, column: 7, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 10, column: 10, scope: !42)
// CHECK:STDOUT: !51 = !DILocation(line: 10, column: 3, scope: !42)
// CHECK:STDOUT: !52 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.7c4a94efd515b6fa", scope: null, file: !38, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !59)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !22)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 5, scope: !52)
// CHECK:STDOUT: !59 = !{!53}
// CHECK:STDOUT: !60 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.457746e0e6b66a51.2fad9574e050607e", scope: null, file: !20, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !64)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !22)
// CHECK:STDOUT: !63 = !DILocation(line: 5, column: 57, scope: !60)
// CHECK:STDOUT: !64 = !{!61}
// CHECK:STDOUT: !65 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.f7ae899d984eeaea.2fad9574e050607e", scope: null, file: !38, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !69)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !22)
// CHECK:STDOUT: !68 = !DILocation(line: 9, column: 40, scope: !65)
// CHECK:STDOUT: !69 = !{!66}
// CHECK:STDOUT: !70 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.4410c9017de102ba.2fad9574e050607e", scope: null, file: !38, line: 10, type: !24, spFlags: DISPFlagDefinition, unit: !39, retainedNodes: !74)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !22)
// CHECK:STDOUT: !73 = !DILocation(line: 10, column: 38, scope: !70)
// CHECK:STDOUT: !74 = !{!71}
// CHECK:STDOUT:
