// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/string_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/string_indexing.carbon

fn F() -> char {
    let c: char = "Test"[0];
    return c;
}

fn F2(index: i64) -> char {
    let c: char = "Test"[index];
    return c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'string_indexing.carbon'
// CHECK:STDOUT: source_filename = "string_indexing.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @0 = private unnamed_addr constant [5 x i8] c"Test\00", align 1
// CHECK:STDOUT: @String.val.String.val = internal constant { ptr, i64 } { ptr @0, i64 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CF.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 84, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CF2.Main(i64 %index) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call.string.load = load { ptr, i64 }, ptr @String.val.String.val, align 8, !dbg !25
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call.string.ptr = extractvalue { ptr, i64 } %str.as.IndexWith.impl.At.call.string.load, 0, !dbg !25
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call.string.char_ptr = getelementptr inbounds i8, ptr %str.as.IndexWith.impl.At.call.string.ptr, i64 %index, !dbg !25
// CHECK:STDOUT:   %str.as.IndexWith.impl.At.call.string.char = load i8, ptr %str.as.IndexWith.impl.At.call.string.char_ptr, align 1, !dbg !25
// CHECK:STDOUT:   ret i8 %str.as.IndexWith.impl.At.call.string.char, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "string_indexing.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!6}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{!6, !7}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !4, line: 13, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 5, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "F2", linkageName: "_CF2.Main", scope: null, file: !4, line: 18, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !25 = !DILocation(line: 19, column: 19, scope: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 5, scope: !19)
// CHECK:STDOUT: !30 = !{!20}
// CHECK:STDOUT:
