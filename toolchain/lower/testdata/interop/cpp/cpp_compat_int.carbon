// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/cpp_compat_int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/cpp_compat_int.carbon

// --- primitives.carbon

library "[[@TEST_NAME]]";

// TODO: Return `Core.CppCompat.Long32` when it's copyable.
fn DoubleLong32(l32: Core.CppCompat.Long32) -> i32 {
  // TODO: Avoid explicit conversion when multiplication on `Long32` is available.
  return 2 * (l32 as i32);
}

// TODO: Return `Core.CppCompat.ULong32` when it's copyable.
fn DoubleULong32(ul32: Core.CppCompat.ULong32) -> u32 {
  // TODO: Avoid explicit conversion when multiplication on `ULong32` is available.
  return 2 * (ul32 as u32);
}

// TODO: Return `Core.CppCompat.Long64` when it's copyable.
fn DoubleLong64(l64: Core.CppCompat.Long64) -> i64 {
  // TODO: Avoid explicit conversion when multiplication on `Long64` is available.
  return 2 * (l64 as i64);
}

// TODO: Return `Core.CppCompat.ULong64` when it's copyable.
fn DoubleULong64(ul64: Core.CppCompat.ULong64) -> u64 {
  // TODO: Avoid explicit conversion when multiplication on `ULong64` is available.
  return 2 * (ul64 as u64);
}

// TODO: Return `Core.CppCompat.LongLong64` when it's copyable.
fn DoubleLongLong64(ll64: Core.CppCompat.LongLong64) -> i64 {
  // TODO: Avoid explicit conversion when multiplication on `LongLong64` is available.
  return 2 * (ll64 as i64);
}

// TODO: Return `Core.CppCompat.ULongLong64` when it's copyable.
fn DoubleULongLong64(ull64: Core.CppCompat.ULongLong64) -> u64 {
  // TODO: Avoid explicit conversion when multiplication on `ULongLong64` is available.
  return 2 * (ull64 as u64);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'primitives.carbon'
// CHECK:STDOUT: source_filename = "primitives.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CDoubleLong32.Main(i32 %l32) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i32 2, %l32, !dbg !24
// CHECK:STDOUT:   ret i32 %T.as_type.as.MulWith.impl.Op.call, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CDoubleULong32.Main(i32 %ul32) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i32 2, %ul32, !dbg !33
// CHECK:STDOUT:   ret i32 %T.as_type.as.MulWith.impl.Op.call, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CDoubleLong64.Main(i64 %l64) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i64 2, %l64, !dbg !42
// CHECK:STDOUT:   ret i64 %T.as_type.as.MulWith.impl.Op.call, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CDoubleULong64.Main(i64 %ul64) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i64 2, %ul64, !dbg !51
// CHECK:STDOUT:   ret i64 %T.as_type.as.MulWith.impl.Op.call, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CDoubleLongLong64.Main(i64 %ll64) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i64 2, %ll64, !dbg !60
// CHECK:STDOUT:   ret i64 %T.as_type.as.MulWith.impl.Op.call, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CDoubleULongLong64.Main(i64 %ull64) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i64 2, %ull64, !dbg !69
// CHECK:STDOUT:   ret i64 %T.as_type.as.MulWith.impl.Op.call, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "primitives.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !10 = !{!6, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{!8, !8}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{!9, !9}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "DoubleLong32", linkageName: "_CDoubleLong32.Main", scope: null, file: !4, line: 5, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !18, type: !6)
// CHECK:STDOUT: !24 = !DILocation(line: 7, column: 10, scope: !18)
// CHECK:STDOUT: !25 = !DILocation(line: 7, column: 3, scope: !18)
// CHECK:STDOUT: !26 = !{!19}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "DoubleULong32", linkageName: "_CDoubleULong32.Main", scope: null, file: !4, line: 11, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !7)
// CHECK:STDOUT: !33 = !DILocation(line: 13, column: 10, scope: !27)
// CHECK:STDOUT: !34 = !DILocation(line: 13, column: 3, scope: !27)
// CHECK:STDOUT: !35 = !{!28}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "DoubleLong64", linkageName: "_CDoubleLong64.Main", scope: null, file: !4, line: 17, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !44)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !8)
// CHECK:STDOUT: !42 = !DILocation(line: 19, column: 10, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 19, column: 3, scope: !36)
// CHECK:STDOUT: !44 = !{!37}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "DoubleULong64", linkageName: "_CDoubleULong64.Main", scope: null, file: !4, line: 23, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !9)
// CHECK:STDOUT: !51 = !DILocation(line: 25, column: 10, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 25, column: 3, scope: !45)
// CHECK:STDOUT: !53 = !{!46}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "DoubleLongLong64", linkageName: "_CDoubleLongLong64.Main", scope: null, file: !4, line: 29, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !8)
// CHECK:STDOUT: !60 = !DILocation(line: 31, column: 10, scope: !54)
// CHECK:STDOUT: !61 = !DILocation(line: 31, column: 3, scope: !54)
// CHECK:STDOUT: !62 = !{!55}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "DoubleULongLong64", linkageName: "_CDoubleULongLong64.Main", scope: null, file: !4, line: 35, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !71)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !9)
// CHECK:STDOUT: !69 = !DILocation(line: 37, column: 10, scope: !63)
// CHECK:STDOUT: !70 = !DILocation(line: 37, column: 3, scope: !63)
// CHECK:STDOUT: !71 = !{!64}
// CHECK:STDOUT:
