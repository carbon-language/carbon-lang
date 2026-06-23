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

// TODO: Return `Core.CppCompat.LongLong64` when it's copyable.
fn DoubleLongLong64(ll64: Core.CppCompat.LongLong64) -> i64 {
  // TODO: Avoid explicit conversion when multiplication on `LongLong64` is available.
  return 2 * (ll64 as i64);
}

// TODO: Return `Core.CppCompat.ULongLong64` when it's copyable.
fn DoubleULong64(ul64: Core.CppCompat.ULongLong64) -> u64 {
  // TODO: Avoid explicit conversion when multiplication on `ULongLong64` is available.
  return 2 * (ul64 as u64);
}

// CHECK:STDOUT: ; ModuleID = 'primitives.carbon'
// CHECK:STDOUT: source_filename = "primitives.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CDoubleLong32.Main(i32 %l32) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i32 2, %l32, !dbg !10
// CHECK:STDOUT:   ret i32 %T.as_type.as.MulWith.impl.Op.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CDoubleULong32.Main(i32 %ul32) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i32 2, %ul32, !dbg !18
// CHECK:STDOUT:   ret i32 %T.as_type.as.MulWith.impl.Op.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CDoubleLongLong64.Main(i64 %ll64) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i64 2, %ll64, !dbg !26
// CHECK:STDOUT:   ret i64 %T.as_type.as.MulWith.impl.Op.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CDoubleULong64.Main(i64 %ul64) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.MulWith.impl.Op.call = mul i64 2, %ul64, !dbg !34
// CHECK:STDOUT:   ret i64 %T.as_type.as.MulWith.impl.Op.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "primitives.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "DoubleLong32", linkageName: "_CDoubleLong32.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "DoubleULong32", linkageName: "_CDoubleULong32.Main", scope: null, file: !1, line: 11, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15, !15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 13, column: 10, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 13, column: 3, scope: !12)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "DoubleLongLong64", linkageName: "_CDoubleLongLong64.Main", scope: null, file: !1, line: 17, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!23, !23}
// CHECK:STDOUT: !23 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 10, scope: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 3, scope: !20)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "DoubleULong64", linkageName: "_CDoubleULong64.Main", scope: null, file: !1, line: 23, type: !29, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !32)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{!31, !31}
// CHECK:STDOUT: !31 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !28, type: !31)
// CHECK:STDOUT: !34 = !DILocation(line: 25, column: 10, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 25, column: 3, scope: !28)
