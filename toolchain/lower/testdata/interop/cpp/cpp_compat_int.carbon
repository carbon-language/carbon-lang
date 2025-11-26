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
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.call = mul i32 2, %l32, !dbg !7
// CHECK:STDOUT:   ret i32 %T.binding.as_type.as.MulWith.impl.Op.call, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CDoubleULong32.Main(i32 %ul32) #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.call = mul i32 2, %ul32, !dbg !10
// CHECK:STDOUT:   ret i32 %T.binding.as_type.as.MulWith.impl.Op.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CDoubleLongLong64.Main(i64 %ll64) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.call = mul i64 2, %ll64, !dbg !13
// CHECK:STDOUT:   ret i64 %T.binding.as_type.as.MulWith.impl.Op.call, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CDoubleULong64.Main(i64 %ul64) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.binding.as_type.as.MulWith.impl.Op.call = mul i64 2, %ul64, !dbg !16
// CHECK:STDOUT:   ret i64 %T.binding.as_type.as.MulWith.impl.Op.call, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "primitives.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "DoubleLong32", linkageName: "_CDoubleLong32.Main", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "DoubleULong32", linkageName: "_CDoubleULong32.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 10, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 3, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "DoubleLongLong64", linkageName: "_CDoubleLongLong64.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 10, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 19, column: 3, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "DoubleULong64", linkageName: "_CDoubleULong64.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 25, column: 10, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 25, column: 3, scope: !15)
