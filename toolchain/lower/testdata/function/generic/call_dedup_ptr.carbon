// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_dedup_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_dedup_ptr.carbon

fn F[T:! type](x: T) -> T {
  return x;
}

fn M() {
  // TODO: The definitions of F for these calls lower to different identical
  // functions taking a pointer param and returning a pointer. The three
  // specifics could be deduplicated.
  var ptr_i32 : i32*;
  F(ptr_i32);
  var ptr_f64 : f64*;
  F(ptr_f64);
  var ptr_i8 : i8*;
  F(ptr_i8);
}

// CHECK:STDOUT: ; ModuleID = 'call_dedup_ptr.carbon'
// CHECK:STDOUT: source_filename = "call_dedup_ptr.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i32.var), !dbg !7
// CHECK:STDOUT:   %.loc20 = load ptr, ptr %ptr_i32.var, align 8, !dbg !8
// CHECK:STDOUT:   %F.call.loc20 = call ptr @_CF.Main.e8193710fd35b608(ptr %.loc20), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_f64.var), !dbg !7
// CHECK:STDOUT:   %.loc22 = load ptr, ptr %ptr_f64.var, align 8, !dbg !10
// CHECK:STDOUT:   %F.call.loc22 = call ptr @_CF.Main.04bf2edaaa84aa22(ptr %.loc22), !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i8.var), !dbg !7
// CHECK:STDOUT:   %.loc24 = load ptr, ptr %ptr_i8.var, align 8, !dbg !12
// CHECK:STDOUT:   %F.call.loc24 = call ptr @_CF.Main.bda010de15e6a5ad(ptr %.loc24), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CF.Main.e8193710fd35b608(ptr %x) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CF.Main.04bf2edaaa84aa22(ptr %x) !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CF.Main.bda010de15e6a5ad(ptr %x) !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_dedup_ptr.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 20, column: 5, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 24, column: 5, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.e8193710fd35b608", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 12, column: 3, scope: !15)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !18 = !DILocation(line: 12, column: 3, scope: !17)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.bda010de15e6a5ad", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DILocation(line: 12, column: 3, scope: !19)
