// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_deref_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_deref_ptr.carbon

// Builds on call_basic.carbon and call_impl_function.carbon.
// Test that when equivalence is found for the specific fingerprint, we don't
// conclude "equivalent" for all the call graph. This is to account for
// switching to a work-queue from recursion.

fn A[U:! Core.Copy](x: U) -> U {
  return x;
}

fn D[U:! Core.Copy](x: U) -> U {
  return x;
}

fn B[U:! Core.Copy & Core.Destroy](x: U*) {
  D(*x);
}

fn F[T:! Core.Copy & Core.Destroy](x: T*) {
  A(i32);
  B(x);
}

fn M() {
  var ptr_i32 : i32*;
  var ptr_f64 : f64*;

  F(ptr_i32);
  F(ptr_f64);
}

// CHECK:STDOUT: ; ModuleID = 'call_deref_ptr.carbon'
// CHECK:STDOUT: source_filename = "call_deref_ptr.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !8
// CHECK:STDOUT:   %.loc39_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !9
// CHECK:STDOUT:   call void @_CF.Main.dedae426f8f6526d(ptr %.loc39_5), !dbg !10
// CHECK:STDOUT:   %.loc40_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @_CF.Main.42589ad6dce0869e(ptr %.loc40_5), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.dedae426f8f6526d(ptr %x) !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type @_CA.Main.402deed6b8733082(%type zeroinitializer), !dbg !15
// CHECK:STDOUT:   call void @_CB.Main.dedae426f8f6526d(ptr %x), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.42589ad6dce0869e(ptr %x) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.call = call %type @_CA.Main.402deed6b8733082(%type zeroinitializer), !dbg !19
// CHECK:STDOUT:   call void @_CB.Main.42589ad6dce0869e(ptr %x), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr %type @_CA.Main.402deed6b8733082(%type %x) !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type %x, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.dedae426f8f6526d(ptr %x) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_7.3.temp = alloca i32, align 4, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_7.3.temp), !dbg !25
// CHECK:STDOUT:   %.loc27_5.2 = load i32, ptr %x, align 4, !dbg !26
// CHECK:STDOUT:   %D.call = call i32 @_CD.Main.de631560529e9861(i32 %.loc27_5.2), !dbg !25
// CHECK:STDOUT:   store i32 %D.call, ptr %.loc27_7.3.temp, align 4, !dbg !25
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CB.Main.42589ad6dce0869e(ptr %x) !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_7.3.temp = alloca double, align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_7.3.temp), !dbg !29
// CHECK:STDOUT:   %.loc27_5.2 = load double, ptr %x, align 8, !dbg !30
// CHECK:STDOUT:   %D.call = call double @_CD.Main.3836ccc537683c19(double %.loc27_5.2), !dbg !29
// CHECK:STDOUT:   store double %D.call, ptr %.loc27_7.3.temp, align 8, !dbg !29
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CD.Main.de631560529e9861(i32 %x) !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CD.Main.3836ccc537683c19(double %x) !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret double %x, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @_CA.Main.402deed6b8733082, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_deref_ptr.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 36, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 37, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 39, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 39, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 40, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 40, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 35, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.dedae426f8f6526d", scope: null, file: !3, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 31, column: 3, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 32, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 30, column: 1, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.42589ad6dce0869e", scope: null, file: !3, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 31, column: 3, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 32, column: 3, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 30, column: 1, scope: !18)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "A", linkageName: "_CA.Main.402deed6b8733082", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 3, scope: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.dedae426f8f6526d", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 3, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 5, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 26, column: 1, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "B", linkageName: "_CB.Main.42589ad6dce0869e", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DILocation(line: 27, column: 3, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 27, column: 5, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 26, column: 1, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.de631560529e9861", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !33 = !DILocation(line: 23, column: 3, scope: !32)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "D", linkageName: "_CD.Main.3836ccc537683c19", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !35 = !DILocation(line: 23, column: 3, scope: !34)
