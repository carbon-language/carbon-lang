// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/void.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/void.carbon

// --- take_void_ptr.h

void take_void_ptr(void *p);

// --- call.carbon

library "[[@TEST_NAME]]";

import Cpp library "take_void_ptr.h";

fn PassVoidPtr(a: Cpp.void*) {
  Cpp.take_void_ptr(a);
}

// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassVoidPtr.Main(ptr %a) #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_21.2.temp = alloca ptr, align 8, !dbg !10
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.f82b7327366b1770:ImplicitAs.Core.1bbd3eba7f2908d7"(ptr %a), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !10
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !10
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !10
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.f82b7327366b1770:ImplicitAs.Core.1bbd3eba7f2908d7"(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.b95abfcde220df97"(ptr %self), !dbg !15
// CHECK:STDOUT:   ret ptr %1, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.b95abfcde220df97"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.b95abfcde220df97(ptr %self), !dbg !18
// CHECK:STDOUT:   ret ptr %1, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.b95abfcde220df97(ptr %value) #0 !dbg !20 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.4f0b5cc38af595d2:OptionalStorage.Core.a3238f3d1f6ba299"(ptr %value), !dbg !21
// CHECK:STDOUT:   ret ptr %1, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.4f0b5cc38af595d2:OptionalStorage.Core.a3238f3d1f6ba299"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !24
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !25
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !26
// CHECK:STDOUT:   ret ptr %2, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassVoidPtr", linkageName: "_CPassVoidPtr.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 21, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.f82b7327366b1770:ImplicitAs.Core.1bbd3eba7f2908d7", scope: null, file: !14, line: 82, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !14 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !15 = !DILocation(line: 83, column: 12, scope: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 83, column: 5, scope: !13)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.b95abfcde220df97", scope: null, file: !14, line: 68, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 69, column: 12, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 69, column: 5, scope: !17)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.b95abfcde220df97", scope: null, file: !14, line: 29, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !21 = !DILocation(line: 30, column: 12, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 30, column: 5, scope: !20)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.4f0b5cc38af595d2:OptionalStorage.Core.a3238f3d1f6ba299", scope: null, file: !14, line: 125, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !24 = !DILocation(line: 126, column: 14, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 127, column: 5, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 126, column: 18, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 128, column: 5, scope: !23)
