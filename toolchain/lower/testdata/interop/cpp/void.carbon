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
// CHECK:STDOUT:   %.loc7_21.2.temp = alloca ptr, align 8, !dbg !13
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.5c62f2dbad753cd9"(ptr %a), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !13
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !13
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !13
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.5c62f2dbad753cd9"(ptr %self) #0 !dbg !16 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d2075df181ac34c1"(ptr %self), !dbg !22
// CHECK:STDOUT:   ret ptr %1, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d2075df181ac34c1"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.d2075df181ac34c1(ptr %self), !dbg !27
// CHECK:STDOUT:   ret ptr %1, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.d2075df181ac34c1(ptr %value) #0 !dbg !29 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299"(ptr %value), !dbg !32
// CHECK:STDOUT:   ret ptr %1, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !37
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !38
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !39
// CHECK:STDOUT:   ret ptr %2, !dbg !40
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
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassVoidPtr", linkageName: "_CPassVoidPtr.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !11)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{null, !10}
// CHECK:STDOUT: !10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !11 = !{!12}
// CHECK:STDOUT: !12 = !DILocalVariable(arg: 1, scope: !7, type: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 21, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.5c62f2dbad753cd9", scope: null, file: !17, line: 82, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !20)
// CHECK:STDOUT: !17 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{!10, !10}
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !16, type: !10)
// CHECK:STDOUT: !22 = !DILocation(line: 83, column: 12, scope: !16)
// CHECK:STDOUT: !23 = !DILocation(line: 83, column: 5, scope: !16)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d2075df181ac34c1", scope: null, file: !17, line: 68, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !10)
// CHECK:STDOUT: !27 = !DILocation(line: 69, column: 12, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 69, column: 5, scope: !24)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d2075df181ac34c1", scope: null, file: !17, line: 29, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !10)
// CHECK:STDOUT: !32 = !DILocation(line: 30, column: 12, scope: !29)
// CHECK:STDOUT: !33 = !DILocation(line: 30, column: 5, scope: !29)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299", scope: null, file: !17, line: 125, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !10)
// CHECK:STDOUT: !37 = !DILocation(line: 126, column: 14, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 127, column: 5, scope: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 126, column: 18, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 128, column: 5, scope: !34)
