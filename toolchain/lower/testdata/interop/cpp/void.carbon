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

// --- void_ptr.h

void take_void_ptr(void *p);
void take_const_void_ptr(const void *p);
void *return_void_ptr();

// --- pass.carbon

library "[[@TEST_NAME]]";

import Cpp library "void_ptr.h";

fn PassVoidPtr(a: Cpp.void*) {
  Cpp.take_void_ptr(a);
}

fn PassIntPtr(a: Cpp.int*) {
  Cpp.take_void_ptr(a);
}

fn PassIntPtrToConstVoidPtr(a: Cpp.int*) {
  Cpp.take_const_void_ptr(a);
}

fn PassConstIntPtrToConstVoidPtr(a: const Cpp.int*) {
  Cpp.take_const_void_ptr(a);
}

// --- receive.carbon

library "[[@TEST_NAME]]";

import Cpp library "void_ptr.h";

fn ReceiveVoidPtr() -> Core.Optional(Cpp.void*) {
  return Cpp.return_void_ptr();
}

fn ConvertFromVoidPtr(p: Cpp.void*) -> i32* {
  return p unsafe as i32*;
}

fn ConvertFromConstVoidPtr(p: const Cpp.void*) -> const i32* {
  return p unsafe as const i32*;
}

// CHECK:STDOUT: ; ModuleID = 'pass.carbon'
// CHECK:STDOUT: source_filename = "pass.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassVoidPtr.Main(ptr %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_21.2.temp = alloca ptr, align 8, !dbg !17
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.66983884551336cb"(ptr %a), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !17
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !23
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.295aa3526801a9d4"(ptr %a), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !23
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !23
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !23
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !29
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e995c7ea1c3d1cc3"(ptr %a), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !29
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !29
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !29
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !35
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e995c7ea1c3d1cc3"(ptr %a), !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !35
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !35
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !35
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.66983884551336cb"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.7bed839a0b822504"(ptr %self), !dbg !44
// CHECK:STDOUT:   ret ptr %1, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.295aa3526801a9d4"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.57bd8d0c2213b1aa"(ptr %self), !dbg !49
// CHECK:STDOUT:   ret ptr %1, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e995c7ea1c3d1cc3"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.02dd0be934e1c38a"(ptr %self), !dbg !54
// CHECK:STDOUT:   ret ptr %1, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.7bed839a0b822504"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.7bed839a0b822504(ptr %self), !dbg !59
// CHECK:STDOUT:   ret ptr %1, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.57bd8d0c2213b1aa"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !64
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !64
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !64
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.7bed839a0b822504(ptr %1), !dbg !65
// CHECK:STDOUT:   ret ptr %2, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.02dd0be934e1c38a"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !70
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk.e8f8f92d3d08d149:ImplicitAs.ffd58aeb29886b72.Core.b88d1103f417c6d4"(ptr %self), !dbg !70
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !70
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !70
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.9544e72a46e24ed6(ptr %2), !dbg !71
// CHECK:STDOUT:   ret ptr %3, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.7bed839a0b822504(ptr %value) #0 !dbg !73 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299"(ptr %value), !dbg !76
// CHECK:STDOUT:   ret ptr %1, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk.e8f8f92d3d08d149:ImplicitAs.ffd58aeb29886b72.Core.b88d1103f417c6d4"(ptr %self) #3 !dbg !78 {
// CHECK:STDOUT:   ret ptr %self, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.9544e72a46e24ed6(ptr %value) #0 !dbg !83 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.2e8a3baa837dcd9f"(ptr %value), !dbg !86
// CHECK:STDOUT:   ret ptr %1, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299"(ptr %self) #0 !dbg !88 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !91
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !91
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !92
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !93
// CHECK:STDOUT:   ret ptr %2, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.2e8a3baa837dcd9f"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !98
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !98
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !99
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !100
// CHECK:STDOUT:   ret ptr %2, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e995c7ea1c3d1cc3", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.7bed839a0b822504, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "pass.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassVoidPtr", linkageName: "_CPassVoidPtr.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 21, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !6, line: 10, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !21)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !20, type: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 21, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 3, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 10, column: 1, scope: !20)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !6, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 15, column: 27, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 3, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 1, scope: !26)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !6, line: 18, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !14)
// CHECK:STDOUT: !35 = !DILocation(line: 19, column: 27, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 19, column: 3, scope: !32)
// CHECK:STDOUT: !37 = !DILocation(line: 18, column: 1, scope: !32)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.66983884551336cb", scope: null, file: !39, line: 93, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !39 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!14, !14}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !38, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 94, column: 12, scope: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 94, column: 5, scope: !38)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.295aa3526801a9d4", scope: null, file: !39, line: 93, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
// CHECK:STDOUT: !47 = !{!48}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !46, type: !14)
// CHECK:STDOUT: !49 = !DILocation(line: 94, column: 12, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 94, column: 5, scope: !46)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e995c7ea1c3d1cc3", scope: null, file: !39, line: 93, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !52)
// CHECK:STDOUT: !52 = !{!53}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !51, type: !14)
// CHECK:STDOUT: !54 = !DILocation(line: 94, column: 12, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 94, column: 5, scope: !51)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.7bed839a0b822504", scope: null, file: !39, line: 68, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !57 = !{!58}
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !56, type: !14)
// CHECK:STDOUT: !59 = !DILocation(line: 69, column: 12, scope: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 69, column: 5, scope: !56)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.57bd8d0c2213b1aa", scope: null, file: !39, line: 75, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !14)
// CHECK:STDOUT: !64 = !DILocation(line: 76, column: 29, scope: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 76, column: 12, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 76, column: 5, scope: !61)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.02dd0be934e1c38a", scope: null, file: !39, line: 75, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !14)
// CHECK:STDOUT: !70 = !DILocation(line: 76, column: 29, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 76, column: 12, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 76, column: 5, scope: !67)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.7bed839a0b822504", scope: null, file: !39, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !14)
// CHECK:STDOUT: !76 = !DILocation(line: 30, column: 12, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 30, column: 5, scope: !73)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk.e8f8f92d3d08d149:ImplicitAs.ffd58aeb29886b72.Core.b88d1103f417c6d4", scope: null, file: !79, line: 40, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !79 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !80 = !{!81}
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !78, type: !14)
// CHECK:STDOUT: !82 = !DILocation(line: 40, column: 3, scope: !78)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.9544e72a46e24ed6", scope: null, file: !39, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !84 = !{!85}
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !83, type: !14)
// CHECK:STDOUT: !86 = !DILocation(line: 30, column: 12, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 30, column: 5, scope: !83)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299", scope: null, file: !39, line: 138, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !14)
// CHECK:STDOUT: !91 = !DILocation(line: 139, column: 14, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 140, column: 5, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 139, column: 18, scope: !88)
// CHECK:STDOUT: !94 = !DILocation(line: 141, column: 5, scope: !88)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.2e8a3baa837dcd9f", scope: null, file: !39, line: 138, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !95, type: !14)
// CHECK:STDOUT: !98 = !DILocation(line: 139, column: 14, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 140, column: 5, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 139, column: 18, scope: !95)
// CHECK:STDOUT: !101 = !DILocation(line: 141, column: 5, scope: !95)
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !15
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z15return_void_ptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 10, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !6, line: 10, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{!14, !14}
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !17, type: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 11, column: 3, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !6, line: 14, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !14)
// CHECK:STDOUT: !26 = !DILocation(line: 15, column: 3, scope: !23)
