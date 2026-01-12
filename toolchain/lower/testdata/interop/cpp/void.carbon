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
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !19
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.63e244d930b21b2a"(ptr %a), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !19
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !19
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !19
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !25
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.3fcbaa3d70c86d44"(ptr %a), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !25
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !25
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !25
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !31
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.3fcbaa3d70c86d44"(ptr %a), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !31
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !31
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !31
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.5c62f2dbad753cd9"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d2075df181ac34c1"(ptr %self), !dbg !40
// CHECK:STDOUT:   ret ptr %1, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.63e244d930b21b2a"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.599fcf6815ccd825"(ptr %self), !dbg !45
// CHECK:STDOUT:   ret ptr %1, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.3fcbaa3d70c86d44"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.40332936f52138f0"(ptr %self), !dbg !50
// CHECK:STDOUT:   ret ptr %1, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d2075df181ac34c1"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.d2075df181ac34c1(ptr %self), !dbg !55
// CHECK:STDOUT:   ret ptr %1, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.599fcf6815ccd825"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !60
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !60
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !60
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.d2075df181ac34c1(ptr %1), !dbg !61
// CHECK:STDOUT:   ret ptr %2, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.40332936f52138f0"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !66
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk.e8f8f92d3d08d149:ImplicitAs.ffd58aeb29886b72.Core.b88d1103f417c6d4"(ptr %self), !dbg !66
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !66
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !66
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.d2075df181ac34c1(ptr %2), !dbg !67
// CHECK:STDOUT:   ret ptr %3, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.d2075df181ac34c1(ptr %value) #0 !dbg !69 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299"(ptr %value), !dbg !72
// CHECK:STDOUT:   ret ptr %1, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk.e8f8f92d3d08d149:ImplicitAs.ffd58aeb29886b72.Core.b88d1103f417c6d4"(ptr %self) #2 !dbg !74 {
// CHECK:STDOUT:   ret ptr %self, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !82
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !83
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !84
// CHECK:STDOUT:   ret ptr %2, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 6, 5, 4, 3 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.3fcbaa3d70c86d44", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.d2075df181ac34c1, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "pass.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassVoidPtr", linkageName: "_CPassVoidPtr.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !11)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{null, !10}
// CHECK:STDOUT: !10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !11 = !{!12}
// CHECK:STDOUT: !12 = !DILocalVariable(arg: 1, scope: !7, type: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 21, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !6, line: 10, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !17)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !16, type: !10)
// CHECK:STDOUT: !19 = !DILocation(line: 11, column: 21, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 11, column: 3, scope: !16)
// CHECK:STDOUT: !21 = !DILocation(line: 10, column: 1, scope: !16)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !6, line: 14, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !10)
// CHECK:STDOUT: !25 = !DILocation(line: 15, column: 27, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 15, column: 3, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 14, column: 1, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !6, line: 18, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !10)
// CHECK:STDOUT: !31 = !DILocation(line: 19, column: 27, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 3, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 1, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.5c62f2dbad753cd9", scope: null, file: !35, line: 93, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !35 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !36 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !37 = !{!10, !10}
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !34, type: !10)
// CHECK:STDOUT: !40 = !DILocation(line: 94, column: 12, scope: !34)
// CHECK:STDOUT: !41 = !DILocation(line: 94, column: 5, scope: !34)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.63e244d930b21b2a", scope: null, file: !35, line: 93, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !10)
// CHECK:STDOUT: !45 = !DILocation(line: 94, column: 12, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 94, column: 5, scope: !42)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.3fcbaa3d70c86d44", scope: null, file: !35, line: 93, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !48)
// CHECK:STDOUT: !48 = !{!49}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !47, type: !10)
// CHECK:STDOUT: !50 = !DILocation(line: 94, column: 12, scope: !47)
// CHECK:STDOUT: !51 = !DILocation(line: 94, column: 5, scope: !47)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d2075df181ac34c1", scope: null, file: !35, line: 68, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !53 = !{!54}
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !52, type: !10)
// CHECK:STDOUT: !55 = !DILocation(line: 69, column: 12, scope: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 69, column: 5, scope: !52)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.599fcf6815ccd825", scope: null, file: !35, line: 75, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !10)
// CHECK:STDOUT: !60 = !DILocation(line: 76, column: 29, scope: !57)
// CHECK:STDOUT: !61 = !DILocation(line: 76, column: 12, scope: !57)
// CHECK:STDOUT: !62 = !DILocation(line: 76, column: 5, scope: !57)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.40332936f52138f0", scope: null, file: !35, line: 75, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !64 = !{!65}
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !63, type: !10)
// CHECK:STDOUT: !66 = !DILocation(line: 76, column: 29, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 76, column: 12, scope: !63)
// CHECK:STDOUT: !68 = !DILocation(line: 76, column: 5, scope: !63)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d2075df181ac34c1", scope: null, file: !35, line: 29, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !10)
// CHECK:STDOUT: !72 = !DILocation(line: 30, column: 12, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 30, column: 5, scope: !69)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk.e8f8f92d3d08d149:ImplicitAs.ffd58aeb29886b72.Core.b88d1103f417c6d4", scope: null, file: !75, line: 40, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !75 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !76 = !{!77}
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !74, type: !10)
// CHECK:STDOUT: !78 = !DILocation(line: 40, column: 3, scope: !74)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.a3238f3d1f6ba299", scope: null, file: !35, line: 138, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !80 = !{!81}
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !79, type: !10)
// CHECK:STDOUT: !82 = !DILocation(line: 139, column: 14, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 140, column: 5, scope: !79)
// CHECK:STDOUT: !84 = !DILocation(line: 139, column: 18, scope: !79)
// CHECK:STDOUT: !85 = !DILocation(line: 141, column: 5, scope: !79)
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !11
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z15return_void_ptrv()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 10, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !6, line: 10, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !16)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!10, !10}
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !13, type: !10)
// CHECK:STDOUT: !18 = !DILocation(line: 11, column: 3, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !6, line: 14, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !10)
// CHECK:STDOUT: !22 = !DILocation(line: 15, column: 3, scope: !19)
