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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pass.carbon'
// CHECK:STDOUT: source_filename = "pass.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassVoidPtr.Main(ptr %a) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_21.2.temp = alloca ptr, align 8, !dbg !13
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0532bac23645330a"(ptr %a), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !13
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !13
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !13
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5ba24784f2f66cc5(ptr %.loc7_21.2.temp), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a7d1b77eb1279818:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1ac0350a118ac030:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.14b4a412ddf73a96:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !34
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.d9d27b4a65f5a4ac"(ptr %a), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !34
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !34
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !34
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5ba24784f2f66cc5(ptr %.loc11_21.2.temp), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !41
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.78eec894ae91c250"(ptr %a), !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !41
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !41
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !41
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.4debc401c3faa65c(ptr %.loc15_27.2.temp), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8d3ff16138365f48:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0c31d4c1cfd98792:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dfb26e502f641cc5:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !60
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.78eec894ae91c250"(ptr %a), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !60
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !60
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !60
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !61
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.4debc401c3faa65c(ptr %.loc19_27.2.temp), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0532bac23645330a"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret ptr %1, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5ba24784f2f66cc5(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.14b4a412ddf73a96:core.Destroy.Core"(ptr %self), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.d9d27b4a65f5a4ac"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.e64c9fdc52b433db"(ptr %self), !dbg !86
// CHECK:STDOUT:   ret ptr %1, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.78eec894ae91c250"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.5ca05e01af772fb6"(ptr %self), !dbg !94
// CHECK:STDOUT:   ret ptr %1, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.4debc401c3faa65c(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dfb26e502f641cc5:core.Destroy.Core"(ptr %self), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642"(ptr %self) #0 !dbg !112 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %self), !dbg !118
// CHECK:STDOUT:   ret ptr %1, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.e64c9fdc52b433db"(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !126
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !126
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !126
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !126
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %1), !dbg !127
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a5fa084ac3ef9b63(ptr %temp), !dbg !126
// CHECK:STDOUT:   ret ptr %2, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.5ca05e01af772fb6"(ptr %self) #0 !dbg !130 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !135
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self), !dbg !135
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !135
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !135
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.e4523e01ac2590ba(ptr %2), !dbg !136
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a254950d58f0acf2(ptr %temp), !dbg !135
// CHECK:STDOUT:   ret ptr %3, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %value) #0 !dbg !148 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %value), !dbg !153
// CHECK:STDOUT:   ret ptr %1, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.a5fa084ac3ef9b63(ptr %self) #0 !dbg !156 {
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self) #3 !dbg !164 {
// CHECK:STDOUT:   ret ptr %self, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.e4523e01ac2590ba(ptr %value) #0 !dbg !169 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %value), !dbg !174
// CHECK:STDOUT:   ret ptr %1, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.a254950d58f0acf2(ptr %self) #0 !dbg !177 {
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %self) #0 !dbg !189 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !194
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !194
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !194
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !198
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !196
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.de1f6e9eabe877e6(ptr %1), !dbg !194
// CHECK:STDOUT:   ret ptr %2, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %self) #0 !dbg !202 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !207
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !207
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !207
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !211
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !209
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.97f649105ffcf130(ptr %1), !dbg !207
// CHECK:STDOUT:   ret ptr %2, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.de1f6e9eabe877e6(ptr %self) #0 !dbg !215 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1ac0350a118ac030:core.Destroy.Core"(ptr %self), !dbg !219
// CHECK:STDOUT:   ret void, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.97f649105ffcf130(ptr %self) #0 !dbg !222 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0c31d4c1cfd98792:core.Destroy.Core"(ptr %self), !dbg !226
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.5ba24784f2f66cc5, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.78eec894ae91c250", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.4debc401c3faa65c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.4201ec9a263c5642, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!229, !230, !231, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!236}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "pass.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "PassVoidPtr", linkageName: "_CPassVoidPtr.Main", scope: null, file: !4, line: 6, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !18)
// CHECK:STDOUT: !11 = !DILocalVariable(arg: 1, scope: !10, type: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 21, scope: !10)
// CHECK:STDOUT: !14 = !{!7, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 3, scope: !10)
// CHECK:STDOUT: !17 = !DILocation(line: 6, column: 1, scope: !10)
// CHECK:STDOUT: !18 = !{!11}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a7d1b77eb1279818:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 7, column: 21, scope: !19)
// CHECK:STDOUT: !22 = !{!20}
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1ac0350a118ac030:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !25 = !DILocation(line: 7, column: 21, scope: !23)
// CHECK:STDOUT: !26 = !{!24}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.14b4a412ddf73a96:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !7)
// CHECK:STDOUT: !29 = !DILocation(line: 7, column: 21, scope: !27)
// CHECK:STDOUT: !30 = !{!28}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !4, line: 10, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 11, column: 21, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 11, column: 3, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 10, column: 1, scope: !31)
// CHECK:STDOUT: !37 = !{!32}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !4, line: 14, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !44)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 15, column: 27, scope: !38)
// CHECK:STDOUT: !42 = !DILocation(line: 15, column: 3, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 1, scope: !38)
// CHECK:STDOUT: !44 = !{!39}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8d3ff16138365f48:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 15, column: 27, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0c31d4c1cfd98792:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !51 = !DILocation(line: 15, column: 27, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dfb26e502f641cc5:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !7)
// CHECK:STDOUT: !55 = !DILocation(line: 15, column: 27, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !7)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 27, scope: !57)
// CHECK:STDOUT: !61 = !DILocation(line: 19, column: 3, scope: !57)
// CHECK:STDOUT: !62 = !DILocation(line: 18, column: 1, scope: !57)
// CHECK:STDOUT: !63 = !{!58}
// CHECK:STDOUT: !64 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0532bac23645330a", scope: null, file: !64, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !72)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !7)
// CHECK:STDOUT: !70 = !DILocation(line: 106, column: 12, scope: !65)
// CHECK:STDOUT: !71 = !DILocation(line: 106, column: 5, scope: !65)
// CHECK:STDOUT: !72 = !{!66}
// CHECK:STDOUT: !73 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5ba24784f2f66cc5", scope: null, file: !73, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !7)
// CHECK:STDOUT: !78 = !DILocation(line: 28, column: 5, scope: !74)
// CHECK:STDOUT: !79 = !DILocation(line: 26, column: 3, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.d9d27b4a65f5a4ac", scope: null, file: !64, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !86 = !DILocation(line: 106, column: 12, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 106, column: 5, scope: !81)
// CHECK:STDOUT: !88 = !{!82}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.78eec894ae91c250", scope: null, file: !64, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !7)
// CHECK:STDOUT: !94 = !DILocation(line: 106, column: 12, scope: !89)
// CHECK:STDOUT: !95 = !DILocation(line: 106, column: 5, scope: !89)
// CHECK:STDOUT: !96 = !{!90}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.4debc401c3faa65c", scope: null, file: !73, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !103)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !101 = !DILocation(line: 28, column: 5, scope: !97)
// CHECK:STDOUT: !102 = !DILocation(line: 26, column: 3, scope: !97)
// CHECK:STDOUT: !103 = !{!98}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642", scope: null, file: !64, line: 80, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !120)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !7)
// CHECK:STDOUT: !118 = !DILocation(line: 81, column: 12, scope: !112)
// CHECK:STDOUT: !119 = !DILocation(line: 81, column: 5, scope: !112)
// CHECK:STDOUT: !120 = !{!113}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.e64c9fdc52b433db", scope: null, file: !64, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !129)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !7)
// CHECK:STDOUT: !126 = !DILocation(line: 88, column: 29, scope: !121)
// CHECK:STDOUT: !127 = !DILocation(line: 88, column: 12, scope: !121)
// CHECK:STDOUT: !128 = !DILocation(line: 88, column: 5, scope: !121)
// CHECK:STDOUT: !129 = !{!122}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.5ca05e01af772fb6", scope: null, file: !64, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !138)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !7)
// CHECK:STDOUT: !135 = !DILocation(line: 88, column: 29, scope: !130)
// CHECK:STDOUT: !136 = !DILocation(line: 88, column: 12, scope: !130)
// CHECK:STDOUT: !137 = !DILocation(line: 88, column: 5, scope: !130)
// CHECK:STDOUT: !138 = !{!131}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.4201ec9a263c5642", scope: null, file: !64, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !155)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !7)
// CHECK:STDOUT: !153 = !DILocation(line: 34, column: 12, scope: !148)
// CHECK:STDOUT: !154 = !DILocation(line: 34, column: 5, scope: !148)
// CHECK:STDOUT: !155 = !{!149}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.a5fa084ac3ef9b63", scope: null, file: !73, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !162)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !7)
// CHECK:STDOUT: !161 = !DILocation(line: 26, column: 3, scope: !156)
// CHECK:STDOUT: !162 = !{!157}
// CHECK:STDOUT: !163 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba", scope: null, file: !163, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !168)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !7)
// CHECK:STDOUT: !167 = !DILocation(line: 40, column: 3, scope: !164)
// CHECK:STDOUT: !168 = !{!165}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.e4523e01ac2590ba", scope: null, file: !64, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !176)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !7)
// CHECK:STDOUT: !174 = !DILocation(line: 34, column: 12, scope: !169)
// CHECK:STDOUT: !175 = !DILocation(line: 34, column: 5, scope: !169)
// CHECK:STDOUT: !176 = !{!170}
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.a254950d58f0acf2", scope: null, file: !73, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !183)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !177, type: !7)
// CHECK:STDOUT: !182 = !DILocation(line: 26, column: 3, scope: !177)
// CHECK:STDOUT: !183 = !{!178}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271", scope: null, file: !64, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !201)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !7)
// CHECK:STDOUT: !194 = !DILocation(line: 167, column: 14, scope: !189)
// CHECK:STDOUT: !196 = !DILocation(line: 167, column: 18, scope: !189)
// CHECK:STDOUT: !198 = !DILocation(line: 169, column: 5, scope: !189)
// CHECK:STDOUT: !200 = !DILocation(line: 170, column: 5, scope: !189)
// CHECK:STDOUT: !201 = !{!190}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137", scope: null, file: !64, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !214)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !202, type: !7)
// CHECK:STDOUT: !207 = !DILocation(line: 167, column: 14, scope: !202)
// CHECK:STDOUT: !209 = !DILocation(line: 167, column: 18, scope: !202)
// CHECK:STDOUT: !211 = !DILocation(line: 169, column: 5, scope: !202)
// CHECK:STDOUT: !213 = !DILocation(line: 170, column: 5, scope: !202)
// CHECK:STDOUT: !214 = !{!203}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.de1f6e9eabe877e6", scope: null, file: !73, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !221)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !7)
// CHECK:STDOUT: !219 = !DILocation(line: 28, column: 5, scope: !215)
// CHECK:STDOUT: !220 = !DILocation(line: 26, column: 3, scope: !215)
// CHECK:STDOUT: !221 = !{!216}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.97f649105ffcf130", scope: null, file: !73, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !228)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !7)
// CHECK:STDOUT: !226 = !DILocation(line: 28, column: 5, scope: !222)
// CHECK:STDOUT: !227 = !DILocation(line: 26, column: 3, scope: !222)
// CHECK:STDOUT: !228 = !{!223}
// CHECK:STDOUT: !229 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !230 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !231 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !232 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !233 = !{!"omnipotent char", !232, i64 0}
// CHECK:STDOUT: !234 = !{!"int", !233, i64 0}
// CHECK:STDOUT: !235 = !{!"__libc_errno", !234, i64 0}
// CHECK:STDOUT: !236 = !{!235, !234, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !241 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !244
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z15return_void_ptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !246 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !252
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !254 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!238}
// CHECK:STDOUT: !llvm.module.flags = !{!229, !230, !231, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!236}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !14 = !{!7, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !229 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !230 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !231 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !232 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !233 = !{!"omnipotent char", !232, i64 0}
// CHECK:STDOUT: !234 = !{!"int", !233, i64 0}
// CHECK:STDOUT: !235 = !{!"__libc_errno", !234, i64 0}
// CHECK:STDOUT: !236 = !{!235, !234, i64 0}
// CHECK:STDOUT: !237 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !238 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !237, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !239 = !{!7}
// CHECK:STDOUT: !240 = !DISubroutineType(types: !239)
// CHECK:STDOUT: !241 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !237, line: 6, type: !240, spFlags: DISPFlagDefinition, unit: !238)
// CHECK:STDOUT: !244 = !DILocation(line: 7, column: 10, scope: !241)
// CHECK:STDOUT: !245 = !DILocation(line: 7, column: 3, scope: !241)
// CHECK:STDOUT: !246 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !237, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !238, retainedNodes: !253)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 1, scope: !246, type: !7)
// CHECK:STDOUT: !252 = !DILocation(line: 11, column: 3, scope: !246)
// CHECK:STDOUT: !253 = !{!247}
// CHECK:STDOUT: !254 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !237, line: 14, type: !15, spFlags: DISPFlagDefinition, unit: !238, retainedNodes: !261)
// CHECK:STDOUT: !255 = !DILocalVariable(arg: 1, scope: !254, type: !7)
// CHECK:STDOUT: !260 = !DILocation(line: 15, column: 3, scope: !254)
// CHECK:STDOUT: !261 = !{!255}
// CHECK:STDOUT:
