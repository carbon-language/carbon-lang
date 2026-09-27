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
// CHECK:STDOUT:   call void @"_CSelfDestruct.14b4a412ddf73a96:core.Destroy.Core"(ptr %.loc7_21.2.temp), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a7d1b77eb1279818:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.174120d4d03791b0:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.174120d4d03791b0:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.174120d4d03791b0:core.Destroy.Core"(ptr %self), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.14b4a412ddf73a96:core.Destroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.14b4a412ddf73a96:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.14b4a412ddf73a96:core.Destroy.Core"(ptr %self), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !47
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.6031252d10ea5365"(ptr %a), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !47
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !47
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !47
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.14b4a412ddf73a96:core.Destroy.Core"(ptr %.loc11_21.2.temp), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !58
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.d72303bab6c7803c"(ptr %a), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !58
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !58
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !58
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !59
// CHECK:STDOUT:   call void @"_CSelfDestruct.dfb26e502f641cc5:core.Destroy.Core"(ptr %.loc15_27.2.temp), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e9afb99fe676df82:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8d3ff16138365f48:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4e1f0f3b7725eea2:core.Destroy.Core"(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4e1f0f3b7725eea2:core.Destroy.Core"(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4e1f0f3b7725eea2:core.Destroy.Core"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dfb26e502f641cc5:core.Destroy.Core"(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dfb26e502f641cc5:core.Destroy.Core"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dfb26e502f641cc5:core.Destroy.Core"(ptr %self), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !89
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.d72303bab6c7803c"(ptr %a), !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !89
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !89
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !89
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !90
// CHECK:STDOUT:   call void @"_CSelfDestruct.dfb26e502f641cc5:core.Destroy.Core"(ptr %.loc19_27.2.temp), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7742c7b82efb7924:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0532bac23645330a"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642"(ptr %self), !dbg !102
// CHECK:STDOUT:   ret ptr %1, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.6031252d10ea5365"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.1e16d6a5a3286ade"(ptr %self), !dbg !110
// CHECK:STDOUT:   ret ptr %1, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.d72303bab6c7803c"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.ea6f599e1a77a7ea"(ptr %self), !dbg !118
// CHECK:STDOUT:   ret ptr %1, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642"(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %self), !dbg !135
// CHECK:STDOUT:   ret ptr %1, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.1e16d6a5a3286ade"(ptr %self) #0 !dbg !138 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !143
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !143
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !143
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !143
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %1), !dbg !144
// CHECK:STDOUT:   call void @"_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core"(ptr %temp), !dbg !143
// CHECK:STDOUT:   ret ptr %2, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.ea6f599e1a77a7ea"(ptr %self) #0 !dbg !147 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !152
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !152
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self), !dbg !152
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !152
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !152
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.e4523e01ac2590ba(ptr %2), !dbg !153
// CHECK:STDOUT:   call void @"_CSelfDestruct.e9afb99fe676df82:core.Destroy.Core"(ptr %temp), !dbg !152
// CHECK:STDOUT:   ret ptr %3, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %value) #0 !dbg !165 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %value), !dbg !170
// CHECK:STDOUT:   ret ptr %1, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self) #3 !dbg !174 {
// CHECK:STDOUT:   ret ptr %self, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.e4523e01ac2590ba(ptr %value) #0 !dbg !179 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %value), !dbg !184
// CHECK:STDOUT:   ret ptr %1, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %self) #0 !dbg !192 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !197
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !197
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !197
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !201
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !199
// CHECK:STDOUT:   call void @"_CSelfDestruct.174120d4d03791b0:core.Destroy.Core"(ptr %1), !dbg !197
// CHECK:STDOUT:   ret ptr %2, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %self) #0 !dbg !205 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !210
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !210
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !210
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !214
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !212
// CHECK:STDOUT:   call void @"_CSelfDestruct.4e1f0f3b7725eea2:core.Destroy.Core"(ptr %1), !dbg !210
// CHECK:STDOUT:   ret ptr %2, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.d72303bab6c7803c", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.4201ec9a263c5642, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!218, !219, !220, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!225}
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
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !23)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !20, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 164, column: 19, scope: !20)
// CHECK:STDOUT: !23 = !{!21}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a7d1b77eb1279818:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !27)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 7, column: 21, scope: !24)
// CHECK:STDOUT: !27 = !{!25}
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.174120d4d03791b0:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 7, column: 21, scope: !28)
// CHECK:STDOUT: !31 = !{!29}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.174120d4d03791b0:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 7, column: 21, scope: !32)
// CHECK:STDOUT: !35 = !{!33}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.14b4a412ddf73a96:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !39)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !7)
// CHECK:STDOUT: !38 = !DILocation(line: 7, column: 21, scope: !36)
// CHECK:STDOUT: !39 = !{!37}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.14b4a412ddf73a96:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 7, column: 21, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !4, line: 10, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 11, column: 21, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 11, column: 3, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 10, column: 1, scope: !44)
// CHECK:STDOUT: !50 = !{!45}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !7)
// CHECK:STDOUT: !53 = !DILocation(line: 164, column: 19, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !4, line: 14, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !58 = !DILocation(line: 15, column: 27, scope: !55)
// CHECK:STDOUT: !59 = !DILocation(line: 15, column: 3, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 14, column: 1, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e9afb99fe676df82:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !7)
// CHECK:STDOUT: !64 = !DILocation(line: 164, column: 19, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8d3ff16138365f48:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !7)
// CHECK:STDOUT: !68 = !DILocation(line: 15, column: 27, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4e1f0f3b7725eea2:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !7)
// CHECK:STDOUT: !72 = !DILocation(line: 15, column: 27, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4e1f0f3b7725eea2:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !77)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !7)
// CHECK:STDOUT: !76 = !DILocation(line: 15, column: 27, scope: !74)
// CHECK:STDOUT: !77 = !{!75}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dfb26e502f641cc5:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !7)
// CHECK:STDOUT: !80 = !DILocation(line: 15, column: 27, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dfb26e502f641cc5:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !7)
// CHECK:STDOUT: !84 = !DILocation(line: 15, column: 27, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !92)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !7)
// CHECK:STDOUT: !89 = !DILocation(line: 19, column: 27, scope: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 19, column: 3, scope: !86)
// CHECK:STDOUT: !91 = !DILocation(line: 18, column: 1, scope: !86)
// CHECK:STDOUT: !92 = !{!87}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7742c7b82efb7924:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 164, column: 19, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0532bac23645330a", scope: null, file: !19, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !102 = !DILocation(line: 106, column: 12, scope: !97)
// CHECK:STDOUT: !103 = !DILocation(line: 106, column: 5, scope: !97)
// CHECK:STDOUT: !104 = !{!98}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.6031252d10ea5365", scope: null, file: !19, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !7)
// CHECK:STDOUT: !110 = !DILocation(line: 106, column: 12, scope: !105)
// CHECK:STDOUT: !111 = !DILocation(line: 106, column: 5, scope: !105)
// CHECK:STDOUT: !112 = !{!106}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.d72303bab6c7803c", scope: null, file: !19, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !120)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !118 = !DILocation(line: 106, column: 12, scope: !113)
// CHECK:STDOUT: !119 = !DILocation(line: 106, column: 5, scope: !113)
// CHECK:STDOUT: !120 = !{!114}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642", scope: null, file: !19, line: 80, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !137)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !7)
// CHECK:STDOUT: !135 = !DILocation(line: 81, column: 12, scope: !129)
// CHECK:STDOUT: !136 = !DILocation(line: 81, column: 5, scope: !129)
// CHECK:STDOUT: !137 = !{!130}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.1e16d6a5a3286ade", scope: null, file: !19, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !146)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !7)
// CHECK:STDOUT: !143 = !DILocation(line: 88, column: 29, scope: !138)
// CHECK:STDOUT: !144 = !DILocation(line: 88, column: 12, scope: !138)
// CHECK:STDOUT: !145 = !DILocation(line: 88, column: 5, scope: !138)
// CHECK:STDOUT: !146 = !{!139}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.ea6f599e1a77a7ea", scope: null, file: !19, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !155)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !7)
// CHECK:STDOUT: !152 = !DILocation(line: 88, column: 29, scope: !147)
// CHECK:STDOUT: !153 = !DILocation(line: 88, column: 12, scope: !147)
// CHECK:STDOUT: !154 = !DILocation(line: 88, column: 5, scope: !147)
// CHECK:STDOUT: !155 = !{!148}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.4201ec9a263c5642", scope: null, file: !19, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !172)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !7)
// CHECK:STDOUT: !170 = !DILocation(line: 34, column: 12, scope: !165)
// CHECK:STDOUT: !171 = !DILocation(line: 34, column: 5, scope: !165)
// CHECK:STDOUT: !172 = !{!166}
// CHECK:STDOUT: !173 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba", scope: null, file: !173, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !178)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !7)
// CHECK:STDOUT: !177 = !DILocation(line: 40, column: 3, scope: !174)
// CHECK:STDOUT: !178 = !{!175}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.e4523e01ac2590ba", scope: null, file: !19, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !186)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !7)
// CHECK:STDOUT: !184 = !DILocation(line: 34, column: 12, scope: !179)
// CHECK:STDOUT: !185 = !DILocation(line: 34, column: 5, scope: !179)
// CHECK:STDOUT: !186 = !{!180}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271", scope: null, file: !19, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !204)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !7)
// CHECK:STDOUT: !197 = !DILocation(line: 167, column: 14, scope: !192)
// CHECK:STDOUT: !199 = !DILocation(line: 167, column: 18, scope: !192)
// CHECK:STDOUT: !201 = !DILocation(line: 169, column: 5, scope: !192)
// CHECK:STDOUT: !203 = !DILocation(line: 170, column: 5, scope: !192)
// CHECK:STDOUT: !204 = !{!193}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137", scope: null, file: !19, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !217)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !7)
// CHECK:STDOUT: !210 = !DILocation(line: 167, column: 14, scope: !205)
// CHECK:STDOUT: !212 = !DILocation(line: 167, column: 18, scope: !205)
// CHECK:STDOUT: !214 = !DILocation(line: 169, column: 5, scope: !205)
// CHECK:STDOUT: !216 = !DILocation(line: 170, column: 5, scope: !205)
// CHECK:STDOUT: !217 = !{!206}
// CHECK:STDOUT: !218 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !219 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !220 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !221 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !222 = !{!"omnipotent char", !221, i64 0}
// CHECK:STDOUT: !223 = !{!"int", !222, i64 0}
// CHECK:STDOUT: !224 = !{!"__libc_errno", !223, i64 0}
// CHECK:STDOUT: !225 = !{!224, !223, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core"(ptr %self) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !232
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !234 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !237
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !238
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z15return_void_ptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!227}
// CHECK:STDOUT: !llvm.module.flags = !{!218, !219, !220, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!225}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !14 = !{!7, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !218 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !219 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !220 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !221 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !222 = !{!"omnipotent char", !221, i64 0}
// CHECK:STDOUT: !223 = !{!"int", !222, i64 0}
// CHECK:STDOUT: !224 = !{!"__libc_errno", !223, i64 0}
// CHECK:STDOUT: !225 = !{!224, !223, i64 0}
// CHECK:STDOUT: !226 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !227 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !226, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !228 = !{!7}
// CHECK:STDOUT: !229 = !DISubroutineType(types: !228)
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !227, retainedNodes: !233)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !7)
// CHECK:STDOUT: !232 = !DILocation(line: 164, column: 19, scope: !230)
// CHECK:STDOUT: !233 = !{!231}
// CHECK:STDOUT: !234 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !226, line: 6, type: !229, spFlags: DISPFlagDefinition, unit: !227)
// CHECK:STDOUT: !237 = !DILocation(line: 7, column: 10, scope: !234)
// CHECK:STDOUT: !238 = !DILocation(line: 7, column: 3, scope: !234)
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !226, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !227, retainedNodes: !246)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !7)
// CHECK:STDOUT: !245 = !DILocation(line: 11, column: 3, scope: !239)
// CHECK:STDOUT: !246 = !{!240}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !226, line: 14, type: !15, spFlags: DISPFlagDefinition, unit: !227, retainedNodes: !254)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !7)
// CHECK:STDOUT: !253 = !DILocation(line: 15, column: 3, scope: !247)
// CHECK:STDOUT: !254 = !{!248}
// CHECK:STDOUT:
