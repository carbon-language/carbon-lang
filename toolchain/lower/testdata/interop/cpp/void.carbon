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
// CHECK:STDOUT:   call void @"_COp.ab1af2e999d1dfe1:core.Destroy.Core"(ptr %.loc7_21.2.temp), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2bb3ddafbe2fe5d2:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.24996695b716f57d:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ab1af2e999d1dfe1:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !34
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9db51af7092ba247"(ptr %a), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !34
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !34
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !34
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !35
// CHECK:STDOUT:   call void @"_COp.ab1af2e999d1dfe1:core.Destroy.Core"(ptr %.loc11_21.2.temp), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !41
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.218005b34e691dcb"(ptr %a), !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !41
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !41
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !41
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !42
// CHECK:STDOUT:   call void @"_COp.05b8ef735d2db490:core.Destroy.Core"(ptr %.loc15_27.2.temp), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1b9dec9ba7b5442e:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0ca0de5f18f7d6e9:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.05b8ef735d2db490:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !60
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.218005b34e691dcb"(ptr %a), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !60
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !60
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !60
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !61
// CHECK:STDOUT:   call void @"_COp.05b8ef735d2db490:core.Destroy.Core"(ptr %.loc19_27.2.temp), !dbg !60
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
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9db51af7092ba247"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.a6387af2c84c1f91"(ptr %self), !dbg !78
// CHECK:STDOUT:   ret ptr %1, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.218005b34e691dcb"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.09ed4effa59a8188"(ptr %self), !dbg !86
// CHECK:STDOUT:   ret ptr %1, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %self), !dbg !103
// CHECK:STDOUT:   ret ptr %1, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.a6387af2c84c1f91"(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !111
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !111
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !111
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %1), !dbg !112
// CHECK:STDOUT:   ret ptr %2, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.09ed4effa59a8188"(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !120
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !120
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self), !dbg !120
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !120
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !120
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.e4523e01ac2590ba(ptr %2), !dbg !121
// CHECK:STDOUT:   ret ptr %3, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %value) #0 !dbg !133 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %value), !dbg !138
// CHECK:STDOUT:   ret ptr %1, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self) #3 !dbg !142 {
// CHECK:STDOUT:   ret ptr %self, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.e4523e01ac2590ba(ptr %value) #0 !dbg !147 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %value), !dbg !152
// CHECK:STDOUT:   ret ptr %1, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !165
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !165
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !165
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !169
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !167
// CHECK:STDOUT:   call void @"_COp.24996695b716f57d:core.Destroy.Core"(ptr %1), !dbg !165
// CHECK:STDOUT:   ret ptr %2, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %self) #0 !dbg !173 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !178
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !178
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !182
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !180
// CHECK:STDOUT:   call void @"_COp.0ca0de5f18f7d6e9:core.Destroy.Core"(ptr %1), !dbg !178
// CHECK:STDOUT:   ret ptr %2, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.218005b34e691dcb", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.4201ec9a263c5642, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!186, !187, !188, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!193}
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
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2bb3ddafbe2fe5d2:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 7, column: 21, scope: !19)
// CHECK:STDOUT: !22 = !{!20}
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.24996695b716f57d:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !25 = !DILocation(line: 7, column: 21, scope: !23)
// CHECK:STDOUT: !26 = !{!24}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ab1af2e999d1dfe1:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
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
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1b9dec9ba7b5442e:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 15, column: 27, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0ca0de5f18f7d6e9:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !51 = !DILocation(line: 15, column: 27, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.05b8ef735d2db490:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
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
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9db51af7092ba247", scope: null, file: !64, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !78 = !DILocation(line: 106, column: 12, scope: !73)
// CHECK:STDOUT: !79 = !DILocation(line: 106, column: 5, scope: !73)
// CHECK:STDOUT: !80 = !{!74}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.218005b34e691dcb", scope: null, file: !64, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !86 = !DILocation(line: 106, column: 12, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 106, column: 5, scope: !81)
// CHECK:STDOUT: !88 = !{!82}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642", scope: null, file: !64, line: 80, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !105)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !103 = !DILocation(line: 81, column: 12, scope: !97)
// CHECK:STDOUT: !104 = !DILocation(line: 81, column: 5, scope: !97)
// CHECK:STDOUT: !105 = !{!98}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.a6387af2c84c1f91", scope: null, file: !64, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !114)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !7)
// CHECK:STDOUT: !111 = !DILocation(line: 88, column: 29, scope: !106)
// CHECK:STDOUT: !112 = !DILocation(line: 88, column: 12, scope: !106)
// CHECK:STDOUT: !113 = !DILocation(line: 88, column: 5, scope: !106)
// CHECK:STDOUT: !114 = !{!107}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.09ed4effa59a8188", scope: null, file: !64, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !123)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !7)
// CHECK:STDOUT: !120 = !DILocation(line: 88, column: 29, scope: !115)
// CHECK:STDOUT: !121 = !DILocation(line: 88, column: 12, scope: !115)
// CHECK:STDOUT: !122 = !DILocation(line: 88, column: 5, scope: !115)
// CHECK:STDOUT: !123 = !{!116}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.4201ec9a263c5642", scope: null, file: !64, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !7)
// CHECK:STDOUT: !138 = !DILocation(line: 34, column: 12, scope: !133)
// CHECK:STDOUT: !139 = !DILocation(line: 34, column: 5, scope: !133)
// CHECK:STDOUT: !140 = !{!134}
// CHECK:STDOUT: !141 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba", scope: null, file: !141, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !146)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !7)
// CHECK:STDOUT: !145 = !DILocation(line: 40, column: 3, scope: !142)
// CHECK:STDOUT: !146 = !{!143}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.e4523e01ac2590ba", scope: null, file: !64, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !154)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !7)
// CHECK:STDOUT: !152 = !DILocation(line: 34, column: 12, scope: !147)
// CHECK:STDOUT: !153 = !DILocation(line: 34, column: 5, scope: !147)
// CHECK:STDOUT: !154 = !{!148}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271", scope: null, file: !64, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !172)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !7)
// CHECK:STDOUT: !165 = !DILocation(line: 167, column: 14, scope: !160)
// CHECK:STDOUT: !167 = !DILocation(line: 167, column: 18, scope: !160)
// CHECK:STDOUT: !169 = !DILocation(line: 169, column: 5, scope: !160)
// CHECK:STDOUT: !171 = !DILocation(line: 170, column: 5, scope: !160)
// CHECK:STDOUT: !172 = !{!161}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137", scope: null, file: !64, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !185)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !7)
// CHECK:STDOUT: !178 = !DILocation(line: 167, column: 14, scope: !173)
// CHECK:STDOUT: !180 = !DILocation(line: 167, column: 18, scope: !173)
// CHECK:STDOUT: !182 = !DILocation(line: 169, column: 5, scope: !173)
// CHECK:STDOUT: !184 = !DILocation(line: 170, column: 5, scope: !173)
// CHECK:STDOUT: !185 = !{!174}
// CHECK:STDOUT: !186 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !187 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !188 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !189 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !190 = !{!"omnipotent char", !189, i64 0}
// CHECK:STDOUT: !191 = !{!"int", !190, i64 0}
// CHECK:STDOUT: !192 = !{!"__libc_errno", !191, i64 0}
// CHECK:STDOUT: !193 = !{!192, !191, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !201
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z15return_void_ptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!195}
// CHECK:STDOUT: !llvm.module.flags = !{!186, !187, !188, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!193}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !14 = !{!7, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !186 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !187 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !188 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !189 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !190 = !{!"omnipotent char", !189, i64 0}
// CHECK:STDOUT: !191 = !{!"int", !190, i64 0}
// CHECK:STDOUT: !192 = !{!"__libc_errno", !191, i64 0}
// CHECK:STDOUT: !193 = !{!192, !191, i64 0}
// CHECK:STDOUT: !194 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !195 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !194, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !196 = !{!7}
// CHECK:STDOUT: !197 = !DISubroutineType(types: !196)
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !194, line: 6, type: !197, spFlags: DISPFlagDefinition, unit: !195)
// CHECK:STDOUT: !201 = !DILocation(line: 7, column: 10, scope: !198)
// CHECK:STDOUT: !202 = !DILocation(line: 7, column: 3, scope: !198)
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !194, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !195, retainedNodes: !210)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !7)
// CHECK:STDOUT: !209 = !DILocation(line: 11, column: 3, scope: !203)
// CHECK:STDOUT: !210 = !{!204}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !194, line: 14, type: !15, spFlags: DISPFlagDefinition, unit: !195, retainedNodes: !218)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !7)
// CHECK:STDOUT: !217 = !DILocation(line: 15, column: 3, scope: !211)
// CHECK:STDOUT: !218 = !{!212}
// CHECK:STDOUT:
