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
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.ecebba3ed00bdf07"(ptr %a), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !13
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !13
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !13
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !16
// CHECK:STDOUT:   call void @"_CSelfDestruct.7348f8acc4e1ca43:core.Destroy.Core"(ptr %.loc7_21.2.temp), !dbg !13
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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.9a022b98a98c20dc:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7348f8acc4e1ca43:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !39
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.6c13586110e95697"(ptr %a), !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !39
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !39
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !39
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !40
// CHECK:STDOUT:   call void @"_CSelfDestruct.7348f8acc4e1ca43:core.Destroy.Core"(ptr %.loc11_21.2.temp), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !50
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.ee2741b107e8d32c"(ptr %a), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !50
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !50
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !50
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !51
// CHECK:STDOUT:   call void @"_CSelfDestruct.66e137fac49b8f32:core.Destroy.Core"(ptr %.loc15_27.2.temp), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e9afb99fe676df82:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8d3ff16138365f48:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8d7d136fb00e7dd7:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.66e137fac49b8f32:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !73
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.ee2741b107e8d32c"(ptr %a), !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !73
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !73
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !73
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !74
// CHECK:STDOUT:   call void @"_CSelfDestruct.66e137fac49b8f32:core.Destroy.Core"(ptr %.loc19_27.2.temp), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.875c15b48d193d60:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.ecebba3ed00bdf07"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.d733b2d6fcf89db9"(ptr %self), !dbg !86
// CHECK:STDOUT:   ret ptr %1, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.6c13586110e95697"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.bf8402781e65ec12:OptionalAs.1dac4564233fecd1.Core.cd60c9c21b2ffda5"(ptr %self), !dbg !94
// CHECK:STDOUT:   ret ptr %1, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.ee2741b107e8d32c"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.bf8402781e65ec12:OptionalAs.1dac4564233fecd1.Core.8315effbdca996ff"(ptr %self), !dbg !102
// CHECK:STDOUT:   ret ptr %1, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.d733b2d6fcf89db9"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %self), !dbg !119
// CHECK:STDOUT:   ret ptr %1, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.bf8402781e65ec12:OptionalAs.1dac4564233fecd1.Core.cd60c9c21b2ffda5"(ptr %self) #0 !dbg !122 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !127
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !127
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !127
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !127
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %1), !dbg !128
// CHECK:STDOUT:   call void @"_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core"(ptr %temp), !dbg !127
// CHECK:STDOUT:   ret ptr %2, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.bf8402781e65ec12:OptionalAs.1dac4564233fecd1.Core.8315effbdca996ff"(ptr %self) #0 !dbg !131 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !136
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !136
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.6d93ae89cb2e7b0f"(ptr %self), !dbg !136
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !136
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !136
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.2a2e44fc411aca14(ptr %2), !dbg !137
// CHECK:STDOUT:   call void @"_CSelfDestruct.e9afb99fe676df82:core.Destroy.Core"(ptr %temp), !dbg !136
// CHECK:STDOUT:   ret ptr %3, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %value) #0 !dbg !149 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %value), !dbg !154
// CHECK:STDOUT:   ret ptr %1, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.6d93ae89cb2e7b0f"(ptr %self) #3 !dbg !158 {
// CHECK:STDOUT:   ret ptr %self, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.2a2e44fc411aca14(ptr %value) #0 !dbg !163 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %value), !dbg !168
// CHECK:STDOUT:   ret ptr %1, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %self) #0 !dbg !176 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !181
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !181
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !181
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !185
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !183
// CHECK:STDOUT:   call void @"_CSelfDestruct.9a022b98a98c20dc:core.Destroy.Core"(ptr %1), !dbg !181
// CHECK:STDOUT:   ret ptr %2, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %self) #0 !dbg !189 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !194
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !194
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !194
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !198
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !196
// CHECK:STDOUT:   call void @"_CSelfDestruct.8d7d136fb00e7dd7:core.Destroy.Core"(ptr %1), !dbg !194
// CHECK:STDOUT:   ret ptr %2, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.ee2741b107e8d32c", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.d733b2d6fcf89db9, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!202, !203, !204, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!209}
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
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.9a022b98a98c20dc:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 7, column: 21, scope: !28)
// CHECK:STDOUT: !31 = !{!29}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7348f8acc4e1ca43:core.Destroy.Core", scope: null, file: !4, line: 7, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 7, column: 21, scope: !32)
// CHECK:STDOUT: !35 = !{!33}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !4, line: 10, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !7)
// CHECK:STDOUT: !39 = !DILocation(line: 11, column: 21, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 11, column: 3, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 10, column: 1, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1feff81df232a20a:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !45 = !DILocation(line: 164, column: 19, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !4, line: 14, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !7)
// CHECK:STDOUT: !50 = !DILocation(line: 15, column: 27, scope: !47)
// CHECK:STDOUT: !51 = !DILocation(line: 15, column: 3, scope: !47)
// CHECK:STDOUT: !52 = !DILocation(line: 14, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e9afb99fe676df82:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !7)
// CHECK:STDOUT: !56 = !DILocation(line: 164, column: 19, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8d3ff16138365f48:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !7)
// CHECK:STDOUT: !60 = !DILocation(line: 15, column: 27, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8d7d136fb00e7dd7:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !7)
// CHECK:STDOUT: !64 = !DILocation(line: 15, column: 27, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.66e137fac49b8f32:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !7)
// CHECK:STDOUT: !68 = !DILocation(line: 15, column: 27, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !7)
// CHECK:STDOUT: !73 = !DILocation(line: 19, column: 27, scope: !70)
// CHECK:STDOUT: !74 = !DILocation(line: 19, column: 3, scope: !70)
// CHECK:STDOUT: !75 = !DILocation(line: 18, column: 1, scope: !70)
// CHECK:STDOUT: !76 = !{!71}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.875c15b48d193d60:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !7)
// CHECK:STDOUT: !79 = !DILocation(line: 164, column: 19, scope: !77)
// CHECK:STDOUT: !80 = !{!78}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.ecebba3ed00bdf07", scope: null, file: !19, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !86 = !DILocation(line: 106, column: 12, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 106, column: 5, scope: !81)
// CHECK:STDOUT: !88 = !{!82}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.6c13586110e95697", scope: null, file: !19, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !7)
// CHECK:STDOUT: !94 = !DILocation(line: 106, column: 12, scope: !89)
// CHECK:STDOUT: !95 = !DILocation(line: 106, column: 5, scope: !89)
// CHECK:STDOUT: !96 = !{!90}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.9a59d9a8bc3e9f77.Core.ee2741b107e8d32c", scope: null, file: !19, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !102 = !DILocation(line: 106, column: 12, scope: !97)
// CHECK:STDOUT: !103 = !DILocation(line: 106, column: 5, scope: !97)
// CHECK:STDOUT: !104 = !{!98}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.d733b2d6fcf89db9", scope: null, file: !19, line: 80, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !121)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 81, column: 12, scope: !113)
// CHECK:STDOUT: !120 = !DILocation(line: 81, column: 5, scope: !113)
// CHECK:STDOUT: !121 = !{!114}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.bf8402781e65ec12:OptionalAs.1dac4564233fecd1.Core.cd60c9c21b2ffda5", scope: null, file: !19, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !130)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !7)
// CHECK:STDOUT: !127 = !DILocation(line: 88, column: 29, scope: !122)
// CHECK:STDOUT: !128 = !DILocation(line: 88, column: 12, scope: !122)
// CHECK:STDOUT: !129 = !DILocation(line: 88, column: 5, scope: !122)
// CHECK:STDOUT: !130 = !{!123}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.bf8402781e65ec12:OptionalAs.1dac4564233fecd1.Core.8315effbdca996ff", scope: null, file: !19, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !139)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !7)
// CHECK:STDOUT: !136 = !DILocation(line: 88, column: 29, scope: !131)
// CHECK:STDOUT: !137 = !DILocation(line: 88, column: 12, scope: !131)
// CHECK:STDOUT: !138 = !DILocation(line: 88, column: 5, scope: !131)
// CHECK:STDOUT: !139 = !{!132}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d733b2d6fcf89db9", scope: null, file: !19, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !7)
// CHECK:STDOUT: !154 = !DILocation(line: 34, column: 12, scope: !149)
// CHECK:STDOUT: !155 = !DILocation(line: 34, column: 5, scope: !149)
// CHECK:STDOUT: !156 = !{!150}
// CHECK:STDOUT: !157 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.6d93ae89cb2e7b0f", scope: null, file: !157, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !162)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !7)
// CHECK:STDOUT: !161 = !DILocation(line: 40, column: 3, scope: !158)
// CHECK:STDOUT: !162 = !{!159}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.2a2e44fc411aca14", scope: null, file: !19, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !170)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !7)
// CHECK:STDOUT: !168 = !DILocation(line: 34, column: 12, scope: !163)
// CHECK:STDOUT: !169 = !DILocation(line: 34, column: 5, scope: !163)
// CHECK:STDOUT: !170 = !{!164}
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271", scope: null, file: !19, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !188)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 1, scope: !176, type: !7)
// CHECK:STDOUT: !181 = !DILocation(line: 167, column: 14, scope: !176)
// CHECK:STDOUT: !183 = !DILocation(line: 167, column: 18, scope: !176)
// CHECK:STDOUT: !185 = !DILocation(line: 169, column: 5, scope: !176)
// CHECK:STDOUT: !187 = !DILocation(line: 170, column: 5, scope: !176)
// CHECK:STDOUT: !188 = !{!177}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137", scope: null, file: !19, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !201)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !7)
// CHECK:STDOUT: !194 = !DILocation(line: 167, column: 14, scope: !189)
// CHECK:STDOUT: !196 = !DILocation(line: 167, column: 18, scope: !189)
// CHECK:STDOUT: !198 = !DILocation(line: 169, column: 5, scope: !189)
// CHECK:STDOUT: !200 = !DILocation(line: 170, column: 5, scope: !189)
// CHECK:STDOUT: !201 = !{!190}
// CHECK:STDOUT: !202 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !203 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !204 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !205 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !206 = !{!"omnipotent char", !205, i64 0}
// CHECK:STDOUT: !207 = !{!"int", !206, i64 0}
// CHECK:STDOUT: !208 = !{!"__libc_errno", !207, i64 0}
// CHECK:STDOUT: !209 = !{!208, !207, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core"(ptr %self) #0 !dbg !214 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !218 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !221
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !222
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z15return_void_ptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!211}
// CHECK:STDOUT: !llvm.module.flags = !{!202, !203, !204, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!209}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !14 = !{!7, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !202 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !203 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !204 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !205 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !206 = !{!"omnipotent char", !205, i64 0}
// CHECK:STDOUT: !207 = !{!"int", !206, i64 0}
// CHECK:STDOUT: !208 = !{!"__libc_errno", !207, i64 0}
// CHECK:STDOUT: !209 = !{!208, !207, i64 0}
// CHECK:STDOUT: !210 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !211 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !210, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !212 = !{!7}
// CHECK:STDOUT: !213 = !DISubroutineType(types: !212)
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.00ac8ef5084ea134:core.Destroy.Core", scope: null, file: !19, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !217)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !7)
// CHECK:STDOUT: !216 = !DILocation(line: 164, column: 19, scope: !214)
// CHECK:STDOUT: !217 = !{!215}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !210, line: 6, type: !213, spFlags: DISPFlagDefinition, unit: !211)
// CHECK:STDOUT: !221 = !DILocation(line: 7, column: 10, scope: !218)
// CHECK:STDOUT: !222 = !DILocation(line: 7, column: 3, scope: !218)
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !210, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !230)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !7)
// CHECK:STDOUT: !229 = !DILocation(line: 11, column: 3, scope: !223)
// CHECK:STDOUT: !230 = !{!224}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !210, line: 14, type: !15, spFlags: DISPFlagDefinition, unit: !211, retainedNodes: !238)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !7)
// CHECK:STDOUT: !237 = !DILocation(line: 15, column: 3, scope: !231)
// CHECK:STDOUT: !238 = !{!232}
// CHECK:STDOUT:
