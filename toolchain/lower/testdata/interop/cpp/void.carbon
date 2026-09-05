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
// CHECK:STDOUT: define void @_CPassVoidPtr.Main(ptr %a) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_21.2.temp = alloca ptr, align 8, !dbg !18
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0532bac23645330a"(ptr %a), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !18
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !18
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !18
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !19
// CHECK:STDOUT:   call void @"_COp.14b4a412ddf73a96:core.Destroy.Core"(ptr %.loc7_21.2.temp), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a7d1b77eb1279818:core.Destroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.39995c0257abf9d3:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.14b4a412ddf73a96:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !36
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9db51af7092ba247"(ptr %a), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !36
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !36
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !36
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !37
// CHECK:STDOUT:   call void @"_COp.14b4a412ddf73a96:core.Destroy.Core"(ptr %.loc11_21.2.temp), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !42
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.218005b34e691dcb"(ptr %a), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !42
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !42
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !42
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !43
// CHECK:STDOUT:   call void @"_COp.dfb26e502f641cc5:core.Destroy.Core"(ptr %.loc15_27.2.temp), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8d3ff16138365f48:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6e1a8e11d94ef934:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dfb26e502f641cc5:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
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
// CHECK:STDOUT:   call void @"_COp.dfb26e502f641cc5:core.Destroy.Core"(ptr %.loc19_27.2.temp), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0532bac23645330a"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642"(ptr %self), !dbg !69
// CHECK:STDOUT:   ret ptr %1, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9db51af7092ba247"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.a6387af2c84c1f91"(ptr %self), !dbg !74
// CHECK:STDOUT:   ret ptr %1, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.218005b34e691dcb"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.09ed4effa59a8188"(ptr %self), !dbg !79
// CHECK:STDOUT:   ret ptr %1, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %self), !dbg !84
// CHECK:STDOUT:   ret ptr %1, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.a6387af2c84c1f91"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !89
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !89
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !89
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %1), !dbg !90
// CHECK:STDOUT:   ret ptr %2, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.09ed4effa59a8188"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !95
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !95
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self), !dbg !95
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !95
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !95
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.e4523e01ac2590ba(ptr %2), !dbg !96
// CHECK:STDOUT:   ret ptr %3, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.4201ec9a263c5642(ptr %value) #0 !dbg !98 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %value), !dbg !101
// CHECK:STDOUT:   ret ptr %1, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self) #3 !dbg !103 {
// CHECK:STDOUT:   ret ptr %self, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.e4523e01ac2590ba(ptr %value) #0 !dbg !108 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %value), !dbg !111
// CHECK:STDOUT:   ret ptr %1, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !116
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !116
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !116
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !117
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !118
// CHECK:STDOUT:   call void @"_COp.39995c0257abf9d3:core.Destroy.Core"(ptr %1), !dbg !116
// CHECK:STDOUT:   ret ptr %2, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %self) #0 !dbg !120 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !123
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !123
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !123
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !124
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !125
// CHECK:STDOUT:   call void @"_COp.6e1a8e11d94ef934:core.Destroy.Core"(ptr %1), !dbg !123
// CHECK:STDOUT:   ret ptr %2, !dbg !126
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "pass.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "PassVoidPtr", linkageName: "_CPassVoidPtr.Main", scope: null, file: !1, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 21, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7d1b77eb1279818:core.Destroy.Core", scope: null, file: !1, line: 7, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !21, type: !15)
// CHECK:STDOUT: !24 = !DILocation(line: 7, column: 21, scope: !21)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.39995c0257abf9d3:core.Destroy.Core", scope: null, file: !1, line: 7, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !15)
// CHECK:STDOUT: !28 = !DILocation(line: 7, column: 21, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.14b4a412ddf73a96:core.Destroy.Core", scope: null, file: !1, line: 7, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !15)
// CHECK:STDOUT: !32 = !DILocation(line: 7, column: 21, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !1, line: 10, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !15)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 21, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 11, column: 3, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 10, column: 1, scope: !33)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !1, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !15)
// CHECK:STDOUT: !42 = !DILocation(line: 15, column: 27, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 15, column: 3, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 14, column: 1, scope: !39)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8d3ff16138365f48:core.Destroy.Core", scope: null, file: !1, line: 15, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !15)
// CHECK:STDOUT: !48 = !DILocation(line: 15, column: 27, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6e1a8e11d94ef934:core.Destroy.Core", scope: null, file: !1, line: 15, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !15)
// CHECK:STDOUT: !52 = !DILocation(line: 15, column: 27, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dfb26e502f641cc5:core.Destroy.Core", scope: null, file: !1, line: 15, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !15)
// CHECK:STDOUT: !56 = !DILocation(line: 15, column: 27, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !1, line: 18, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !15)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 27, scope: !57)
// CHECK:STDOUT: !61 = !DILocation(line: 19, column: 3, scope: !57)
// CHECK:STDOUT: !62 = !DILocation(line: 18, column: 1, scope: !57)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0532bac23645330a", scope: null, file: !64, line: 105, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !67)
// CHECK:STDOUT: !64 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !65 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !66 = !{!15, !15}
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !63, type: !15)
// CHECK:STDOUT: !69 = !DILocation(line: 106, column: 12, scope: !63)
// CHECK:STDOUT: !70 = !DILocation(line: 106, column: 5, scope: !63)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9db51af7092ba247", scope: null, file: !64, line: 105, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !72)
// CHECK:STDOUT: !72 = !{!73}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !71, type: !15)
// CHECK:STDOUT: !74 = !DILocation(line: 106, column: 12, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 106, column: 5, scope: !71)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.218005b34e691dcb", scope: null, file: !64, line: 105, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !77)
// CHECK:STDOUT: !77 = !{!78}
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !76, type: !15)
// CHECK:STDOUT: !79 = !DILocation(line: 106, column: 12, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 106, column: 5, scope: !76)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.4201ec9a263c5642", scope: null, file: !64, line: 80, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !15)
// CHECK:STDOUT: !84 = !DILocation(line: 81, column: 12, scope: !81)
// CHECK:STDOUT: !85 = !DILocation(line: 81, column: 5, scope: !81)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.a6387af2c84c1f91", scope: null, file: !64, line: 87, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !87)
// CHECK:STDOUT: !87 = !{!88}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !86, type: !15)
// CHECK:STDOUT: !89 = !DILocation(line: 88, column: 29, scope: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 88, column: 12, scope: !86)
// CHECK:STDOUT: !91 = !DILocation(line: 88, column: 5, scope: !86)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.09ed4effa59a8188", scope: null, file: !64, line: 87, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !15)
// CHECK:STDOUT: !95 = !DILocation(line: 88, column: 29, scope: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 88, column: 12, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 88, column: 5, scope: !92)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.4201ec9a263c5642", scope: null, file: !64, line: 33, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !99)
// CHECK:STDOUT: !99 = !{!100}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !98, type: !15)
// CHECK:STDOUT: !101 = !DILocation(line: 34, column: 12, scope: !98)
// CHECK:STDOUT: !102 = !DILocation(line: 34, column: 5, scope: !98)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.a569d188b7ffe592.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba", scope: null, file: !104, line: 40, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !105)
// CHECK:STDOUT: !104 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !105 = !{!106}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !103, type: !15)
// CHECK:STDOUT: !107 = !DILocation(line: 40, column: 3, scope: !103)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.e4523e01ac2590ba", scope: null, file: !64, line: 33, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !109)
// CHECK:STDOUT: !109 = !{!110}
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !108, type: !15)
// CHECK:STDOUT: !111 = !DILocation(line: 34, column: 12, scope: !108)
// CHECK:STDOUT: !112 = !DILocation(line: 34, column: 5, scope: !108)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271", scope: null, file: !64, line: 166, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !114)
// CHECK:STDOUT: !114 = !{!115}
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !113, type: !15)
// CHECK:STDOUT: !116 = !DILocation(line: 167, column: 14, scope: !113)
// CHECK:STDOUT: !117 = !DILocation(line: 169, column: 5, scope: !113)
// CHECK:STDOUT: !118 = !DILocation(line: 167, column: 18, scope: !113)
// CHECK:STDOUT: !119 = !DILocation(line: 170, column: 5, scope: !113)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137", scope: null, file: !64, line: 166, type: !65, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !121)
// CHECK:STDOUT: !121 = !{!122}
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !120, type: !15)
// CHECK:STDOUT: !123 = !DILocation(line: 167, column: 14, scope: !120)
// CHECK:STDOUT: !124 = !DILocation(line: 169, column: 5, scope: !120)
// CHECK:STDOUT: !125 = !DILocation(line: 167, column: 18, scope: !120)
// CHECK:STDOUT: !126 = !DILocation(line: 170, column: 5, scope: !120)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !16
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z15return_void_ptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !1, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 10, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !1, line: 10, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{!15, !15}
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !18, type: !15)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 3, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !1, line: 14, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !15)
// CHECK:STDOUT: !27 = !DILocation(line: 15, column: 3, scope: !24)
// CHECK:STDOUT:
