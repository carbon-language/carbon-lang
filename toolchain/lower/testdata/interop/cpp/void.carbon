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
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.52f71e11dc373f9a"(ptr %a), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !17
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !18
// CHECK:STDOUT:   call void @"_COp.801be0c493a24c71:core.Destroy.Core"(ptr %.loc7_21.2.temp), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fd39a15e7d61228d:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.125695b29f2cf97c:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.801be0c493a24c71:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !35
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.ed3bde3932287ced"(ptr %a), !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !35
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !35
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !35
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !36
// CHECK:STDOUT:   call void @"_COp.801be0c493a24c71:core.Destroy.Core"(ptr %.loc11_21.2.temp), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !41
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0dc2e35b064d71e9"(ptr %a), !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !41
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !41
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !41
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !42
// CHECK:STDOUT:   call void @"_COp.d7b9c2d58a39f2df:core.Destroy.Core"(ptr %.loc15_27.2.temp), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0aac4ce7c7765bae:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b8b38d4701ca7f8c:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d7b9c2d58a39f2df:core.Destroy.Core"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !59
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0dc2e35b064d71e9"(ptr %a), !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !59
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !59
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !59
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !60
// CHECK:STDOUT:   call void @"_COp.d7b9c2d58a39f2df:core.Destroy.Core"(ptr %.loc19_27.2.temp), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.52f71e11dc373f9a"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.258ff42e9fd73c28"(ptr %self), !dbg !68
// CHECK:STDOUT:   ret ptr %1, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.ed3bde3932287ced"(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.ff6ab29b5872c34c"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret ptr %1, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0dc2e35b064d71e9"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.e316e6328124c0b3"(ptr %self), !dbg !78
// CHECK:STDOUT:   ret ptr %1, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.258ff42e9fd73c28"(ptr %self) #0 !dbg !80 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.258ff42e9fd73c28(ptr %self), !dbg !83
// CHECK:STDOUT:   ret ptr %1, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.ff6ab29b5872c34c"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !88
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !88
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !88
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.258ff42e9fd73c28(ptr %1), !dbg !89
// CHECK:STDOUT:   ret ptr %2, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.e316e6328124c0b3"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !94
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.d035546bf1a99743.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self), !dbg !94
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !94
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !94
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.e5c5bc28e9d686cf(ptr %2), !dbg !95
// CHECK:STDOUT:   ret ptr %3, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.258ff42e9fd73c28(ptr %value) #0 !dbg !97 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %value), !dbg !100
// CHECK:STDOUT:   ret ptr %1, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.d035546bf1a99743.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba"(ptr %self) #3 !dbg !102 {
// CHECK:STDOUT:   ret ptr %self, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.e5c5bc28e9d686cf(ptr %value) #0 !dbg !107 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %value), !dbg !110
// CHECK:STDOUT:   ret ptr %1, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %self) #0 !dbg !112 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !115
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !115
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !115
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !116
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !117
// CHECK:STDOUT:   call void @"_COp.125695b29f2cf97c:core.Destroy.Core"(ptr %1), !dbg !115
// CHECK:STDOUT:   ret ptr %2, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %self) #0 !dbg !119 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !122
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !122
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !122
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !123
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !124
// CHECK:STDOUT:   call void @"_COp.b8b38d4701ca7f8c:core.Destroy.Core"(ptr %1), !dbg !122
// CHECK:STDOUT:   ret ptr %2, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0dc2e35b064d71e9", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.258ff42e9fd73c28, { 1, 0 }
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
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassVoidPtr", linkageName: "_CPassVoidPtr.Main", scope: null, file: !1, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 21, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fd39a15e7d61228d:core.Destroy.Core", scope: null, file: !1, line: 7, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !20, type: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 7, column: 21, scope: !20)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.125695b29f2cf97c:core.Destroy.Core", scope: null, file: !1, line: 7, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 7, column: 21, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.801be0c493a24c71:core.Destroy.Core", scope: null, file: !1, line: 7, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 7, column: 21, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !1, line: 10, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !14)
// CHECK:STDOUT: !35 = !DILocation(line: 11, column: 21, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 3, scope: !32)
// CHECK:STDOUT: !37 = !DILocation(line: 10, column: 1, scope: !32)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !1, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !14)
// CHECK:STDOUT: !41 = !DILocation(line: 15, column: 27, scope: !38)
// CHECK:STDOUT: !42 = !DILocation(line: 15, column: 3, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 1, scope: !38)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0aac4ce7c7765bae:core.Destroy.Core", scope: null, file: !1, line: 15, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !14)
// CHECK:STDOUT: !47 = !DILocation(line: 15, column: 27, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b8b38d4701ca7f8c:core.Destroy.Core", scope: null, file: !1, line: 15, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !49)
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !48, type: !14)
// CHECK:STDOUT: !51 = !DILocation(line: 15, column: 27, scope: !48)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d7b9c2d58a39f2df:core.Destroy.Core", scope: null, file: !1, line: 15, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !53)
// CHECK:STDOUT: !53 = !{!54}
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !52, type: !14)
// CHECK:STDOUT: !55 = !DILocation(line: 15, column: 27, scope: !52)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !1, line: 18, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !57)
// CHECK:STDOUT: !57 = !{!58}
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !56, type: !14)
// CHECK:STDOUT: !59 = !DILocation(line: 19, column: 27, scope: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 3, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 18, column: 1, scope: !56)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.52f71e11dc373f9a", scope: null, file: !63, line: 105, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !66)
// CHECK:STDOUT: !63 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !64 = !DISubroutineType(types: !65)
// CHECK:STDOUT: !65 = !{!14, !14}
// CHECK:STDOUT: !66 = !{!67}
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !62, type: !14)
// CHECK:STDOUT: !68 = !DILocation(line: 106, column: 12, scope: !62)
// CHECK:STDOUT: !69 = !DILocation(line: 106, column: 5, scope: !62)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.ed3bde3932287ced", scope: null, file: !63, line: 105, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !71)
// CHECK:STDOUT: !71 = !{!72}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !70, type: !14)
// CHECK:STDOUT: !73 = !DILocation(line: 106, column: 12, scope: !70)
// CHECK:STDOUT: !74 = !DILocation(line: 106, column: 5, scope: !70)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0dc2e35b064d71e9", scope: null, file: !63, line: 105, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !76)
// CHECK:STDOUT: !76 = !{!77}
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !75, type: !14)
// CHECK:STDOUT: !78 = !DILocation(line: 106, column: 12, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 106, column: 5, scope: !75)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.258ff42e9fd73c28", scope: null, file: !63, line: 80, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !81)
// CHECK:STDOUT: !81 = !{!82}
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !80, type: !14)
// CHECK:STDOUT: !83 = !DILocation(line: 81, column: 12, scope: !80)
// CHECK:STDOUT: !84 = !DILocation(line: 81, column: 5, scope: !80)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.ff6ab29b5872c34c", scope: null, file: !63, line: 87, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !86)
// CHECK:STDOUT: !86 = !{!87}
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !85, type: !14)
// CHECK:STDOUT: !88 = !DILocation(line: 88, column: 29, scope: !85)
// CHECK:STDOUT: !89 = !DILocation(line: 88, column: 12, scope: !85)
// CHECK:STDOUT: !90 = !DILocation(line: 88, column: 5, scope: !85)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.e316e6328124c0b3", scope: null, file: !63, line: 87, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !92)
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !91, type: !14)
// CHECK:STDOUT: !94 = !DILocation(line: 88, column: 29, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 88, column: 12, scope: !91)
// CHECK:STDOUT: !96 = !DILocation(line: 88, column: 5, scope: !91)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.258ff42e9fd73c28", scope: null, file: !63, line: 33, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !98)
// CHECK:STDOUT: !98 = !{!99}
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !97, type: !14)
// CHECK:STDOUT: !100 = !DILocation(line: 34, column: 12, scope: !97)
// CHECK:STDOUT: !101 = !DILocation(line: 34, column: 5, scope: !97)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.d035546bf1a99743.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.84588f41d61dafba", scope: null, file: !103, line: 40, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !104)
// CHECK:STDOUT: !103 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !104 = !{!105}
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !102, type: !14)
// CHECK:STDOUT: !106 = !DILocation(line: 40, column: 3, scope: !102)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.e5c5bc28e9d686cf", scope: null, file: !63, line: 33, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !108)
// CHECK:STDOUT: !108 = !{!109}
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !107, type: !14)
// CHECK:STDOUT: !110 = !DILocation(line: 34, column: 12, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 34, column: 5, scope: !107)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271", scope: null, file: !63, line: 166, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !113)
// CHECK:STDOUT: !113 = !{!114}
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !112, type: !14)
// CHECK:STDOUT: !115 = !DILocation(line: 167, column: 14, scope: !112)
// CHECK:STDOUT: !116 = !DILocation(line: 169, column: 5, scope: !112)
// CHECK:STDOUT: !117 = !DILocation(line: 167, column: 18, scope: !112)
// CHECK:STDOUT: !118 = !DILocation(line: 170, column: 5, scope: !112)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137", scope: null, file: !63, line: 166, type: !64, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !120)
// CHECK:STDOUT: !120 = !{!121}
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !119, type: !14)
// CHECK:STDOUT: !122 = !DILocation(line: 167, column: 14, scope: !119)
// CHECK:STDOUT: !123 = !DILocation(line: 169, column: 5, scope: !119)
// CHECK:STDOUT: !124 = !DILocation(line: 167, column: 18, scope: !119)
// CHECK:STDOUT: !125 = !DILocation(line: 170, column: 5, scope: !119)
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
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !1, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 10, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !1, line: 10, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{!14, !14}
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !17, type: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 11, column: 3, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !1, line: 14, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !14)
// CHECK:STDOUT: !26 = !DILocation(line: 15, column: 3, scope: !23)
