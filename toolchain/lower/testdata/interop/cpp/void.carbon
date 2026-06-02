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
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.e2eb6345a0817262"(ptr %a), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !17
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !18
// CHECK:STDOUT:   call void @"_COp.ef40c1ad0063352d:core.Destroy.Core"(ptr %.loc7_21.2.temp), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e109d927282ba62:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ef40c1ad0063352d:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !31
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.92bc13198414a0a5"(ptr %a), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !31
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !31
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !31
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !32
// CHECK:STDOUT:   call void @"_COp.ef40c1ad0063352d:core.Destroy.Core"(ptr %.loc11_21.2.temp), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !37
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.e8abce1df9e7383b"(ptr %a), !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !37
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !37
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !37
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !38
// CHECK:STDOUT:   call void @"_COp.ded630241aa40147:core.Destroy.Core"(ptr %.loc15_27.2.temp), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a704aa97233bbaa6:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ded630241aa40147:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !51
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.e8abce1df9e7383b"(ptr %a), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !51
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !51
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !51
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !52
// CHECK:STDOUT:   call void @"_COp.ded630241aa40147:core.Destroy.Core"(ptr %.loc19_27.2.temp), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.e2eb6345a0817262"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.a0c6f87edec590a2:OptionalAs.3a7c842b5a6bca51.Core.16518ec96f303f23"(ptr %self), !dbg !60
// CHECK:STDOUT:   ret ptr %1, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.92bc13198414a0a5"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.5efd019d705c0b26"(ptr %self), !dbg !65
// CHECK:STDOUT:   ret ptr %1, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.e8abce1df9e7383b"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.ff5aadc808ace668"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret ptr %1, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.a0c6f87edec590a2:OptionalAs.3a7c842b5a6bca51.Core.16518ec96f303f23"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.16518ec96f303f23(ptr %self), !dbg !75
// CHECK:STDOUT:   ret ptr %1, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.5efd019d705c0b26"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !80
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !80
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !80
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.16518ec96f303f23(ptr %1), !dbg !81
// CHECK:STDOUT:   ret ptr %2, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.ff5aadc808ace668"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !86
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.9ec76089d908647a.Core:e8f8f92d3d08d149:ImplicitAs.37effc28ee11587a.Core.5e5386123d4b6287"(ptr %self), !dbg !86
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !86
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !86
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.15cb76de18318062(ptr %2), !dbg !87
// CHECK:STDOUT:   ret ptr %3, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.16518ec96f303f23(ptr %value) #0 !dbg !89 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.b38e2ff2a271a787"(ptr %value), !dbg !92
// CHECK:STDOUT:   ret ptr %1, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.9ec76089d908647a.Core:e8f8f92d3d08d149:ImplicitAs.37effc28ee11587a.Core.5e5386123d4b6287"(ptr %self) #3 !dbg !94 {
// CHECK:STDOUT:   ret ptr %self, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.15cb76de18318062(ptr %value) #0 !dbg !99 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.430391d0f8a776a2"(ptr %value), !dbg !102
// CHECK:STDOUT:   ret ptr %1, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.b38e2ff2a271a787"(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !107
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !107
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !107
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !108
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !109
// CHECK:STDOUT:   call void @"_COp.3e109d927282ba62:core.Destroy.Core"(ptr %1), !dbg !107
// CHECK:STDOUT:   ret ptr %2, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.430391d0f8a776a2"(ptr %self) #0 !dbg !111 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !114
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !114
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !115
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !116
// CHECK:STDOUT:   call void @"_COp.a704aa97233bbaa6:core.Destroy.Core"(ptr %1), !dbg !114
// CHECK:STDOUT:   ret ptr %2, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.e8abce1df9e7383b", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.16518ec96f303f23, { 1, 0 }
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
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 21, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e109d927282ba62:core.Destroy.Core", scope: null, file: !6, line: 7, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !21)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !20, type: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 7, column: 21, scope: !20)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ef40c1ad0063352d:core.Destroy.Core", scope: null, file: !6, line: 7, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 7, column: 21, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !6, line: 10, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 11, column: 21, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 11, column: 3, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 10, column: 1, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !6, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !14)
// CHECK:STDOUT: !37 = !DILocation(line: 15, column: 27, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 15, column: 3, scope: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 14, column: 1, scope: !34)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a704aa97233bbaa6:core.Destroy.Core", scope: null, file: !6, line: 15, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !14)
// CHECK:STDOUT: !43 = !DILocation(line: 15, column: 27, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ded630241aa40147:core.Destroy.Core", scope: null, file: !6, line: 15, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !14)
// CHECK:STDOUT: !47 = !DILocation(line: 15, column: 27, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !6, line: 18, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !48, type: !14)
// CHECK:STDOUT: !51 = !DILocation(line: 19, column: 27, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 19, column: 3, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 18, column: 1, scope: !48)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.e2eb6345a0817262", scope: null, file: !55, line: 105, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !55 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !56 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !57 = !{!14, !14}
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !54, type: !14)
// CHECK:STDOUT: !60 = !DILocation(line: 106, column: 12, scope: !54)
// CHECK:STDOUT: !61 = !DILocation(line: 106, column: 5, scope: !54)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.92bc13198414a0a5", scope: null, file: !55, line: 105, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !14)
// CHECK:STDOUT: !65 = !DILocation(line: 106, column: 12, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 106, column: 5, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.e8abce1df9e7383b", scope: null, file: !55, line: 105, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !14)
// CHECK:STDOUT: !70 = !DILocation(line: 106, column: 12, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 106, column: 5, scope: !67)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a0c6f87edec590a2:OptionalAs.3a7c842b5a6bca51.Core.16518ec96f303f23", scope: null, file: !55, line: 80, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !73)
// CHECK:STDOUT: !73 = !{!74}
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !72, type: !14)
// CHECK:STDOUT: !75 = !DILocation(line: 81, column: 12, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 81, column: 5, scope: !72)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.5efd019d705c0b26", scope: null, file: !55, line: 87, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !78)
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !77, type: !14)
// CHECK:STDOUT: !80 = !DILocation(line: 88, column: 29, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 88, column: 12, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 88, column: 5, scope: !77)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.ff5aadc808ace668", scope: null, file: !55, line: 87, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !84 = !{!85}
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !83, type: !14)
// CHECK:STDOUT: !86 = !DILocation(line: 88, column: 29, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 88, column: 12, scope: !83)
// CHECK:STDOUT: !88 = !DILocation(line: 88, column: 5, scope: !83)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.16518ec96f303f23", scope: null, file: !55, line: 33, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !90)
// CHECK:STDOUT: !90 = !{!91}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !89, type: !14)
// CHECK:STDOUT: !92 = !DILocation(line: 34, column: 12, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 34, column: 5, scope: !89)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.9ec76089d908647a.Core:e8f8f92d3d08d149:ImplicitAs.37effc28ee11587a.Core.5e5386123d4b6287", scope: null, file: !95, line: 40, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !95 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !94, type: !14)
// CHECK:STDOUT: !98 = !DILocation(line: 40, column: 3, scope: !94)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.15cb76de18318062", scope: null, file: !55, line: 33, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !100 = !{!101}
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !99, type: !14)
// CHECK:STDOUT: !102 = !DILocation(line: 34, column: 12, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 34, column: 5, scope: !99)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.b38e2ff2a271a787", scope: null, file: !55, line: 166, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !105)
// CHECK:STDOUT: !105 = !{!106}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !104, type: !14)
// CHECK:STDOUT: !107 = !DILocation(line: 167, column: 14, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 169, column: 5, scope: !104)
// CHECK:STDOUT: !109 = !DILocation(line: 167, column: 18, scope: !104)
// CHECK:STDOUT: !110 = !DILocation(line: 170, column: 5, scope: !104)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.430391d0f8a776a2", scope: null, file: !55, line: 166, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !112 = !{!113}
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !111, type: !14)
// CHECK:STDOUT: !114 = !DILocation(line: 167, column: 14, scope: !111)
// CHECK:STDOUT: !115 = !DILocation(line: 169, column: 5, scope: !111)
// CHECK:STDOUT: !116 = !DILocation(line: 167, column: 18, scope: !111)
// CHECK:STDOUT: !117 = !DILocation(line: 170, column: 5, scope: !111)
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
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
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
