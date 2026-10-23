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
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.ecebba3ed00bdf07"(ptr %a), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_21.2.temp), !dbg !13
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc7_21.2.temp, align 8, !dbg !13
// CHECK:STDOUT:   %.loc7_21.4 = load ptr, ptr %.loc7_21.2.temp, align 8, !dbg !13
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc7_21.4), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.761f5b00f4a1dad7(ptr %.loc7_21.2.temp), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13take_void_ptrPv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtr.Main(ptr %a) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.2.temp = alloca ptr, align 8, !dbg !22
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.c462f23fba5b8cbc"(ptr %a), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.2.temp), !dbg !22
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.2.temp, align 8, !dbg !22
// CHECK:STDOUT:   %.loc11_21.4 = load ptr, ptr %.loc11_21.2.temp, align 8, !dbg !22
// CHECK:STDOUT:   call void @_Z13take_void_ptrPv(ptr %.loc11_21.4), !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.761f5b00f4a1dad7(ptr %.loc11_21.2.temp), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_27.2.temp = alloca ptr, align 8, !dbg !29
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.f095c12fd9a5b125"(ptr %a), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc15_27.2.temp), !dbg !29
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc15_27.2.temp, align 8, !dbg !29
// CHECK:STDOUT:   %.loc15_27.4 = load ptr, ptr %.loc15_27.2.temp, align 8, !dbg !29
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc15_27.4), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.fe28508134f64114(ptr %.loc15_27.2.temp), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19take_const_void_ptrPKv(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassConstIntPtrToConstVoidPtr.Main(ptr %a) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_27.2.temp = alloca ptr, align 8, !dbg !36
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.f095c12fd9a5b125"(ptr %a), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_27.2.temp), !dbg !36
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc19_27.2.temp, align 8, !dbg !36
// CHECK:STDOUT:   %.loc19_27.4 = load ptr, ptr %.loc19_27.2.temp, align 8, !dbg !36
// CHECK:STDOUT:   call void @_Z19take_const_void_ptrPKv(ptr %.loc19_27.4), !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.fe28508134f64114(ptr %.loc19_27.2.temp), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.ecebba3ed00bdf07"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.d733b2d6fcf89db9"(ptr %self), !dbg !46
// CHECK:STDOUT:   ret ptr %1, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.761f5b00f4a1dad7(ptr %t) #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.c462f23fba5b8cbc"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.b78ba5413650c292"(ptr %self), !dbg !62
// CHECK:STDOUT:   ret ptr %1, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.f095c12fd9a5b125"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.8ac16a46fbc024c1"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret ptr %1, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.fe28508134f64114(ptr %t) #0 !dbg !73 {
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.d733b2d6fcf89db9"(ptr %self) #0 !dbg !88 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %self), !dbg !94
// CHECK:STDOUT:   ret ptr %1, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.b78ba5413650c292"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !102
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !102
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !102
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !102
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %1), !dbg !103
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f45dc3cf254dd925(ptr %temp), !dbg !102
// CHECK:STDOUT:   ret ptr %2, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.8ac16a46fbc024c1"(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !111
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.55456748a8a2eb70"(ptr %self), !dbg !111
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !111
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !111
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.2a2e44fc411aca14(ptr %2), !dbg !112
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9990b95724ee87af(ptr %temp), !dbg !111
// CHECK:STDOUT:   ret ptr %3, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %value) #0 !dbg !124 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %value), !dbg !129
// CHECK:STDOUT:   ret ptr %1, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f45dc3cf254dd925(ptr %t) #0 !dbg !132 {
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.55456748a8a2eb70"(ptr %self) #3 !dbg !140 {
// CHECK:STDOUT:   ret ptr %self, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.2a2e44fc411aca14(ptr %value) #0 !dbg !145 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %value), !dbg !150
// CHECK:STDOUT:   ret ptr %1, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9990b95724ee87af(ptr %t) #0 !dbg !153 {
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %self) #0 !dbg !165 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !170
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !170
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !170
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !174
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !172
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25d45deb924f0301(ptr %1), !dbg !170
// CHECK:STDOUT:   ret ptr %2, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137"(ptr %self) #0 !dbg !178 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !183
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !183
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !183
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !187
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !185
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b2d5ddedd40c515e(ptr %1), !dbg !183
// CHECK:STDOUT:   ret ptr %2, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.25d45deb924f0301(ptr %t) #0 !dbg !191 {
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b2d5ddedd40c515e(ptr %t) #0 !dbg !198 {
// CHECK:STDOUT:   ret void, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.761f5b00f4a1dad7, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.f095c12fd9a5b125", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.fe28508134f64114, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.d733b2d6fcf89db9, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!205, !206, !207, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!212}
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
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "PassIntPtr", linkageName: "_CPassIntPtr.Main", scope: null, file: !4, line: 10, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 11, column: 21, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 3, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 10, column: 1, scope: !19)
// CHECK:STDOUT: !25 = !{!20}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "PassIntPtrToConstVoidPtr", linkageName: "_CPassIntPtrToConstVoidPtr.Main", scope: null, file: !4, line: 14, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !32)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !7)
// CHECK:STDOUT: !29 = !DILocation(line: 15, column: 27, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 3, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 1, scope: !26)
// CHECK:STDOUT: !32 = !{!27}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "PassConstIntPtrToConstVoidPtr", linkageName: "_CPassConstIntPtrToConstVoidPtr.Main", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !39)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !36 = !DILocation(line: 19, column: 27, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 3, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 18, column: 1, scope: !33)
// CHECK:STDOUT: !39 = !{!34}
// CHECK:STDOUT: !40 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.ecebba3ed00bdf07", scope: null, file: !40, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !48)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !7)
// CHECK:STDOUT: !46 = !DILocation(line: 106, column: 12, scope: !41)
// CHECK:STDOUT: !47 = !DILocation(line: 106, column: 5, scope: !41)
// CHECK:STDOUT: !48 = !{!42}
// CHECK:STDOUT: !49 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.761f5b00f4a1dad7", scope: null, file: !49, line: 35, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !7)
// CHECK:STDOUT: !55 = !DILocation(line: 35, column: 1, scope: !50)
// CHECK:STDOUT: !56 = !{!51}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.c462f23fba5b8cbc", scope: null, file: !40, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !7)
// CHECK:STDOUT: !62 = !DILocation(line: 106, column: 12, scope: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 106, column: 5, scope: !57)
// CHECK:STDOUT: !64 = !{!58}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.f095c12fd9a5b125", scope: null, file: !40, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !72)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !7)
// CHECK:STDOUT: !70 = !DILocation(line: 106, column: 12, scope: !65)
// CHECK:STDOUT: !71 = !DILocation(line: 106, column: 5, scope: !65)
// CHECK:STDOUT: !72 = !{!66}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.fe28508134f64114", scope: null, file: !49, line: 35, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !79)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !78 = !DILocation(line: 35, column: 1, scope: !73)
// CHECK:STDOUT: !79 = !{!74}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.d733b2d6fcf89db9", scope: null, file: !40, line: 80, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !7)
// CHECK:STDOUT: !94 = !DILocation(line: 81, column: 12, scope: !88)
// CHECK:STDOUT: !95 = !DILocation(line: 81, column: 5, scope: !88)
// CHECK:STDOUT: !96 = !{!89}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.b78ba5413650c292", scope: null, file: !40, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !105)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !102 = !DILocation(line: 88, column: 29, scope: !97)
// CHECK:STDOUT: !103 = !DILocation(line: 88, column: 12, scope: !97)
// CHECK:STDOUT: !104 = !DILocation(line: 88, column: 5, scope: !97)
// CHECK:STDOUT: !105 = !{!98}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.8ac16a46fbc024c1", scope: null, file: !40, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !114)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !7)
// CHECK:STDOUT: !111 = !DILocation(line: 88, column: 29, scope: !106)
// CHECK:STDOUT: !112 = !DILocation(line: 88, column: 12, scope: !106)
// CHECK:STDOUT: !113 = !DILocation(line: 88, column: 5, scope: !106)
// CHECK:STDOUT: !114 = !{!107}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d733b2d6fcf89db9", scope: null, file: !40, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !131)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !7)
// CHECK:STDOUT: !129 = !DILocation(line: 34, column: 12, scope: !124)
// CHECK:STDOUT: !130 = !DILocation(line: 34, column: 5, scope: !124)
// CHECK:STDOUT: !131 = !{!125}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f45dc3cf254dd925", scope: null, file: !49, line: 35, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !138)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !7)
// CHECK:STDOUT: !137 = !DILocation(line: 35, column: 1, scope: !132)
// CHECK:STDOUT: !138 = !{!133}
// CHECK:STDOUT: !139 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.55456748a8a2eb70", scope: null, file: !139, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !144)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !7)
// CHECK:STDOUT: !143 = !DILocation(line: 40, column: 3, scope: !140)
// CHECK:STDOUT: !144 = !{!141}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.2a2e44fc411aca14", scope: null, file: !40, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !7)
// CHECK:STDOUT: !150 = !DILocation(line: 34, column: 12, scope: !145)
// CHECK:STDOUT: !151 = !DILocation(line: 34, column: 5, scope: !145)
// CHECK:STDOUT: !152 = !{!146}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9990b95724ee87af", scope: null, file: !49, line: 35, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !159)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !7)
// CHECK:STDOUT: !158 = !DILocation(line: 35, column: 1, scope: !153)
// CHECK:STDOUT: !159 = !{!154}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271", scope: null, file: !40, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !7)
// CHECK:STDOUT: !170 = !DILocation(line: 167, column: 14, scope: !165)
// CHECK:STDOUT: !172 = !DILocation(line: 167, column: 18, scope: !165)
// CHECK:STDOUT: !174 = !DILocation(line: 169, column: 5, scope: !165)
// CHECK:STDOUT: !176 = !DILocation(line: 170, column: 5, scope: !165)
// CHECK:STDOUT: !177 = !{!166}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8f1054b47505c137", scope: null, file: !40, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !190)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !7)
// CHECK:STDOUT: !183 = !DILocation(line: 167, column: 14, scope: !178)
// CHECK:STDOUT: !185 = !DILocation(line: 167, column: 18, scope: !178)
// CHECK:STDOUT: !187 = !DILocation(line: 169, column: 5, scope: !178)
// CHECK:STDOUT: !189 = !DILocation(line: 170, column: 5, scope: !178)
// CHECK:STDOUT: !190 = !{!179}
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.25d45deb924f0301", scope: null, file: !49, line: 35, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !197)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !7)
// CHECK:STDOUT: !196 = !DILocation(line: 35, column: 1, scope: !191)
// CHECK:STDOUT: !197 = !{!192}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b2d5ddedd40c515e", scope: null, file: !49, line: 35, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !204)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !7)
// CHECK:STDOUT: !203 = !DILocation(line: 35, column: 1, scope: !198)
// CHECK:STDOUT: !204 = !{!199}
// CHECK:STDOUT: !205 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !206 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !207 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !208 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !209 = !{!"omnipotent char", !208, i64 0}
// CHECK:STDOUT: !210 = !{!"int", !209, i64 0}
// CHECK:STDOUT: !211 = !{!"__libc_errno", !210, i64 0}
// CHECK:STDOUT: !212 = !{!211, !210, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !220
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z15return_void_ptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !222 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!214}
// CHECK:STDOUT: !llvm.module.flags = !{!205, !206, !207, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!212}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !14 = !{!7, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !205 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !206 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !207 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !208 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !209 = !{!"omnipotent char", !208, i64 0}
// CHECK:STDOUT: !210 = !{!"int", !209, i64 0}
// CHECK:STDOUT: !211 = !{!"__libc_errno", !210, i64 0}
// CHECK:STDOUT: !212 = !{!211, !210, i64 0}
// CHECK:STDOUT: !213 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !214 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !213, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !215 = !{!7}
// CHECK:STDOUT: !216 = !DISubroutineType(types: !215)
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !213, line: 6, type: !216, spFlags: DISPFlagDefinition, unit: !214)
// CHECK:STDOUT: !220 = !DILocation(line: 7, column: 10, scope: !217)
// CHECK:STDOUT: !221 = !DILocation(line: 7, column: 3, scope: !217)
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !213, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !214, retainedNodes: !229)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !7)
// CHECK:STDOUT: !228 = !DILocation(line: 11, column: 3, scope: !222)
// CHECK:STDOUT: !229 = !{!223}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !213, line: 14, type: !15, spFlags: DISPFlagDefinition, unit: !214, retainedNodes: !237)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !7)
// CHECK:STDOUT: !236 = !DILocation(line: 15, column: 3, scope: !230)
// CHECK:STDOUT: !237 = !{!231}
// CHECK:STDOUT:
