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
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.c462f23fba5b8cbc"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.b78ba5413650c292"(ptr %self), !dbg !54
// CHECK:STDOUT:   ret ptr %1, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.f095c12fd9a5b125"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.8ac16a46fbc024c1"(ptr %self), !dbg !62
// CHECK:STDOUT:   ret ptr %1, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.d733b2d6fcf89db9"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %self), !dbg !79
// CHECK:STDOUT:   ret ptr %1, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.b78ba5413650c292"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !87
// CHECK:STDOUT:   store ptr %self, ptr %temp, align 8, !dbg !87
// CHECK:STDOUT:   %1 = load ptr, ptr %temp, align 8, !dbg !87
// CHECK:STDOUT:   %2 = call ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %1), !dbg !88
// CHECK:STDOUT:   ret ptr %2, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.8ac16a46fbc024c1"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !96
// CHECK:STDOUT:   %1 = call ptr @"_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.55456748a8a2eb70"(ptr %self), !dbg !96
// CHECK:STDOUT:   store ptr %1, ptr %temp, align 8, !dbg !96
// CHECK:STDOUT:   %2 = load ptr, ptr %temp, align 8, !dbg !96
// CHECK:STDOUT:   %3 = call ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %2), !dbg !97
// CHECK:STDOUT:   ret ptr %3, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.d733b2d6fcf89db9(ptr %value) #0 !dbg !109 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %value), !dbg !114
// CHECK:STDOUT:   ret ptr %1, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.55456748a8a2eb70"(ptr %self) #3 !dbg !118 {
// CHECK:STDOUT:   ret ptr %self, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271"(ptr %self) #0 !dbg !136 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !141
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !141
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !141
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !145
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !143
// CHECK:STDOUT:   ret ptr %2, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 6, 5, 4, 3 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.f095c12fd9a5b125", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSome.Optional.Core.d733b2d6fcf89db9, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!162, !163, !164, !165, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!170}
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
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.c462f23fba5b8cbc", scope: null, file: !40, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !56)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !54 = !DILocation(line: 106, column: 12, scope: !49)
// CHECK:STDOUT: !55 = !DILocation(line: 106, column: 5, scope: !49)
// CHECK:STDOUT: !56 = !{!50}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.f095c12fd9a5b125", scope: null, file: !40, line: 105, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !7)
// CHECK:STDOUT: !62 = !DILocation(line: 106, column: 12, scope: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 106, column: 5, scope: !57)
// CHECK:STDOUT: !64 = !{!58}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.d733b2d6fcf89db9", scope: null, file: !40, line: 80, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !81)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !79 = !DILocation(line: 81, column: 12, scope: !73)
// CHECK:STDOUT: !80 = !DILocation(line: 81, column: 5, scope: !73)
// CHECK:STDOUT: !81 = !{!74}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.b78ba5413650c292", scope: null, file: !40, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !90)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !7)
// CHECK:STDOUT: !87 = !DILocation(line: 88, column: 29, scope: !82)
// CHECK:STDOUT: !88 = !DILocation(line: 88, column: 12, scope: !82)
// CHECK:STDOUT: !89 = !DILocation(line: 88, column: 5, scope: !82)
// CHECK:STDOUT: !90 = !{!83}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.8ac16a46fbc024c1", scope: null, file: !40, line: 87, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !99)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !7)
// CHECK:STDOUT: !96 = !DILocation(line: 88, column: 29, scope: !91)
// CHECK:STDOUT: !97 = !DILocation(line: 88, column: 12, scope: !91)
// CHECK:STDOUT: !98 = !DILocation(line: 88, column: 5, scope: !91)
// CHECK:STDOUT: !99 = !{!92}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d733b2d6fcf89db9", scope: null, file: !40, line: 33, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !116)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !114 = !DILocation(line: 34, column: 12, scope: !109)
// CHECK:STDOUT: !115 = !DILocation(line: 34, column: 5, scope: !109)
// CHECK:STDOUT: !116 = !{!110}
// CHECK:STDOUT: !117 = !DIFile(filename: "{{.*}}/prelude/types/cpp/void.carbon", directory: "")
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert:thunk:ImplicitAs.8c382885c27c04bd.Core:e8f8f92d3d08d149:ImplicitAs.01c4a560c986e7d1.Core.55456748a8a2eb70", scope: null, file: !117, line: 40, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !122)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !7)
// CHECK:STDOUT: !121 = !DILocation(line: 40, column: 3, scope: !118)
// CHECK:STDOUT: !122 = !{!119}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.7a595a06c724a271", scope: null, file: !40, line: 166, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !7)
// CHECK:STDOUT: !141 = !DILocation(line: 167, column: 14, scope: !136)
// CHECK:STDOUT: !143 = !DILocation(line: 167, column: 18, scope: !136)
// CHECK:STDOUT: !145 = !DILocation(line: 169, column: 5, scope: !136)
// CHECK:STDOUT: !147 = !DILocation(line: 170, column: 5, scope: !136)
// CHECK:STDOUT: !148 = !{!137}
// CHECK:STDOUT: !162 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !163 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !164 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !165 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !166 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !167 = !{!"omnipotent char", !166, i64 0}
// CHECK:STDOUT: !168 = !{!"int", !167, i64 0}
// CHECK:STDOUT: !169 = !{!"__libc_errno", !168, i64 0}
// CHECK:STDOUT: !170 = !{!169, !168, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'receive.carbon'
// CHECK:STDOUT: source_filename = "receive.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReceiveVoidPtr.Main() #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return_void_ptr.call = call ptr @_Z15return_void_ptrv(), !dbg !178
// CHECK:STDOUT:   ret ptr %return_void_ptr.call, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z15return_void_ptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromVoidPtr.Main(ptr %p) #0 !dbg !180 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !186
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertFromConstVoidPtr.Main(ptr %p) #0 !dbg !188 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %p, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!172}
// CHECK:STDOUT: !llvm.module.flags = !{!162, !163, !164, !165, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!170}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !14 = !{!7, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !162 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !163 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !164 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !165 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !166 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !167 = !{!"omnipotent char", !166, i64 0}
// CHECK:STDOUT: !168 = !{!"int", !167, i64 0}
// CHECK:STDOUT: !169 = !{!"__libc_errno", !168, i64 0}
// CHECK:STDOUT: !170 = !{!169, !168, i64 0}
// CHECK:STDOUT: !171 = !DIFile(filename: "receive.carbon", directory: "")
// CHECK:STDOUT: !172 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !171, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !173 = !{!7}
// CHECK:STDOUT: !174 = !DISubroutineType(types: !173)
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "ReceiveVoidPtr", linkageName: "_CReceiveVoidPtr.Main", scope: null, file: !171, line: 6, type: !174, spFlags: DISPFlagDefinition, unit: !172)
// CHECK:STDOUT: !178 = !DILocation(line: 7, column: 10, scope: !175)
// CHECK:STDOUT: !179 = !DILocation(line: 7, column: 3, scope: !175)
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "ConvertFromVoidPtr", linkageName: "_CConvertFromVoidPtr.Main", scope: null, file: !171, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !172, retainedNodes: !187)
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 1, scope: !180, type: !7)
// CHECK:STDOUT: !186 = !DILocation(line: 11, column: 3, scope: !180)
// CHECK:STDOUT: !187 = !{!181}
// CHECK:STDOUT: !188 = distinct !DISubprogram(name: "ConvertFromConstVoidPtr", linkageName: "_CConvertFromConstVoidPtr.Main", scope: null, file: !171, line: 14, type: !15, spFlags: DISPFlagDefinition, unit: !172, retainedNodes: !195)
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 1, scope: !188, type: !7)
// CHECK:STDOUT: !194 = !DILocation(line: 15, column: 3, scope: !188)
// CHECK:STDOUT: !195 = !{!189}
// CHECK:STDOUT:
