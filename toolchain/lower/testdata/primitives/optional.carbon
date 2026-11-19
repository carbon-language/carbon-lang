// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is an integration test for the Core.Optional in the prelude, so use the
// real prelude.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/optional.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/optional.carbon

// --- convert.carbon
library "[[@TEST_NAME]]";

fn Convert(o: Core.Optional(i32*)) -> Core.Optional(i32) {
  if (o.HasValue()) {
    return *o.Get();
  }
  return Core.Optional(i32).None();
}

fn AddOrRemoveConst(a: i32, b: const i32) {
  var _: Core.Optional(i32) = a;
  var _: const Core.Optional(i32) = a;
  var _: Core.Optional(const i32) = a;
  var _: const Core.Optional(const i32) = a;
  var _: Core.Optional(i32) = b;
  var _: const Core.Optional(i32) = b;
  var _: Core.Optional(const i32) = b;
  var _: const Core.Optional(const i32) = b;
}

// --- copy.carbon
library "[[@TEST_NAME]]";

class Copyable {
  impl as Core.Copy {
    fn Op(self) -> Self;
  }
}

fn F(c: Core.Optional(Copyable)) -> Core.Optional(Copyable) {
  // Should only perform one copy.
  return c;
}

// --- copy_pointer.carbon
library "[[@TEST_NAME]]";

class C {}

fn F(c: Core.Optional(C*)) -> Core.Optional(C*) {
  return c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.2313f7f09b1ddaea(ptr %o), !dbg !17
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.2313f7f09b1ddaea(ptr %o), !dbg !19
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1"(ptr %return, i32 %.loc5_12.2), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.787b67b8b3ba12b7(ptr %return), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 4, !dbg !34
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 4, !dbg !37
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 4, !dbg !40
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 4, !dbg !43
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 4, !dbg !46
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 4, !dbg !49
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 4, !dbg !52
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 4, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !34
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1"(ptr %_.var.loc11, i32 %a), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !37
// CHECK:STDOUT:   call void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.bd87f884f714419a"(ptr %_.var.loc12, i32 %a), !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !40
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0"(ptr %_.var.loc13, i32 %a), !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !43
// CHECK:STDOUT:   call void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.9bf3cfed350cc00b"(ptr %_.var.loc14, i32 %a), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !46
// CHECK:STDOUT:   call void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.bd87f884f714419a"(ptr %_.var.loc15, i32 %b), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !49
// CHECK:STDOUT:   call void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.c0697012bd57e06d"(ptr %_.var.loc16, i32 %b), !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !52
// CHECK:STDOUT:   call void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.9bf3cfed350cc00b"(ptr %_.var.loc17, i32 %b), !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !55
// CHECK:STDOUT:   call void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.3d5265f4c4ec9eb6"(ptr %_.var.loc18, i32 %b), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.2313f7f09b1ddaea(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70"(ptr %self), !dbg !67
// CHECK:STDOUT:   ret i1 %1, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.2313f7f09b1ddaea(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret ptr %1, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !79 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.787b67b8b3ba12b7"(ptr %return, i32 %self), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.787b67b8b3ba12b7(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6"(ptr %return), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.bd87f884f714419a"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !93 {
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1"(ptr %return, i32 %self), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !101 {
// CHECK:STDOUT:   call void @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.98b8f93c1b92fd53"(ptr %return, i32 %self), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.9bf3cfed350cc00b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !109 {
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0"(ptr %return, i32 %self), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.bd87f884f714419a"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !117 {
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1"(ptr %return, i32 %self), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.c0697012bd57e06d"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !126 {
// CHECK:STDOUT:   call void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.bd87f884f714419a"(ptr %return, i32 %self), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.9bf3cfed350cc00b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !135 {
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0"(ptr %return, i32 %self), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.3d5265f4c4ec9eb6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !144 {
// CHECK:STDOUT:   call void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.9bf3cfed350cc00b"(ptr %return, i32 %self), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70"(ptr %value) #0 !dbg !153 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !159
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !160
// CHECK:STDOUT:   ret i1 %2, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70"(ptr %value) #0 !dbg !163 {
// CHECK:STDOUT:   ret ptr %value, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.787b67b8b3ba12b7"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !171 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.787b67b8b3ba12b7(ptr %return, i32 %self), !dbg !177
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !180 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !187
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !187
// CHECK:STDOUT:   ret void, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.98b8f93c1b92fd53"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !189 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !194
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !194
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.1f633674d24603c2"(i32 %self), !dbg !194
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !194
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !194
// CHECK:STDOUT:   call void @_CSome.Optional.Core.45dd3cdcb0ab47d7(ptr %return, i32 %2), !dbg !195
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.787b67b8b3ba12b7(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !198 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6"(ptr %return, i32 %value), !dbg !203
// CHECK:STDOUT:   ret void, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.1f633674d24603c2"(i32 %self) #0 !dbg !206 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self), !dbg !211
// CHECK:STDOUT:   ret i32 %1, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.45dd3cdcb0ab47d7(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !216 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.823a8f9f91058c72"(ptr %return, i32 %value), !dbg !221
// CHECK:STDOUT:   ret void, !dbg !222
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !224 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !233
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !233
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !235
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !235
// CHECK:STDOUT:   ret void, !dbg !236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !239 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !244
// CHECK:STDOUT:   ret i32 %1, !dbg !246
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.823a8f9f91058c72"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !248 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !257
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.2fad9574e050607e"(i32 %self), !dbg !258
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !257
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !259
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !259
// CHECK:STDOUT:   ret void, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !262 {
// CHECK:STDOUT:   ret i32 %self, !dbg !268
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.2fad9574e050607e"(i32 %self) #0 !dbg !271 {
// CHECK:STDOUT:   ret i32 %self, !dbg !278
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 8, 7, 6, 5, 4, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null, !6, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !4, line: 3, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !28)
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 4, column: 7, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 4, column: 6, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 5, column: 13, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 5, column: 12, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 5, column: 5, scope: !12)
// CHECK:STDOUT: !22 = !{!7, !6}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 7, column: 10, scope: !12)
// CHECK:STDOUT: !25 = !{!7}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !28 = !{!13}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !4, line: 10, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !59)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !6)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 2, scope: !29, type: !6)
// CHECK:STDOUT: !34 = !DILocation(line: 11, column: 3, scope: !29)
// CHECK:STDOUT: !37 = !DILocation(line: 12, column: 3, scope: !29)
// CHECK:STDOUT: !40 = !DILocation(line: 13, column: 3, scope: !29)
// CHECK:STDOUT: !43 = !DILocation(line: 14, column: 3, scope: !29)
// CHECK:STDOUT: !46 = !DILocation(line: 15, column: 3, scope: !29)
// CHECK:STDOUT: !49 = !DILocation(line: 16, column: 3, scope: !29)
// CHECK:STDOUT: !52 = !DILocation(line: 17, column: 3, scope: !29)
// CHECK:STDOUT: !55 = !DILocation(line: 18, column: 3, scope: !29)
// CHECK:STDOUT: !58 = !DILocation(line: 10, column: 1, scope: !29)
// CHECK:STDOUT: !59 = !{!30, !31}
// CHECK:STDOUT: !60 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.2313f7f09b1ddaea", scope: null, file: !60, line: 36, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !7)
// CHECK:STDOUT: !67 = !DILocation(line: 37, column: 12, scope: !61)
// CHECK:STDOUT: !68 = !DILocation(line: 37, column: 5, scope: !61)
// CHECK:STDOUT: !69 = !{!62}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.2313f7f09b1ddaea", scope: null, file: !60, line: 39, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !78)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !7)
// CHECK:STDOUT: !76 = !DILocation(line: 40, column: 12, scope: !70)
// CHECK:STDOUT: !77 = !DILocation(line: 40, column: 5, scope: !70)
// CHECK:STDOUT: !78 = !{!71}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1", scope: null, file: !60, line: 105, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !86)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !6)
// CHECK:STDOUT: !84 = !DILocation(line: 106, column: 12, scope: !79)
// CHECK:STDOUT: !85 = !DILocation(line: 106, column: 5, scope: !79)
// CHECK:STDOUT: !86 = !{!80}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.787b67b8b3ba12b7", scope: null, file: !60, line: 30, type: !26, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !90 = !DILocation(line: 31, column: 12, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 31, column: 5, scope: !87)
// CHECK:STDOUT: !92 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.bd87f884f714419a", scope: null, file: !92, line: 43, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !6)
// CHECK:STDOUT: !98 = !DILocation(line: 43, column: 40, scope: !93)
// CHECK:STDOUT: !99 = !DILocation(line: 43, column: 33, scope: !93)
// CHECK:STDOUT: !100 = !{!94}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0", scope: null, file: !60, line: 105, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !6)
// CHECK:STDOUT: !106 = !DILocation(line: 106, column: 12, scope: !101)
// CHECK:STDOUT: !107 = !DILocation(line: 106, column: 5, scope: !101)
// CHECK:STDOUT: !108 = !{!102}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.9bf3cfed350cc00b", scope: null, file: !92, line: 43, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !116)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !6)
// CHECK:STDOUT: !114 = !DILocation(line: 43, column: 40, scope: !109)
// CHECK:STDOUT: !115 = !DILocation(line: 43, column: 33, scope: !109)
// CHECK:STDOUT: !116 = !{!110}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.bd87f884f714419a", scope: null, file: !92, line: 47, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !125)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !6)
// CHECK:STDOUT: !123 = !DILocation(line: 47, column: 34, scope: !117)
// CHECK:STDOUT: !124 = !DILocation(line: 47, column: 27, scope: !117)
// CHECK:STDOUT: !125 = !{!118}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.c0697012bd57e06d", scope: null, file: !92, line: 47, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !134)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !6)
// CHECK:STDOUT: !132 = !DILocation(line: 47, column: 34, scope: !126)
// CHECK:STDOUT: !133 = !DILocation(line: 47, column: 27, scope: !126)
// CHECK:STDOUT: !134 = !{!127}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.9bf3cfed350cc00b", scope: null, file: !92, line: 47, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !143)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !6)
// CHECK:STDOUT: !141 = !DILocation(line: 47, column: 34, scope: !135)
// CHECK:STDOUT: !142 = !DILocation(line: 47, column: 27, scope: !135)
// CHECK:STDOUT: !143 = !{!136}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.3d5265f4c4ec9eb6", scope: null, file: !92, line: 47, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !6)
// CHECK:STDOUT: !150 = !DILocation(line: 47, column: 34, scope: !144)
// CHECK:STDOUT: !151 = !DILocation(line: 47, column: 27, scope: !144)
// CHECK:STDOUT: !152 = !{!145}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70", scope: null, file: !60, line: 172, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !162)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !7)
// CHECK:STDOUT: !159 = !DILocation(line: 173, column: 16, scope: !153)
// CHECK:STDOUT: !160 = !DILocation(line: 173, column: 12, scope: !153)
// CHECK:STDOUT: !161 = !DILocation(line: 173, column: 5, scope: !153)
// CHECK:STDOUT: !162 = !{!154}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70", scope: null, file: !60, line: 175, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !170)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !7)
// CHECK:STDOUT: !169 = !DILocation(line: 176, column: 5, scope: !163)
// CHECK:STDOUT: !170 = !{!164}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.787b67b8b3ba12b7", scope: null, file: !60, line: 80, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !179)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !6)
// CHECK:STDOUT: !177 = !DILocation(line: 81, column: 12, scope: !171)
// CHECK:STDOUT: !178 = !DILocation(line: 81, column: 5, scope: !171)
// CHECK:STDOUT: !179 = !{!172}
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6", scope: null, file: !60, line: 125, type: !26, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !187 = !DILocation(line: 128, column: 5, scope: !180)
// CHECK:STDOUT: !188 = !DILocation(line: 129, column: 5, scope: !180)
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.98b8f93c1b92fd53", scope: null, file: !60, line: 87, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !197)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !6)
// CHECK:STDOUT: !194 = !DILocation(line: 88, column: 29, scope: !189)
// CHECK:STDOUT: !195 = !DILocation(line: 88, column: 12, scope: !189)
// CHECK:STDOUT: !196 = !DILocation(line: 88, column: 5, scope: !189)
// CHECK:STDOUT: !197 = !{!190}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.787b67b8b3ba12b7", scope: null, file: !60, line: 33, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !205)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !6)
// CHECK:STDOUT: !203 = !DILocation(line: 34, column: 12, scope: !198)
// CHECK:STDOUT: !204 = !DILocation(line: 34, column: 5, scope: !198)
// CHECK:STDOUT: !205 = !{!199}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.1f633674d24603c2", scope: null, file: !92, line: 43, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !215)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !6)
// CHECK:STDOUT: !211 = !DILocation(line: 43, column: 40, scope: !206)
// CHECK:STDOUT: !212 = !{!6, !6}
// CHECK:STDOUT: !213 = !DISubroutineType(types: !212)
// CHECK:STDOUT: !214 = !DILocation(line: 43, column: 33, scope: !206)
// CHECK:STDOUT: !215 = !{!207}
// CHECK:STDOUT: !216 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.45dd3cdcb0ab47d7", scope: null, file: !60, line: 33, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !223)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 1, scope: !216, type: !6)
// CHECK:STDOUT: !221 = !DILocation(line: 34, column: 12, scope: !216)
// CHECK:STDOUT: !222 = !DILocation(line: 34, column: 5, scope: !216)
// CHECK:STDOUT: !223 = !{!217}
// CHECK:STDOUT: !224 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6", scope: null, file: !60, line: 131, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !237)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 1, scope: !224, type: !6)
// CHECK:STDOUT: !233 = !DILocation(line: 137, column: 5, scope: !224)
// CHECK:STDOUT: !235 = !DILocation(line: 138, column: 5, scope: !224)
// CHECK:STDOUT: !236 = !DILocation(line: 139, column: 5, scope: !224)
// CHECK:STDOUT: !237 = !{!225}
// CHECK:STDOUT: !238 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !238, line: 64, type: !213, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !247)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !6)
// CHECK:STDOUT: !244 = !DILocation(line: 66, column: 17, scope: !239)
// CHECK:STDOUT: !246 = !DILocation(line: 66, column: 5, scope: !239)
// CHECK:STDOUT: !247 = !{!240}
// CHECK:STDOUT: !248 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.823a8f9f91058c72", scope: null, file: !60, line: 131, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !261)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 1, scope: !248, type: !6)
// CHECK:STDOUT: !257 = !DILocation(line: 137, column: 5, scope: !248)
// CHECK:STDOUT: !258 = !DILocation(line: 137, column: 28, scope: !248)
// CHECK:STDOUT: !259 = !DILocation(line: 138, column: 5, scope: !248)
// CHECK:STDOUT: !260 = !DILocation(line: 139, column: 5, scope: !248)
// CHECK:STDOUT: !261 = !{!249}
// CHECK:STDOUT: !262 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !238, line: 59, type: !213, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !269)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 1, scope: !262, type: !6)
// CHECK:STDOUT: !268 = !DILocation(line: 59, column: 28, scope: !262)
// CHECK:STDOUT: !269 = !{!263}
// CHECK:STDOUT: !270 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.2fad9574e050607e", scope: null, file: !270, line: 23, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !279)
// CHECK:STDOUT: !272 = !DILocalVariable(arg: 1, scope: !271, type: !6)
// CHECK:STDOUT: !278 = !DILocation(line: 23, column: 25, scope: !271)
// CHECK:STDOUT: !279 = !{!272}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !282 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.60b535298816170e"(ptr %return, ptr %c), !dbg !287
// CHECK:STDOUT:   ret void, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.60b535298816170e"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !290 {
// CHECK:STDOUT:   call void @"_CCopy.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %return, ptr %self), !dbg !296
// CHECK:STDOUT:   ret void, !dbg !297
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !299 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %value), !dbg !305
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !306
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !307
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %return, ptr %value1), !dbg !308
// CHECK:STDOUT:   ret void, !dbg !309
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %return), !dbg !310
// CHECK:STDOUT:   ret void, !dbg !311
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %value) #0 !dbg !313 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !318
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !318
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !318
// CHECK:STDOUT:   ret i1 %2, !dbg !319
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !321 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !330
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !331
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !332
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !332
// CHECK:STDOUT:   ret void, !dbg !333
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr sret({ {}, i1 }) %return) #0 !dbg !335 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !342
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !342
// CHECK:STDOUT:   ret void, !dbg !343
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!281}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !25 = !{!7}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !60 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !280 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !281 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !280, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !282 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !280, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !281, retainedNodes: !289)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 1, scope: !282, type: !7)
// CHECK:STDOUT: !287 = !DILocation(line: 11, column: 10, scope: !282)
// CHECK:STDOUT: !288 = !DILocation(line: 11, column: 3, scope: !282)
// CHECK:STDOUT: !289 = !{!283}
// CHECK:STDOUT: !290 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.60b535298816170e", scope: null, file: !60, line: 50, type: !9, spFlags: DISPFlagDefinition, unit: !281, retainedNodes: !298)
// CHECK:STDOUT: !291 = !DILocalVariable(arg: 1, scope: !290, type: !7)
// CHECK:STDOUT: !296 = !DILocation(line: 51, column: 12, scope: !290)
// CHECK:STDOUT: !297 = !DILocation(line: 51, column: 5, scope: !290)
// CHECK:STDOUT: !298 = !{!291}
// CHECK:STDOUT: !299 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28", scope: null, file: !60, line: 147, type: !9, spFlags: DISPFlagDefinition, unit: !281, retainedNodes: !312)
// CHECK:STDOUT: !300 = !DILocalVariable(arg: 1, scope: !299, type: !7)
// CHECK:STDOUT: !305 = !DILocation(line: 148, column: 9, scope: !299)
// CHECK:STDOUT: !306 = !DILocation(line: 148, column: 8, scope: !299)
// CHECK:STDOUT: !307 = !DILocation(line: 149, column: 15, scope: !299)
// CHECK:STDOUT: !308 = !DILocation(line: 149, column: 14, scope: !299)
// CHECK:STDOUT: !309 = !DILocation(line: 149, column: 7, scope: !299)
// CHECK:STDOUT: !310 = !DILocation(line: 151, column: 14, scope: !299)
// CHECK:STDOUT: !311 = !DILocation(line: 151, column: 7, scope: !299)
// CHECK:STDOUT: !312 = !{!300}
// CHECK:STDOUT: !313 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28", scope: null, file: !60, line: 141, type: !9, spFlags: DISPFlagDefinition, unit: !281, retainedNodes: !320)
// CHECK:STDOUT: !314 = !DILocalVariable(arg: 1, scope: !313, type: !7)
// CHECK:STDOUT: !318 = !DILocation(line: 142, column: 12, scope: !313)
// CHECK:STDOUT: !319 = !DILocation(line: 142, column: 5, scope: !313)
// CHECK:STDOUT: !320 = !{!314}
// CHECK:STDOUT: !321 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28", scope: null, file: !60, line: 131, type: !9, spFlags: DISPFlagDefinition, unit: !281, retainedNodes: !334)
// CHECK:STDOUT: !322 = !DILocalVariable(arg: 1, scope: !321, type: !7)
// CHECK:STDOUT: !330 = !DILocation(line: 137, column: 5, scope: !321)
// CHECK:STDOUT: !331 = !DILocation(line: 137, column: 28, scope: !321)
// CHECK:STDOUT: !332 = !DILocation(line: 138, column: 5, scope: !321)
// CHECK:STDOUT: !333 = !DILocation(line: 139, column: 5, scope: !321)
// CHECK:STDOUT: !334 = !{!322}
// CHECK:STDOUT: !335 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28", scope: null, file: !60, line: 125, type: !26, spFlags: DISPFlagDefinition, unit: !281)
// CHECK:STDOUT: !342 = !DILocation(line: 128, column: 5, scope: !335)
// CHECK:STDOUT: !343 = !DILocation(line: 129, column: 5, scope: !335)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !346 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.c0462262ee80e8d8"(ptr %c), !dbg !351
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !352
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.c0462262ee80e8d8"(ptr %self) #0 !dbg !354 {
// CHECK:STDOUT:   ret ptr %self, !dbg !361
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!345}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !60 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !344 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !345 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !344, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !346 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !344, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !345, retainedNodes: !353)
// CHECK:STDOUT: !347 = !DILocalVariable(arg: 1, scope: !346, type: !7)
// CHECK:STDOUT: !351 = !DILocation(line: 6, column: 10, scope: !346)
// CHECK:STDOUT: !352 = !DILocation(line: 6, column: 3, scope: !346)
// CHECK:STDOUT: !353 = !{!347}
// CHECK:STDOUT: !354 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.c0462262ee80e8d8", scope: null, file: !60, line: 50, type: !9, spFlags: DISPFlagDefinition, unit: !345, retainedNodes: !362)
// CHECK:STDOUT: !355 = !DILocalVariable(arg: 1, scope: !354, type: !7)
// CHECK:STDOUT: !361 = !DILocation(line: 51, column: 5, scope: !354)
// CHECK:STDOUT: !362 = !{!355}
// CHECK:STDOUT:
