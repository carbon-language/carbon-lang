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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.210689ed411503f3(ptr %_.var.loc18), !dbg !55
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.32ee0da637ab0071(ptr %_.var.loc17), !dbg !52
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9458de9f4c4a772d(ptr %_.var.loc16), !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.53c35e954ca62889(ptr %_.var.loc15), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.210689ed411503f3(ptr %_.var.loc14), !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.32ee0da637ab0071(ptr %_.var.loc13), !dbg !40
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9458de9f4c4a772d(ptr %_.var.loc12), !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.53c35e954ca62889(ptr %_.var.loc11), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.2313f7f09b1ddaea(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70"(ptr %self), !dbg !69
// CHECK:STDOUT:   ret i1 %1, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.2313f7f09b1ddaea(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70"(ptr %self), !dbg !78
// CHECK:STDOUT:   ret ptr %1, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.787b67b8b3ba12b7"(ptr %return, i32 %self), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.787b67b8b3ba12b7(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !89 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6"(ptr %return), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.bd87f884f714419a"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !95 {
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1"(ptr %return, i32 %self), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !103 {
// CHECK:STDOUT:   call void @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.98b8f93c1b92fd53"(ptr %return, i32 %self), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.9bf3cfed350cc00b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !111 {
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0"(ptr %return, i32 %self), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.bd87f884f714419a"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !119 {
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1"(ptr %return, i32 %self), !dbg !125
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.c0697012bd57e06d"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !128 {
// CHECK:STDOUT:   call void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.bd87f884f714419a"(ptr %return, i32 %self), !dbg !134
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.9bf3cfed350cc00b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !137 {
// CHECK:STDOUT:   call void @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0"(ptr %return, i32 %self), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.3d5265f4c4ec9eb6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !146 {
// CHECK:STDOUT:   call void @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.9bf3cfed350cc00b"(ptr %return, i32 %self), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.210689ed411503f3(ptr %t) #0 !dbg !156 {
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.32ee0da637ab0071(ptr %t) #0 !dbg !163 {
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9458de9f4c4a772d(ptr %t) #0 !dbg !170 {
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.53c35e954ca62889(ptr %t) #0 !dbg !177 {
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70"(ptr %value) #0 !dbg !184 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !190
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !191
// CHECK:STDOUT:   ret i1 %2, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70"(ptr %value) #0 !dbg !194 {
// CHECK:STDOUT:   ret ptr %value, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.787b67b8b3ba12b7"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !202 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.787b67b8b3ba12b7(ptr %return, i32 %self), !dbg !208
// CHECK:STDOUT:   ret void, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !211 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !218
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !218
// CHECK:STDOUT:   ret void, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.98b8f93c1b92fd53"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !220 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !225
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !225
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.1f633674d24603c2"(i32 %self), !dbg !225
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !225
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !225
// CHECK:STDOUT:   call void @_CSome.Optional.Core.45dd3cdcb0ab47d7(ptr %return, i32 %2), !dbg !226
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e7e31ffa4cd36781(ptr %temp), !dbg !225
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.787b67b8b3ba12b7(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !229 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6"(ptr %return, i32 %value), !dbg !234
// CHECK:STDOUT:   ret void, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.1f633674d24603c2"(i32 %self) #0 !dbg !237 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self), !dbg !242
// CHECK:STDOUT:   ret i32 %1, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.45dd3cdcb0ab47d7(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !247 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.823a8f9f91058c72"(ptr %return, i32 %value), !dbg !252
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e7e31ffa4cd36781(ptr %t) #0 !dbg !255 {
// CHECK:STDOUT:   ret void, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !262 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !271
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !271
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !273
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !273
// CHECK:STDOUT:   ret void, !dbg !274
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !277 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !282
// CHECK:STDOUT:   ret i32 %1, !dbg !284
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.823a8f9f91058c72"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !286 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !295
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.2fad9574e050607e"(i32 %self), !dbg !296
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !295
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !297
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !297
// CHECK:STDOUT:   ret void, !dbg !298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !300 {
// CHECK:STDOUT:   ret i32 %self, !dbg !306
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.2fad9574e050607e"(i32 %self) #0 !dbg !309 {
// CHECK:STDOUT:   ret i32 %self, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 8, 7, 6, 5, 4, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.210689ed411503f3, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.32ee0da637ab0071, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.9458de9f4c4a772d, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.53c35e954ca62889, { 1, 0 }
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
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !4, line: 10, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !61)
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
// CHECK:STDOUT: !58 = !{null, !7}
// CHECK:STDOUT: !59 = !DISubroutineType(types: !58)
// CHECK:STDOUT: !60 = !DILocation(line: 10, column: 1, scope: !29)
// CHECK:STDOUT: !61 = !{!30, !31}
// CHECK:STDOUT: !62 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.2313f7f09b1ddaea", scope: null, file: !62, line: 36, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !71)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !7)
// CHECK:STDOUT: !69 = !DILocation(line: 37, column: 12, scope: !63)
// CHECK:STDOUT: !70 = !DILocation(line: 37, column: 5, scope: !63)
// CHECK:STDOUT: !71 = !{!64}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.2313f7f09b1ddaea", scope: null, file: !62, line: 39, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !7)
// CHECK:STDOUT: !78 = !DILocation(line: 40, column: 12, scope: !72)
// CHECK:STDOUT: !79 = !DILocation(line: 40, column: 5, scope: !72)
// CHECK:STDOUT: !80 = !{!73}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.74c31136f84330b1", scope: null, file: !62, line: 105, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !6)
// CHECK:STDOUT: !86 = !DILocation(line: 106, column: 12, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 106, column: 5, scope: !81)
// CHECK:STDOUT: !88 = !{!82}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.787b67b8b3ba12b7", scope: null, file: !62, line: 30, type: !26, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !92 = !DILocation(line: 31, column: 12, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 31, column: 5, scope: !89)
// CHECK:STDOUT: !94 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.bd87f884f714419a", scope: null, file: !94, line: 43, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !102)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !6)
// CHECK:STDOUT: !100 = !DILocation(line: 43, column: 40, scope: !95)
// CHECK:STDOUT: !101 = !DILocation(line: 43, column: 33, scope: !95)
// CHECK:STDOUT: !102 = !{!96}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.65d0ac77bdf469d0", scope: null, file: !62, line: 105, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !110)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !6)
// CHECK:STDOUT: !108 = !DILocation(line: 106, column: 12, scope: !103)
// CHECK:STDOUT: !109 = !DILocation(line: 106, column: 5, scope: !103)
// CHECK:STDOUT: !110 = !{!104}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.9bf3cfed350cc00b", scope: null, file: !94, line: 43, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !118)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !6)
// CHECK:STDOUT: !116 = !DILocation(line: 43, column: 40, scope: !111)
// CHECK:STDOUT: !117 = !DILocation(line: 43, column: 33, scope: !111)
// CHECK:STDOUT: !118 = !{!112}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.bd87f884f714419a", scope: null, file: !94, line: 47, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !127)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !6)
// CHECK:STDOUT: !125 = !DILocation(line: 47, column: 34, scope: !119)
// CHECK:STDOUT: !126 = !DILocation(line: 47, column: 27, scope: !119)
// CHECK:STDOUT: !127 = !{!120}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.c0697012bd57e06d", scope: null, file: !94, line: 47, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !136)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !6)
// CHECK:STDOUT: !134 = !DILocation(line: 47, column: 34, scope: !128)
// CHECK:STDOUT: !135 = !DILocation(line: 47, column: 27, scope: !128)
// CHECK:STDOUT: !136 = !{!129}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.9bf3cfed350cc00b", scope: null, file: !94, line: 47, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !145)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !6)
// CHECK:STDOUT: !143 = !DILocation(line: 47, column: 34, scope: !137)
// CHECK:STDOUT: !144 = !DILocation(line: 47, column: 27, scope: !137)
// CHECK:STDOUT: !145 = !{!138}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.aa801aecfd35fc2d:ImplicitAs.eb057aa32837c84e.Core.3d5265f4c4ec9eb6", scope: null, file: !94, line: 47, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !154)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !6)
// CHECK:STDOUT: !152 = !DILocation(line: 47, column: 34, scope: !146)
// CHECK:STDOUT: !153 = !DILocation(line: 47, column: 27, scope: !146)
// CHECK:STDOUT: !154 = !{!147}
// CHECK:STDOUT: !155 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.210689ed411503f3", scope: null, file: !155, line: 35, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !162)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !7)
// CHECK:STDOUT: !161 = !DILocation(line: 35, column: 1, scope: !156)
// CHECK:STDOUT: !162 = !{!157}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.32ee0da637ab0071", scope: null, file: !155, line: 35, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !7)
// CHECK:STDOUT: !168 = !DILocation(line: 35, column: 1, scope: !163)
// CHECK:STDOUT: !169 = !{!164}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9458de9f4c4a772d", scope: null, file: !155, line: 35, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !176)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !7)
// CHECK:STDOUT: !175 = !DILocation(line: 35, column: 1, scope: !170)
// CHECK:STDOUT: !176 = !{!171}
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.53c35e954ca62889", scope: null, file: !155, line: 35, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !183)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !177, type: !7)
// CHECK:STDOUT: !182 = !DILocation(line: 35, column: 1, scope: !177)
// CHECK:STDOUT: !183 = !{!178}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70", scope: null, file: !62, line: 172, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !193)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !7)
// CHECK:STDOUT: !190 = !DILocation(line: 173, column: 16, scope: !184)
// CHECK:STDOUT: !191 = !DILocation(line: 173, column: 12, scope: !184)
// CHECK:STDOUT: !192 = !DILocation(line: 173, column: 5, scope: !184)
// CHECK:STDOUT: !193 = !{!185}
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.55456748a8a2eb70", scope: null, file: !62, line: 175, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !201)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !7)
// CHECK:STDOUT: !200 = !DILocation(line: 176, column: 5, scope: !194)
// CHECK:STDOUT: !201 = !{!195}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.787b67b8b3ba12b7", scope: null, file: !62, line: 80, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !210)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !202, type: !6)
// CHECK:STDOUT: !208 = !DILocation(line: 81, column: 12, scope: !202)
// CHECK:STDOUT: !209 = !DILocation(line: 81, column: 5, scope: !202)
// CHECK:STDOUT: !210 = !{!203}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6", scope: null, file: !62, line: 125, type: !26, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !218 = !DILocation(line: 128, column: 5, scope: !211)
// CHECK:STDOUT: !219 = !DILocation(line: 129, column: 5, scope: !211)
// CHECK:STDOUT: !220 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.56158d26cb64337a:OptionalAs.5cc57aaa5e35eede.Core.98b8f93c1b92fd53", scope: null, file: !62, line: 87, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !228)
// CHECK:STDOUT: !221 = !DILocalVariable(arg: 1, scope: !220, type: !6)
// CHECK:STDOUT: !225 = !DILocation(line: 88, column: 29, scope: !220)
// CHECK:STDOUT: !226 = !DILocation(line: 88, column: 12, scope: !220)
// CHECK:STDOUT: !227 = !DILocation(line: 88, column: 5, scope: !220)
// CHECK:STDOUT: !228 = !{!221}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.787b67b8b3ba12b7", scope: null, file: !62, line: 33, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !236)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !6)
// CHECK:STDOUT: !234 = !DILocation(line: 34, column: 12, scope: !229)
// CHECK:STDOUT: !235 = !DILocation(line: 34, column: 5, scope: !229)
// CHECK:STDOUT: !236 = !{!230}
// CHECK:STDOUT: !237 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef4ef968855d2979:ImplicitAs.ad22d1bbc0605210.Core.1f633674d24603c2", scope: null, file: !94, line: 43, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !246)
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 1, scope: !237, type: !6)
// CHECK:STDOUT: !242 = !DILocation(line: 43, column: 40, scope: !237)
// CHECK:STDOUT: !243 = !{!6, !6}
// CHECK:STDOUT: !244 = !DISubroutineType(types: !243)
// CHECK:STDOUT: !245 = !DILocation(line: 43, column: 33, scope: !237)
// CHECK:STDOUT: !246 = !{!238}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.45dd3cdcb0ab47d7", scope: null, file: !62, line: 33, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !254)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !6)
// CHECK:STDOUT: !252 = !DILocation(line: 34, column: 12, scope: !247)
// CHECK:STDOUT: !253 = !DILocation(line: 34, column: 5, scope: !247)
// CHECK:STDOUT: !254 = !{!248}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e7e31ffa4cd36781", scope: null, file: !155, line: 35, type: !59, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !261)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !7)
// CHECK:STDOUT: !260 = !DILocation(line: 35, column: 1, scope: !255)
// CHECK:STDOUT: !261 = !{!256}
// CHECK:STDOUT: !262 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.de74407914bccca6", scope: null, file: !62, line: 131, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !275)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 1, scope: !262, type: !6)
// CHECK:STDOUT: !271 = !DILocation(line: 137, column: 5, scope: !262)
// CHECK:STDOUT: !273 = !DILocation(line: 138, column: 5, scope: !262)
// CHECK:STDOUT: !274 = !DILocation(line: 139, column: 5, scope: !262)
// CHECK:STDOUT: !275 = !{!263}
// CHECK:STDOUT: !276 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !277 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !276, line: 64, type: !244, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !285)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !277, type: !6)
// CHECK:STDOUT: !282 = !DILocation(line: 66, column: 17, scope: !277)
// CHECK:STDOUT: !284 = !DILocation(line: 66, column: 5, scope: !277)
// CHECK:STDOUT: !285 = !{!278}
// CHECK:STDOUT: !286 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.823a8f9f91058c72", scope: null, file: !62, line: 131, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !299)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 1, scope: !286, type: !6)
// CHECK:STDOUT: !295 = !DILocation(line: 137, column: 5, scope: !286)
// CHECK:STDOUT: !296 = !DILocation(line: 137, column: 28, scope: !286)
// CHECK:STDOUT: !297 = !DILocation(line: 138, column: 5, scope: !286)
// CHECK:STDOUT: !298 = !DILocation(line: 139, column: 5, scope: !286)
// CHECK:STDOUT: !299 = !{!287}
// CHECK:STDOUT: !300 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !276, line: 59, type: !244, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !307)
// CHECK:STDOUT: !301 = !DILocalVariable(arg: 1, scope: !300, type: !6)
// CHECK:STDOUT: !306 = !DILocation(line: 59, column: 28, scope: !300)
// CHECK:STDOUT: !307 = !{!301}
// CHECK:STDOUT: !308 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !309 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.2fad9574e050607e", scope: null, file: !308, line: 23, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !317)
// CHECK:STDOUT: !310 = !DILocalVariable(arg: 1, scope: !309, type: !6)
// CHECK:STDOUT: !316 = !DILocation(line: 23, column: 25, scope: !309)
// CHECK:STDOUT: !317 = !{!310}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !320 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.60b535298816170e"(ptr %return, ptr %c), !dbg !325
// CHECK:STDOUT:   ret void, !dbg !326
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.60b535298816170e"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !328 {
// CHECK:STDOUT:   call void @"_CCopy.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %return, ptr %self), !dbg !334
// CHECK:STDOUT:   ret void, !dbg !335
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !337 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %value), !dbg !343
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !344
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !345
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %return, ptr %value1), !dbg !346
// CHECK:STDOUT:   ret void, !dbg !347
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %return), !dbg !348
// CHECK:STDOUT:   ret void, !dbg !349
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr %value) #0 !dbg !351 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !356
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !356
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !356
// CHECK:STDOUT:   ret i1 %2, !dbg !357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !359 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !368
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !369
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !370
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !370
// CHECK:STDOUT:   ret void, !dbg !371
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28"(ptr sret({ {}, i1 }) %return) #0 !dbg !373 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !380
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !380
// CHECK:STDOUT:   ret void, !dbg !381
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!319}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !25 = !{!7}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !62 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !318 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !319 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !318, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !320 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !318, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !319, retainedNodes: !327)
// CHECK:STDOUT: !321 = !DILocalVariable(arg: 1, scope: !320, type: !7)
// CHECK:STDOUT: !325 = !DILocation(line: 11, column: 10, scope: !320)
// CHECK:STDOUT: !326 = !DILocation(line: 11, column: 3, scope: !320)
// CHECK:STDOUT: !327 = !{!321}
// CHECK:STDOUT: !328 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.60b535298816170e", scope: null, file: !62, line: 50, type: !9, spFlags: DISPFlagDefinition, unit: !319, retainedNodes: !336)
// CHECK:STDOUT: !329 = !DILocalVariable(arg: 1, scope: !328, type: !7)
// CHECK:STDOUT: !334 = !DILocation(line: 51, column: 12, scope: !328)
// CHECK:STDOUT: !335 = !DILocation(line: 51, column: 5, scope: !328)
// CHECK:STDOUT: !336 = !{!329}
// CHECK:STDOUT: !337 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28", scope: null, file: !62, line: 147, type: !9, spFlags: DISPFlagDefinition, unit: !319, retainedNodes: !350)
// CHECK:STDOUT: !338 = !DILocalVariable(arg: 1, scope: !337, type: !7)
// CHECK:STDOUT: !343 = !DILocation(line: 148, column: 9, scope: !337)
// CHECK:STDOUT: !344 = !DILocation(line: 148, column: 8, scope: !337)
// CHECK:STDOUT: !345 = !DILocation(line: 149, column: 15, scope: !337)
// CHECK:STDOUT: !346 = !DILocation(line: 149, column: 14, scope: !337)
// CHECK:STDOUT: !347 = !DILocation(line: 149, column: 7, scope: !337)
// CHECK:STDOUT: !348 = !DILocation(line: 151, column: 14, scope: !337)
// CHECK:STDOUT: !349 = !DILocation(line: 151, column: 7, scope: !337)
// CHECK:STDOUT: !350 = !{!338}
// CHECK:STDOUT: !351 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28", scope: null, file: !62, line: 141, type: !9, spFlags: DISPFlagDefinition, unit: !319, retainedNodes: !358)
// CHECK:STDOUT: !352 = !DILocalVariable(arg: 1, scope: !351, type: !7)
// CHECK:STDOUT: !356 = !DILocation(line: 142, column: 12, scope: !351)
// CHECK:STDOUT: !357 = !DILocation(line: 142, column: 5, scope: !351)
// CHECK:STDOUT: !358 = !{!352}
// CHECK:STDOUT: !359 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28", scope: null, file: !62, line: 131, type: !9, spFlags: DISPFlagDefinition, unit: !319, retainedNodes: !372)
// CHECK:STDOUT: !360 = !DILocalVariable(arg: 1, scope: !359, type: !7)
// CHECK:STDOUT: !368 = !DILocation(line: 137, column: 5, scope: !359)
// CHECK:STDOUT: !369 = !DILocation(line: 137, column: 28, scope: !359)
// CHECK:STDOUT: !370 = !DILocation(line: 138, column: 5, scope: !359)
// CHECK:STDOUT: !371 = !DILocation(line: 139, column: 5, scope: !359)
// CHECK:STDOUT: !372 = !{!360}
// CHECK:STDOUT: !373 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.0cee3c1ed1876f28", scope: null, file: !62, line: 125, type: !26, spFlags: DISPFlagDefinition, unit: !319)
// CHECK:STDOUT: !380 = !DILocation(line: 128, column: 5, scope: !373)
// CHECK:STDOUT: !381 = !DILocation(line: 129, column: 5, scope: !373)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !384 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.c0462262ee80e8d8"(ptr %c), !dbg !389
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !390
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.c0462262ee80e8d8"(ptr %self) #0 !dbg !392 {
// CHECK:STDOUT:   ret ptr %self, !dbg !399
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!383}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !62 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !382 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !383 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !382, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !384 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !382, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !383, retainedNodes: !391)
// CHECK:STDOUT: !385 = !DILocalVariable(arg: 1, scope: !384, type: !7)
// CHECK:STDOUT: !389 = !DILocation(line: 6, column: 10, scope: !384)
// CHECK:STDOUT: !390 = !DILocation(line: 6, column: 3, scope: !384)
// CHECK:STDOUT: !391 = !{!385}
// CHECK:STDOUT: !392 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.c0462262ee80e8d8", scope: null, file: !62, line: 50, type: !9, spFlags: DISPFlagDefinition, unit: !383, retainedNodes: !400)
// CHECK:STDOUT: !393 = !DILocalVariable(arg: 1, scope: !392, type: !7)
// CHECK:STDOUT: !399 = !DILocation(line: 51, column: 5, scope: !392)
// CHECK:STDOUT: !400 = !{!393}
// CHECK:STDOUT:
