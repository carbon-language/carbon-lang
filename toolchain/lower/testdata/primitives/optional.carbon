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
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %o), !dbg !19
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %o), !dbg !21
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !22
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c99b25630ca3d3c8"(ptr %return, i32 %.loc5_12.2), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.ec49bcf7e797ddc0(ptr %return), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 4, !dbg !36
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 4, !dbg !39
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 4, !dbg !42
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 4, !dbg !45
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 4, !dbg !48
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 4, !dbg !51
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 4, !dbg !54
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 4, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !36
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c99b25630ca3d3c8"(ptr %_.var.loc11, i32 %a), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !39
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.203fe310bfdcf0eb"(ptr %_.var.loc12, i32 %a), !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !42
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c0adcd1fb18a00c5"(ptr %_.var.loc13, i32 %a), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !45
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7d39ed218f977539"(ptr %_.var.loc14, i32 %a), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !48
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.203fe310bfdcf0eb"(ptr %_.var.loc15, i32 %b), !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !51
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.adf490e1ba902af7"(ptr %_.var.loc16, i32 %b), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !54
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7d39ed218f977539"(ptr %_.var.loc17, i32 %b), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !57
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.aa31cbde29083ea4"(ptr %_.var.loc18, i32 %b), !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5752b0830759cc30(ptr %_.var.loc18), !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e6482dadb6de45ec(ptr %_.var.loc17), !dbg !54
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d3b02c0c9e01b7f4(ptr %_.var.loc16), !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1943cd2c7dcb2db1(ptr %_.var.loc15), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5752b0830759cc30(ptr %_.var.loc14), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e6482dadb6de45ec(ptr %_.var.loc13), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d3b02c0c9e01b7f4(ptr %_.var.loc12), !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1943cd2c7dcb2db1(ptr %_.var.loc11), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6a8d754fe1453eab:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.96a347e4ca9c10e1:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0de8725637a99ce1:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !72
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d9c498a484a58446(ptr %.loc18_3.1.has_value), !dbg !72
// CHECK:STDOUT:   %.loc18_3.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !72
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3bd7402296fb9888(ptr %.loc18_3.4.value), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3d9277fd2d565dd0:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !76
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d9c498a484a58446(ptr %.loc18_3.1.has_value), !dbg !76
// CHECK:STDOUT:   %.loc18_3.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !76
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3bd7402296fb9888(ptr %.loc18_3.4.value), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1863a21e4fe31075:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1d6a7672bf2f14c6:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.11b8dc3dc9734745:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d9c498a484a58446(ptr %.loc16_3.1.has_value), !dbg !88
// CHECK:STDOUT:   %.loc16_3.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.fcb3d1699ab8a7ba(ptr %.loc16_3.4.value), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e2c6afbaf9a34a7e:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d9c498a484a58446(ptr %.loc16_3.1.has_value), !dbg !94
// CHECK:STDOUT:   %.loc16_3.4.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.fcb3d1699ab8a7ba(ptr %.loc16_3.4.value), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6318e40ee34dd117:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !107
// CHECK:STDOUT:   ret i1 %1, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !110 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !116
// CHECK:STDOUT:   ret ptr %1, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c99b25630ca3d3c8"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !119 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.ec49bcf7e797ddc0"(ptr %return, i32 %self), !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.ec49bcf7e797ddc0(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !127 {
// CHECK:STDOUT:   call void @"_CNone.19c24483098064e0:OptionalStorage.Core.23caa3c38980dd74"(ptr %return), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.203fe310bfdcf0eb"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !133 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c99b25630ca3d3c8"(ptr %return, i32 %self), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c0adcd1fb18a00c5"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !141 {
// CHECK:STDOUT:   call void @"_CConvert.05aac2c2cbdd0ac3:OptionalAs.f29bb853b5a0796c.Core.dfedd876942d0982"(ptr %return, i32 %self), !dbg !146
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7d39ed218f977539"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !149 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c0adcd1fb18a00c5"(ptr %return, i32 %self), !dbg !154
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.203fe310bfdcf0eb"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !157 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c99b25630ca3d3c8"(ptr %return, i32 %self), !dbg !163
// CHECK:STDOUT:   ret void, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.adf490e1ba902af7"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !166 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.203fe310bfdcf0eb"(ptr %return, i32 %self), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7d39ed218f977539"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !175 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c0adcd1fb18a00c5"(ptr %return, i32 %self), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.aa31cbde29083ea4"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !184 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7d39ed218f977539"(ptr %return, i32 %self), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5752b0830759cc30(ptr %t) #0 !dbg !194 {
// CHECK:STDOUT:   call void @"_COp.1863a21e4fe31075:core.SubobjectDestroy.Core"(ptr %t), !dbg !198
// CHECK:STDOUT:   ret void, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e6482dadb6de45ec(ptr %t) #0 !dbg !201 {
// CHECK:STDOUT:   ret void, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d3b02c0c9e01b7f4(ptr %t) #0 !dbg !208 {
// CHECK:STDOUT:   call void @"_COp.6318e40ee34dd117:core.SubobjectDestroy.Core"(ptr %t), !dbg !212
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1943cd2c7dcb2db1(ptr %t) #0 !dbg !215 {
// CHECK:STDOUT:   ret void, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d9c498a484a58446(ptr %t) #0 !dbg !222 {
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3bd7402296fb9888(ptr %t) #0 !dbg !229 {
// CHECK:STDOUT:   ret void, !dbg !234
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.fcb3d1699ab8a7ba(ptr %t) #0 !dbg !236 {
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !243 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !249
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !250
// CHECK:STDOUT:   ret i1 %2, !dbg !251
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !253 {
// CHECK:STDOUT:   ret ptr %value, !dbg !259
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.ec49bcf7e797ddc0"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !261 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.ec49bcf7e797ddc0(ptr %return, i32 %self), !dbg !267
// CHECK:STDOUT:   ret void, !dbg !268
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.19c24483098064e0:OptionalStorage.Core.23caa3c38980dd74"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !270 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !277
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !277
// CHECK:STDOUT:   ret void, !dbg !278
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.05aac2c2cbdd0ac3:OptionalAs.f29bb853b5a0796c.Core.dfedd876942d0982"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !279 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !284
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !284
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self), !dbg !284
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !284
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !284
// CHECK:STDOUT:   call void @_CSome.Optional.Core.ad458686a33e8f93(ptr %return, i32 %2), !dbg !285
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.25b58ef0e135a985(ptr %temp), !dbg !284
// CHECK:STDOUT:   ret void, !dbg !286
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.ec49bcf7e797ddc0(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !288 {
// CHECK:STDOUT:   call void @"_CSome.19c24483098064e0:OptionalStorage.Core.23caa3c38980dd74"(ptr %return, i32 %value), !dbg !293
// CHECK:STDOUT:   ret void, !dbg !294
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self) #0 !dbg !296 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self), !dbg !301
// CHECK:STDOUT:   ret i32 %1, !dbg !304
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.ad458686a33e8f93(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !306 {
// CHECK:STDOUT:   call void @"_CSome.19c24483098064e0:OptionalStorage.Core.226d23e64cca9429"(ptr %return, i32 %value), !dbg !311
// CHECK:STDOUT:   ret void, !dbg !312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.25b58ef0e135a985(ptr %t) #0 !dbg !314 {
// CHECK:STDOUT:   call void @"_COp.6a8d754fe1453eab:core.SubobjectDestroy.Core"(ptr %t), !dbg !318
// CHECK:STDOUT:   ret void, !dbg !319
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.19c24483098064e0:OptionalStorage.Core.23caa3c38980dd74"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !321 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !330
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !330
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !332
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !332
// CHECK:STDOUT:   ret void, !dbg !333
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !336 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !341
// CHECK:STDOUT:   ret i32 %1, !dbg !343
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.19c24483098064e0:OptionalStorage.Core.226d23e64cca9429"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !345 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !354
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self), !dbg !355
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !354
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !356
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !356
// CHECK:STDOUT:   ret void, !dbg !357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !359 {
// CHECK:STDOUT:   ret i32 %self, !dbg !365
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self) #0 !dbg !368 {
// CHECK:STDOUT:   ret i32 %self, !dbg !375
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c99b25630ca3d3c8", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 8, 7, 6, 5, 4, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.5752b0830759cc30, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.e6482dadb6de45ec, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.d3b02c0c9e01b7f4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.1943cd2c7dcb2db1, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.d9c498a484a58446, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3bd7402296fb9888, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.fcb3d1699ab8a7ba, { 1, 0 }
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
// CHECK:STDOUT: !12 = !{null, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !4, line: 3, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !14, type: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 4, column: 7, scope: !14)
// CHECK:STDOUT: !20 = !DILocation(line: 4, column: 6, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 5, column: 13, scope: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 5, column: 12, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 5, column: 5, scope: !14)
// CHECK:STDOUT: !24 = !{!7, !6}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 7, column: 10, scope: !14)
// CHECK:STDOUT: !27 = !{!7}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 7, column: 3, scope: !14)
// CHECK:STDOUT: !30 = !{!15}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !4, line: 10, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !61)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !6)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !31, type: !6)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 3, scope: !31)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 3, scope: !31)
// CHECK:STDOUT: !42 = !DILocation(line: 13, column: 3, scope: !31)
// CHECK:STDOUT: !45 = !DILocation(line: 14, column: 3, scope: !31)
// CHECK:STDOUT: !48 = !DILocation(line: 15, column: 3, scope: !31)
// CHECK:STDOUT: !51 = !DILocation(line: 16, column: 3, scope: !31)
// CHECK:STDOUT: !54 = !DILocation(line: 17, column: 3, scope: !31)
// CHECK:STDOUT: !57 = !DILocation(line: 18, column: 3, scope: !31)
// CHECK:STDOUT: !60 = !DILocation(line: 10, column: 1, scope: !31)
// CHECK:STDOUT: !61 = !{!32, !33}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6a8d754fe1453eab:core.SubobjectDestroy.Core", scope: null, file: !4, line: 13, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !7)
// CHECK:STDOUT: !64 = !DILocation(line: 13, column: 10, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Op", linkageName: "_COp.96a347e4ca9c10e1:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !7)
// CHECK:STDOUT: !68 = !DILocation(line: 18, column: 3, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0de8725637a99ce1:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !7)
// CHECK:STDOUT: !72 = !DILocation(line: 18, column: 3, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3d9277fd2d565dd0:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !77)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !7)
// CHECK:STDOUT: !76 = !DILocation(line: 18, column: 3, scope: !74)
// CHECK:STDOUT: !77 = !{!75}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1863a21e4fe31075:core.SubobjectDestroy.Core", scope: null, file: !4, line: 18, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !7)
// CHECK:STDOUT: !80 = !DILocation(line: 18, column: 3, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1d6a7672bf2f14c6:core.SubobjectDestroy.Core", scope: null, file: !4, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !7)
// CHECK:STDOUT: !84 = !DILocation(line: 16, column: 3, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp.11b8dc3dc9734745:core.SubobjectDestroy.Core", scope: null, file: !4, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !91)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !7)
// CHECK:STDOUT: !88 = !DILocation(line: 16, column: 3, scope: !86)
// CHECK:STDOUT: !89 = !{null, !6}
// CHECK:STDOUT: !90 = !DISubroutineType(types: !89)
// CHECK:STDOUT: !91 = !{!87}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e2c6afbaf9a34a7e:core.SubobjectDestroy.Core", scope: null, file: !4, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !7)
// CHECK:STDOUT: !94 = !DILocation(line: 16, column: 3, scope: !92)
// CHECK:STDOUT: !95 = !{!93}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6318e40ee34dd117:core.SubobjectDestroy.Core", scope: null, file: !4, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !7)
// CHECK:STDOUT: !98 = !DILocation(line: 16, column: 3, scope: !96)
// CHECK:STDOUT: !99 = !{!97}
// CHECK:STDOUT: !100 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.6e9e957697159b53", scope: null, file: !100, line: 36, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !109)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 37, column: 12, scope: !101)
// CHECK:STDOUT: !108 = !DILocation(line: 37, column: 5, scope: !101)
// CHECK:STDOUT: !109 = !{!102}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.6e9e957697159b53", scope: null, file: !100, line: 39, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !118)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !7)
// CHECK:STDOUT: !116 = !DILocation(line: 40, column: 12, scope: !110)
// CHECK:STDOUT: !117 = !DILocation(line: 40, column: 5, scope: !110)
// CHECK:STDOUT: !118 = !{!111}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c99b25630ca3d3c8", scope: null, file: !100, line: 105, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !126)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !6)
// CHECK:STDOUT: !124 = !DILocation(line: 106, column: 12, scope: !119)
// CHECK:STDOUT: !125 = !DILocation(line: 106, column: 5, scope: !119)
// CHECK:STDOUT: !126 = !{!120}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.ec49bcf7e797ddc0", scope: null, file: !100, line: 30, type: !28, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !130 = !DILocation(line: 31, column: 12, scope: !127)
// CHECK:STDOUT: !131 = !DILocation(line: 31, column: 5, scope: !127)
// CHECK:STDOUT: !132 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.203fe310bfdcf0eb", scope: null, file: !132, line: 43, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !6)
// CHECK:STDOUT: !138 = !DILocation(line: 43, column: 40, scope: !133)
// CHECK:STDOUT: !139 = !DILocation(line: 43, column: 33, scope: !133)
// CHECK:STDOUT: !140 = !{!134}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.c0adcd1fb18a00c5", scope: null, file: !100, line: 105, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !6)
// CHECK:STDOUT: !146 = !DILocation(line: 106, column: 12, scope: !141)
// CHECK:STDOUT: !147 = !DILocation(line: 106, column: 5, scope: !141)
// CHECK:STDOUT: !148 = !{!142}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7d39ed218f977539", scope: null, file: !132, line: 43, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !6)
// CHECK:STDOUT: !154 = !DILocation(line: 43, column: 40, scope: !149)
// CHECK:STDOUT: !155 = !DILocation(line: 43, column: 33, scope: !149)
// CHECK:STDOUT: !156 = !{!150}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.203fe310bfdcf0eb", scope: null, file: !132, line: 47, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !165)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !6)
// CHECK:STDOUT: !163 = !DILocation(line: 47, column: 34, scope: !157)
// CHECK:STDOUT: !164 = !DILocation(line: 47, column: 27, scope: !157)
// CHECK:STDOUT: !165 = !{!158}
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.adf490e1ba902af7", scope: null, file: !132, line: 47, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !174)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !166, type: !6)
// CHECK:STDOUT: !172 = !DILocation(line: 47, column: 34, scope: !166)
// CHECK:STDOUT: !173 = !DILocation(line: 47, column: 27, scope: !166)
// CHECK:STDOUT: !174 = !{!167}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7d39ed218f977539", scope: null, file: !132, line: 47, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !183)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !6)
// CHECK:STDOUT: !181 = !DILocation(line: 47, column: 34, scope: !175)
// CHECK:STDOUT: !182 = !DILocation(line: 47, column: 27, scope: !175)
// CHECK:STDOUT: !183 = !{!176}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.aa31cbde29083ea4", scope: null, file: !132, line: 47, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !192)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !6)
// CHECK:STDOUT: !190 = !DILocation(line: 47, column: 34, scope: !184)
// CHECK:STDOUT: !191 = !DILocation(line: 47, column: 27, scope: !184)
// CHECK:STDOUT: !192 = !{!185}
// CHECK:STDOUT: !193 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5752b0830759cc30", scope: null, file: !193, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !200)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !7)
// CHECK:STDOUT: !198 = !DILocation(line: 32, column: 5, scope: !194)
// CHECK:STDOUT: !199 = !DILocation(line: 30, column: 1, scope: !194)
// CHECK:STDOUT: !200 = !{!195}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e6482dadb6de45ec", scope: null, file: !193, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !207)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !7)
// CHECK:STDOUT: !206 = !DILocation(line: 30, column: 1, scope: !201)
// CHECK:STDOUT: !207 = !{!202}
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d3b02c0c9e01b7f4", scope: null, file: !193, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !214)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 1, scope: !208, type: !7)
// CHECK:STDOUT: !212 = !DILocation(line: 32, column: 5, scope: !208)
// CHECK:STDOUT: !213 = !DILocation(line: 30, column: 1, scope: !208)
// CHECK:STDOUT: !214 = !{!209}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1943cd2c7dcb2db1", scope: null, file: !193, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !221)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !7)
// CHECK:STDOUT: !220 = !DILocation(line: 30, column: 1, scope: !215)
// CHECK:STDOUT: !221 = !{!216}
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d9c498a484a58446", scope: null, file: !193, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !228)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 1, scope: !222, type: !7)
// CHECK:STDOUT: !227 = !DILocation(line: 30, column: 1, scope: !222)
// CHECK:STDOUT: !228 = !{!223}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3bd7402296fb9888", scope: null, file: !193, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !235)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !7)
// CHECK:STDOUT: !234 = !DILocation(line: 30, column: 1, scope: !229)
// CHECK:STDOUT: !235 = !{!230}
// CHECK:STDOUT: !236 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.fcb3d1699ab8a7ba", scope: null, file: !193, line: 30, type: !90, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !242)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 1, scope: !236, type: !6)
// CHECK:STDOUT: !241 = !DILocation(line: 30, column: 1, scope: !236)
// CHECK:STDOUT: !242 = !{!237}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !100, line: 172, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !252)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !7)
// CHECK:STDOUT: !249 = !DILocation(line: 173, column: 16, scope: !243)
// CHECK:STDOUT: !250 = !DILocation(line: 173, column: 12, scope: !243)
// CHECK:STDOUT: !251 = !DILocation(line: 173, column: 5, scope: !243)
// CHECK:STDOUT: !252 = !{!244}
// CHECK:STDOUT: !253 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !100, line: 175, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !260)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 1, scope: !253, type: !7)
// CHECK:STDOUT: !259 = !DILocation(line: 176, column: 5, scope: !253)
// CHECK:STDOUT: !260 = !{!254}
// CHECK:STDOUT: !261 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.ec49bcf7e797ddc0", scope: null, file: !100, line: 80, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !269)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 1, scope: !261, type: !6)
// CHECK:STDOUT: !267 = !DILocation(line: 81, column: 12, scope: !261)
// CHECK:STDOUT: !268 = !DILocation(line: 81, column: 5, scope: !261)
// CHECK:STDOUT: !269 = !{!262}
// CHECK:STDOUT: !270 = distinct !DISubprogram(name: "None", linkageName: "_CNone.19c24483098064e0:OptionalStorage.Core.23caa3c38980dd74", scope: null, file: !100, line: 125, type: !28, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !277 = !DILocation(line: 128, column: 5, scope: !270)
// CHECK:STDOUT: !278 = !DILocation(line: 129, column: 5, scope: !270)
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.05aac2c2cbdd0ac3:OptionalAs.f29bb853b5a0796c.Core.dfedd876942d0982", scope: null, file: !100, line: 87, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !287)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !6)
// CHECK:STDOUT: !284 = !DILocation(line: 88, column: 29, scope: !279)
// CHECK:STDOUT: !285 = !DILocation(line: 88, column: 12, scope: !279)
// CHECK:STDOUT: !286 = !DILocation(line: 88, column: 5, scope: !279)
// CHECK:STDOUT: !287 = !{!280}
// CHECK:STDOUT: !288 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.ec49bcf7e797ddc0", scope: null, file: !100, line: 33, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !295)
// CHECK:STDOUT: !289 = !DILocalVariable(arg: 1, scope: !288, type: !6)
// CHECK:STDOUT: !293 = !DILocation(line: 34, column: 12, scope: !288)
// CHECK:STDOUT: !294 = !DILocation(line: 34, column: 5, scope: !288)
// CHECK:STDOUT: !295 = !{!289}
// CHECK:STDOUT: !296 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2", scope: null, file: !132, line: 43, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !305)
// CHECK:STDOUT: !297 = !DILocalVariable(arg: 1, scope: !296, type: !6)
// CHECK:STDOUT: !301 = !DILocation(line: 43, column: 40, scope: !296)
// CHECK:STDOUT: !302 = !{!6, !6}
// CHECK:STDOUT: !303 = !DISubroutineType(types: !302)
// CHECK:STDOUT: !304 = !DILocation(line: 43, column: 33, scope: !296)
// CHECK:STDOUT: !305 = !{!297}
// CHECK:STDOUT: !306 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.ad458686a33e8f93", scope: null, file: !100, line: 33, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !313)
// CHECK:STDOUT: !307 = !DILocalVariable(arg: 1, scope: !306, type: !6)
// CHECK:STDOUT: !311 = !DILocation(line: 34, column: 12, scope: !306)
// CHECK:STDOUT: !312 = !DILocation(line: 34, column: 5, scope: !306)
// CHECK:STDOUT: !313 = !{!307}
// CHECK:STDOUT: !314 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.25b58ef0e135a985", scope: null, file: !193, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !320)
// CHECK:STDOUT: !315 = !DILocalVariable(arg: 1, scope: !314, type: !7)
// CHECK:STDOUT: !318 = !DILocation(line: 32, column: 5, scope: !314)
// CHECK:STDOUT: !319 = !DILocation(line: 30, column: 1, scope: !314)
// CHECK:STDOUT: !320 = !{!315}
// CHECK:STDOUT: !321 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.19c24483098064e0:OptionalStorage.Core.23caa3c38980dd74", scope: null, file: !100, line: 131, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !334)
// CHECK:STDOUT: !322 = !DILocalVariable(arg: 1, scope: !321, type: !6)
// CHECK:STDOUT: !330 = !DILocation(line: 137, column: 5, scope: !321)
// CHECK:STDOUT: !332 = !DILocation(line: 138, column: 5, scope: !321)
// CHECK:STDOUT: !333 = !DILocation(line: 139, column: 5, scope: !321)
// CHECK:STDOUT: !334 = !{!322}
// CHECK:STDOUT: !335 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !336 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !335, line: 64, type: !303, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !344)
// CHECK:STDOUT: !337 = !DILocalVariable(arg: 1, scope: !336, type: !6)
// CHECK:STDOUT: !341 = !DILocation(line: 66, column: 17, scope: !336)
// CHECK:STDOUT: !343 = !DILocation(line: 66, column: 5, scope: !336)
// CHECK:STDOUT: !344 = !{!337}
// CHECK:STDOUT: !345 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.19c24483098064e0:OptionalStorage.Core.226d23e64cca9429", scope: null, file: !100, line: 131, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !358)
// CHECK:STDOUT: !346 = !DILocalVariable(arg: 1, scope: !345, type: !6)
// CHECK:STDOUT: !354 = !DILocation(line: 137, column: 5, scope: !345)
// CHECK:STDOUT: !355 = !DILocation(line: 137, column: 28, scope: !345)
// CHECK:STDOUT: !356 = !DILocation(line: 138, column: 5, scope: !345)
// CHECK:STDOUT: !357 = !DILocation(line: 139, column: 5, scope: !345)
// CHECK:STDOUT: !358 = !{!346}
// CHECK:STDOUT: !359 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !335, line: 59, type: !303, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !366)
// CHECK:STDOUT: !360 = !DILocalVariable(arg: 1, scope: !359, type: !6)
// CHECK:STDOUT: !365 = !DILocation(line: 59, column: 28, scope: !359)
// CHECK:STDOUT: !366 = !{!360}
// CHECK:STDOUT: !367 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !368 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4", scope: null, file: !367, line: 23, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !376)
// CHECK:STDOUT: !369 = !DILocalVariable(arg: 1, scope: !368, type: !6)
// CHECK:STDOUT: !375 = !DILocation(line: 23, column: 25, scope: !368)
// CHECK:STDOUT: !376 = !{!369}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !379 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.0fa40e8d9e2a73f6"(ptr %return, ptr %c), !dbg !384
// CHECK:STDOUT:   ret void, !dbg !385
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.0fa40e8d9e2a73f6"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !387 {
// CHECK:STDOUT:   call void @"_CCopy.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286"(ptr %return, ptr %self), !dbg !393
// CHECK:STDOUT:   ret void, !dbg !394
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !396 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286"(ptr %value), !dbg !402
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !403
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !404
// CHECK:STDOUT:   call void @"_CSome.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286"(ptr %return, ptr %value1), !dbg !405
// CHECK:STDOUT:   ret void, !dbg !406
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286"(ptr %return), !dbg !407
// CHECK:STDOUT:   ret void, !dbg !408
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286"(ptr %value) #0 !dbg !410 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !415
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !415
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !415
// CHECK:STDOUT:   ret i1 %2, !dbg !416
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !418 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !427
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !428
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !429
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !429
// CHECK:STDOUT:   ret void, !dbg !430
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286"(ptr sret({ {}, i1 }) %return) #0 !dbg !432 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !439
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !439
// CHECK:STDOUT:   ret void, !dbg !440
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!378}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !27 = !{!7}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !100 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !377 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !378 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !377, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !379 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !377, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !386)
// CHECK:STDOUT: !380 = !DILocalVariable(arg: 1, scope: !379, type: !7)
// CHECK:STDOUT: !384 = !DILocation(line: 11, column: 10, scope: !379)
// CHECK:STDOUT: !385 = !DILocation(line: 11, column: 3, scope: !379)
// CHECK:STDOUT: !386 = !{!380}
// CHECK:STDOUT: !387 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.0fa40e8d9e2a73f6", scope: null, file: !100, line: 50, type: !9, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !395)
// CHECK:STDOUT: !388 = !DILocalVariable(arg: 1, scope: !387, type: !7)
// CHECK:STDOUT: !393 = !DILocation(line: 51, column: 12, scope: !387)
// CHECK:STDOUT: !394 = !DILocation(line: 51, column: 5, scope: !387)
// CHECK:STDOUT: !395 = !{!388}
// CHECK:STDOUT: !396 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286", scope: null, file: !100, line: 147, type: !9, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !409)
// CHECK:STDOUT: !397 = !DILocalVariable(arg: 1, scope: !396, type: !7)
// CHECK:STDOUT: !402 = !DILocation(line: 148, column: 9, scope: !396)
// CHECK:STDOUT: !403 = !DILocation(line: 148, column: 8, scope: !396)
// CHECK:STDOUT: !404 = !DILocation(line: 149, column: 15, scope: !396)
// CHECK:STDOUT: !405 = !DILocation(line: 149, column: 14, scope: !396)
// CHECK:STDOUT: !406 = !DILocation(line: 149, column: 7, scope: !396)
// CHECK:STDOUT: !407 = !DILocation(line: 151, column: 14, scope: !396)
// CHECK:STDOUT: !408 = !DILocation(line: 151, column: 7, scope: !396)
// CHECK:STDOUT: !409 = !{!397}
// CHECK:STDOUT: !410 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286", scope: null, file: !100, line: 141, type: !9, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !417)
// CHECK:STDOUT: !411 = !DILocalVariable(arg: 1, scope: !410, type: !7)
// CHECK:STDOUT: !415 = !DILocation(line: 142, column: 12, scope: !410)
// CHECK:STDOUT: !416 = !DILocation(line: 142, column: 5, scope: !410)
// CHECK:STDOUT: !417 = !{!411}
// CHECK:STDOUT: !418 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286", scope: null, file: !100, line: 131, type: !9, spFlags: DISPFlagDefinition, unit: !378, retainedNodes: !431)
// CHECK:STDOUT: !419 = !DILocalVariable(arg: 1, scope: !418, type: !7)
// CHECK:STDOUT: !427 = !DILocation(line: 137, column: 5, scope: !418)
// CHECK:STDOUT: !428 = !DILocation(line: 137, column: 28, scope: !418)
// CHECK:STDOUT: !429 = !DILocation(line: 138, column: 5, scope: !418)
// CHECK:STDOUT: !430 = !DILocation(line: 139, column: 5, scope: !418)
// CHECK:STDOUT: !431 = !{!419}
// CHECK:STDOUT: !432 = distinct !DISubprogram(name: "None", linkageName: "_CNone.21149ca0dc753fdf:OptionalStorage.Core.53ffaddca9fbc286", scope: null, file: !100, line: 125, type: !28, spFlags: DISPFlagDefinition, unit: !378)
// CHECK:STDOUT: !439 = !DILocation(line: 128, column: 5, scope: !432)
// CHECK:STDOUT: !440 = !DILocation(line: 129, column: 5, scope: !432)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !443 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %c), !dbg !448
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !449
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %self) #0 !dbg !451 {
// CHECK:STDOUT:   ret ptr %self, !dbg !458
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!442}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !100 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !441 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !442 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !441, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !443 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !441, line: 5, type: !9, spFlags: DISPFlagDefinition, unit: !442, retainedNodes: !450)
// CHECK:STDOUT: !444 = !DILocalVariable(arg: 1, scope: !443, type: !7)
// CHECK:STDOUT: !448 = !DILocation(line: 6, column: 10, scope: !443)
// CHECK:STDOUT: !449 = !DILocation(line: 6, column: 3, scope: !443)
// CHECK:STDOUT: !450 = !{!444}
// CHECK:STDOUT: !451 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608", scope: null, file: !100, line: 50, type: !9, spFlags: DISPFlagDefinition, unit: !442, retainedNodes: !459)
// CHECK:STDOUT: !452 = !DILocalVariable(arg: 1, scope: !451, type: !7)
// CHECK:STDOUT: !458 = !DILocation(line: 51, column: 5, scope: !451)
// CHECK:STDOUT: !459 = !{!452}
// CHECK:STDOUT:
