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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %o), !dbg !34
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %o), !dbg !36
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !37
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %.loc5_12.2), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.3530343734a6ea27(ptr %return), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 4, !dbg !51
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 4, !dbg !54
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 4, !dbg !57
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 4, !dbg !60
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 4, !dbg !63
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 4, !dbg !66
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 4, !dbg !69
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 4, !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !51
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %_.var.loc11, i32 %a), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !54
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr %_.var.loc12, i32 %a), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !57
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %_.var.loc13, i32 %a), !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !60
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr %_.var.loc14, i32 %a), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !63
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465"(ptr %_.var.loc15, i32 %b), !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !66
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821"(ptr %_.var.loc16, i32 %b), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !69
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96"(ptr %_.var.loc17, i32 %b), !dbg !69
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !72
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77"(ptr %_.var.loc18, i32 %b), !dbg !72
// CHECK:STDOUT:   call void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %_.var.loc18), !dbg !72
// CHECK:STDOUT:   call void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %_.var.loc17), !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %_.var.loc16), !dbg !66
// CHECK:STDOUT:   call void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %_.var.loc15), !dbg !63
// CHECK:STDOUT:   call void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %_.var.loc14), !dbg !60
// CHECK:STDOUT:   call void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %_.var.loc13), !dbg !57
// CHECK:STDOUT:   call void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %_.var.loc12), !dbg !54
// CHECK:STDOUT:   call void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %_.var.loc11), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core"(ptr %self), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.ef6922493c513f55:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6854f843fd23bc45:core.Destroy.Core"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core"(ptr %self), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d81dc56afe745c8e:core.Destroy.Core"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core"(ptr %self), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %self) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b90699303d3716f5:core.Destroy.Core"(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %self) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b90699303d3716f5:core.Destroy.Core"(ptr %self), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.51569357b34b54dc:core.Destroy.Core"(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core"(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f6a63055ea5a1824:core.Destroy.Core"(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core"(ptr %self), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core"(ptr %self) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.9f155f365c9e4267:core.Destroy.Core"(ptr %self) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core"(ptr %self), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %self) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.65ac38c7f841b05c:core.Destroy.Core"(ptr %self) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %self) #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.65ac38c7f841b05c:core.Destroy.Core"(ptr %self), !dbg !167
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !169 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !175
// CHECK:STDOUT:   ret i1 %1, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !178 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !184
// CHECK:STDOUT:   ret ptr %1, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !187 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27"(ptr %return, i32 %self), !dbg !192
// CHECK:STDOUT:   ret void, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.3530343734a6ea27(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !195 {
// CHECK:STDOUT:   call void @"_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr %return), !dbg !198
// CHECK:STDOUT:   ret void, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !201 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %self), !dbg !206
// CHECK:STDOUT:   ret void, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !209 {
// CHECK:STDOUT:   call void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7"(ptr %return, i32 %self), !dbg !214
// CHECK:STDOUT:   ret void, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !217 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %return, i32 %self), !dbg !222
// CHECK:STDOUT:   ret void, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !225 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %self), !dbg !231
// CHECK:STDOUT:   ret void, !dbg !232
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !234 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr %return, i32 %self), !dbg !240
// CHECK:STDOUT:   ret void, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !243 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %return, i32 %self), !dbg !249
// CHECK:STDOUT:   ret void, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !252 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr %return, i32 %self), !dbg !258
// CHECK:STDOUT:   ret void, !dbg !259
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !261 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !267
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !268
// CHECK:STDOUT:   ret i1 %2, !dbg !269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !271 {
// CHECK:STDOUT:   ret ptr %value, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !279 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.3530343734a6ea27(ptr %return, i32 %self), !dbg !285
// CHECK:STDOUT:   ret void, !dbg !286
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !288 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !295
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !295
// CHECK:STDOUT:   ret void, !dbg !296
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !297 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !302
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !302
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self), !dbg !302
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !302
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !302
// CHECK:STDOUT:   call void @_CSome.Optional.Core.00b00ea46fa3b046(ptr %return, i32 %2), !dbg !303
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core"(ptr %temp), !dbg !302
// CHECK:STDOUT:   ret void, !dbg !304
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.3530343734a6ea27(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !306 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr %return, i32 %value), !dbg !311
// CHECK:STDOUT:   ret void, !dbg !312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self) #0 !dbg !314 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self), !dbg !319
// CHECK:STDOUT:   ret i32 %1, !dbg !322
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.00b00ea46fa3b046(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !324 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2"(ptr %return, i32 %value), !dbg !329
// CHECK:STDOUT:   ret void, !dbg !330
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !332 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !341
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !341
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !343
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !343
// CHECK:STDOUT:   ret void, !dbg !344
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !347 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !352
// CHECK:STDOUT:   ret i32 %1, !dbg !354
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !356 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !365
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self), !dbg !366
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !365
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !367
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !367
// CHECK:STDOUT:   ret void, !dbg !368
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !370 {
// CHECK:STDOUT:   ret i32 %self, !dbg !376
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self) #0 !dbg !379 {
// CHECK:STDOUT:   ret i32 %self, !dbg !386
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843", { 0, 1, 3, 2 }
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
// CHECK:STDOUT: !8 = !{null, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !6, !6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !16, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 164, column: 19, scope: !17)
// CHECK:STDOUT: !20 = !{!18}
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !24)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !6)
// CHECK:STDOUT: !23 = !DILocation(line: 3, column: 39, scope: !21)
// CHECK:STDOUT: !24 = !{!22}
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !28)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !6)
// CHECK:STDOUT: !27 = !DILocation(line: 3, column: 39, scope: !25)
// CHECK:STDOUT: !28 = !{!26}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !4, line: 3, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 4, column: 7, scope: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 4, column: 6, scope: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 5, column: 13, scope: !29)
// CHECK:STDOUT: !37 = !DILocation(line: 5, column: 12, scope: !29)
// CHECK:STDOUT: !38 = !DILocation(line: 5, column: 5, scope: !29)
// CHECK:STDOUT: !39 = !{!7, !6}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 7, column: 10, scope: !29)
// CHECK:STDOUT: !42 = !{!7}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 7, column: 3, scope: !29)
// CHECK:STDOUT: !45 = !{!30}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !4, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !6)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 2, scope: !46, type: !6)
// CHECK:STDOUT: !51 = !DILocation(line: 11, column: 3, scope: !46)
// CHECK:STDOUT: !54 = !DILocation(line: 12, column: 3, scope: !46)
// CHECK:STDOUT: !57 = !DILocation(line: 13, column: 3, scope: !46)
// CHECK:STDOUT: !60 = !DILocation(line: 14, column: 3, scope: !46)
// CHECK:STDOUT: !63 = !DILocation(line: 15, column: 3, scope: !46)
// CHECK:STDOUT: !66 = !DILocation(line: 16, column: 3, scope: !46)
// CHECK:STDOUT: !69 = !DILocation(line: 17, column: 3, scope: !46)
// CHECK:STDOUT: !72 = !DILocation(line: 18, column: 3, scope: !46)
// CHECK:STDOUT: !75 = !DILocation(line: 10, column: 1, scope: !46)
// CHECK:STDOUT: !76 = !{!47, !48}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core", scope: null, file: !4, line: 13, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !7)
// CHECK:STDOUT: !79 = !DILocation(line: 13, column: 10, scope: !77)
// CHECK:STDOUT: !80 = !{!78}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core", scope: null, file: !4, line: 13, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !83 = !DILocation(line: 13, column: 10, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !7)
// CHECK:STDOUT: !87 = !DILocation(line: 18, column: 3, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.96a347e4ca9c10e1:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !7)
// CHECK:STDOUT: !91 = !DILocation(line: 18, column: 3, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.ef6922493c513f55:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 18, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !99 = !DILocation(line: 18, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !7)
// CHECK:STDOUT: !103 = !DILocation(line: 18, column: 3, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6854f843fd23bc45:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 18, column: 3, scope: !105)
// CHECK:STDOUT: !108 = !{!106}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !111 = !DILocation(line: 18, column: 3, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d81dc56afe745c8e:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !116)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !115 = !DILocation(line: 18, column: 3, scope: !113)
// CHECK:STDOUT: !116 = !{!114}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.60848ede08716e91:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !120)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 18, column: 3, scope: !117)
// CHECK:STDOUT: !120 = !{!118}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b90699303d3716f5:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !124)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !7)
// CHECK:STDOUT: !123 = !DILocation(line: 18, column: 3, scope: !121)
// CHECK:STDOUT: !124 = !{!122}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b90699303d3716f5:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !128)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !7)
// CHECK:STDOUT: !127 = !DILocation(line: 18, column: 3, scope: !125)
// CHECK:STDOUT: !128 = !{!126}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !7)
// CHECK:STDOUT: !131 = !DILocation(line: 16, column: 3, scope: !129)
// CHECK:STDOUT: !132 = !{!130}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !136)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !7)
// CHECK:STDOUT: !135 = !DILocation(line: 16, column: 3, scope: !133)
// CHECK:STDOUT: !136 = !{!134}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.51569357b34b54dc:core.Destroy.Core", scope: null, file: !4, line: 16, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !6)
// CHECK:STDOUT: !139 = !DILocation(line: 16, column: 3, scope: !137)
// CHECK:STDOUT: !140 = !{!138}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !144)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !7)
// CHECK:STDOUT: !143 = !DILocation(line: 16, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !{!142}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f6a63055ea5a1824:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !7)
// CHECK:STDOUT: !147 = !DILocation(line: 16, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !{!146}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !7)
// CHECK:STDOUT: !151 = !DILocation(line: 16, column: 3, scope: !149)
// CHECK:STDOUT: !152 = !{!150}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.9f155f365c9e4267:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !7)
// CHECK:STDOUT: !155 = !DILocation(line: 16, column: 3, scope: !153)
// CHECK:STDOUT: !156 = !{!154}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !160)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !7)
// CHECK:STDOUT: !159 = !DILocation(line: 16, column: 3, scope: !157)
// CHECK:STDOUT: !160 = !{!158}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.65ac38c7f841b05c:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !164)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !7)
// CHECK:STDOUT: !163 = !DILocation(line: 16, column: 3, scope: !161)
// CHECK:STDOUT: !164 = !{!162}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !168)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !7)
// CHECK:STDOUT: !167 = !DILocation(line: 16, column: 3, scope: !165)
// CHECK:STDOUT: !168 = !{!166}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.6e9e957697159b53", scope: null, file: !16, line: 36, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !7)
// CHECK:STDOUT: !175 = !DILocation(line: 37, column: 12, scope: !169)
// CHECK:STDOUT: !176 = !DILocation(line: 37, column: 5, scope: !169)
// CHECK:STDOUT: !177 = !{!170}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.6e9e957697159b53", scope: null, file: !16, line: 39, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !186)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !7)
// CHECK:STDOUT: !184 = !DILocation(line: 40, column: 12, scope: !178)
// CHECK:STDOUT: !185 = !DILocation(line: 40, column: 5, scope: !178)
// CHECK:STDOUT: !186 = !{!179}
// CHECK:STDOUT: !187 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843", scope: null, file: !16, line: 105, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !194)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 1, scope: !187, type: !6)
// CHECK:STDOUT: !192 = !DILocation(line: 106, column: 12, scope: !187)
// CHECK:STDOUT: !193 = !DILocation(line: 106, column: 5, scope: !187)
// CHECK:STDOUT: !194 = !{!188}
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.3530343734a6ea27", scope: null, file: !16, line: 30, type: !43, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !198 = !DILocation(line: 31, column: 12, scope: !195)
// CHECK:STDOUT: !199 = !DILocation(line: 31, column: 5, scope: !195)
// CHECK:STDOUT: !200 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465", scope: null, file: !200, line: 43, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !208)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !6)
// CHECK:STDOUT: !206 = !DILocation(line: 43, column: 40, scope: !201)
// CHECK:STDOUT: !207 = !DILocation(line: 43, column: 33, scope: !201)
// CHECK:STDOUT: !208 = !{!202}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224", scope: null, file: !16, line: 105, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !216)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !6)
// CHECK:STDOUT: !214 = !DILocation(line: 106, column: 12, scope: !209)
// CHECK:STDOUT: !215 = !DILocation(line: 106, column: 5, scope: !209)
// CHECK:STDOUT: !216 = !{!210}
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96", scope: null, file: !200, line: 43, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !224)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 1, scope: !217, type: !6)
// CHECK:STDOUT: !222 = !DILocation(line: 43, column: 40, scope: !217)
// CHECK:STDOUT: !223 = !DILocation(line: 43, column: 33, scope: !217)
// CHECK:STDOUT: !224 = !{!218}
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465", scope: null, file: !200, line: 47, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !233)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !6)
// CHECK:STDOUT: !231 = !DILocation(line: 47, column: 34, scope: !225)
// CHECK:STDOUT: !232 = !DILocation(line: 47, column: 27, scope: !225)
// CHECK:STDOUT: !233 = !{!226}
// CHECK:STDOUT: !234 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821", scope: null, file: !200, line: 47, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !242)
// CHECK:STDOUT: !235 = !DILocalVariable(arg: 1, scope: !234, type: !6)
// CHECK:STDOUT: !240 = !DILocation(line: 47, column: 34, scope: !234)
// CHECK:STDOUT: !241 = !DILocation(line: 47, column: 27, scope: !234)
// CHECK:STDOUT: !242 = !{!235}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96", scope: null, file: !200, line: 47, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !251)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !6)
// CHECK:STDOUT: !249 = !DILocation(line: 47, column: 34, scope: !243)
// CHECK:STDOUT: !250 = !DILocation(line: 47, column: 27, scope: !243)
// CHECK:STDOUT: !251 = !{!244}
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77", scope: null, file: !200, line: 47, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !260)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 1, scope: !252, type: !6)
// CHECK:STDOUT: !258 = !DILocation(line: 47, column: 34, scope: !252)
// CHECK:STDOUT: !259 = !DILocation(line: 47, column: 27, scope: !252)
// CHECK:STDOUT: !260 = !{!253}
// CHECK:STDOUT: !261 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !16, line: 172, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !270)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 1, scope: !261, type: !7)
// CHECK:STDOUT: !267 = !DILocation(line: 173, column: 16, scope: !261)
// CHECK:STDOUT: !268 = !DILocation(line: 173, column: 12, scope: !261)
// CHECK:STDOUT: !269 = !DILocation(line: 173, column: 5, scope: !261)
// CHECK:STDOUT: !270 = !{!262}
// CHECK:STDOUT: !271 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !16, line: 175, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !278)
// CHECK:STDOUT: !272 = !DILocalVariable(arg: 1, scope: !271, type: !7)
// CHECK:STDOUT: !277 = !DILocation(line: 176, column: 5, scope: !271)
// CHECK:STDOUT: !278 = !{!272}
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27", scope: null, file: !16, line: 80, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !287)
// CHECK:STDOUT: !280 = !DILocalVariable(arg: 1, scope: !279, type: !6)
// CHECK:STDOUT: !285 = !DILocation(line: 81, column: 12, scope: !279)
// CHECK:STDOUT: !286 = !DILocation(line: 81, column: 5, scope: !279)
// CHECK:STDOUT: !287 = !{!280}
// CHECK:STDOUT: !288 = distinct !DISubprogram(name: "None", linkageName: "_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc", scope: null, file: !16, line: 125, type: !43, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !295 = !DILocation(line: 128, column: 5, scope: !288)
// CHECK:STDOUT: !296 = !DILocation(line: 129, column: 5, scope: !288)
// CHECK:STDOUT: !297 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7", scope: null, file: !16, line: 87, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !305)
// CHECK:STDOUT: !298 = !DILocalVariable(arg: 1, scope: !297, type: !6)
// CHECK:STDOUT: !302 = !DILocation(line: 88, column: 29, scope: !297)
// CHECK:STDOUT: !303 = !DILocation(line: 88, column: 12, scope: !297)
// CHECK:STDOUT: !304 = !DILocation(line: 88, column: 5, scope: !297)
// CHECK:STDOUT: !305 = !{!298}
// CHECK:STDOUT: !306 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.3530343734a6ea27", scope: null, file: !16, line: 33, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !313)
// CHECK:STDOUT: !307 = !DILocalVariable(arg: 1, scope: !306, type: !6)
// CHECK:STDOUT: !311 = !DILocation(line: 34, column: 12, scope: !306)
// CHECK:STDOUT: !312 = !DILocation(line: 34, column: 5, scope: !306)
// CHECK:STDOUT: !313 = !{!307}
// CHECK:STDOUT: !314 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2", scope: null, file: !200, line: 43, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !323)
// CHECK:STDOUT: !315 = !DILocalVariable(arg: 1, scope: !314, type: !6)
// CHECK:STDOUT: !319 = !DILocation(line: 43, column: 40, scope: !314)
// CHECK:STDOUT: !320 = !{!6, !6}
// CHECK:STDOUT: !321 = !DISubroutineType(types: !320)
// CHECK:STDOUT: !322 = !DILocation(line: 43, column: 33, scope: !314)
// CHECK:STDOUT: !323 = !{!315}
// CHECK:STDOUT: !324 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.00b00ea46fa3b046", scope: null, file: !16, line: 33, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !331)
// CHECK:STDOUT: !325 = !DILocalVariable(arg: 1, scope: !324, type: !6)
// CHECK:STDOUT: !329 = !DILocation(line: 34, column: 12, scope: !324)
// CHECK:STDOUT: !330 = !DILocation(line: 34, column: 5, scope: !324)
// CHECK:STDOUT: !331 = !{!325}
// CHECK:STDOUT: !332 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc", scope: null, file: !16, line: 131, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !345)
// CHECK:STDOUT: !333 = !DILocalVariable(arg: 1, scope: !332, type: !6)
// CHECK:STDOUT: !341 = !DILocation(line: 137, column: 5, scope: !332)
// CHECK:STDOUT: !343 = !DILocation(line: 138, column: 5, scope: !332)
// CHECK:STDOUT: !344 = !DILocation(line: 139, column: 5, scope: !332)
// CHECK:STDOUT: !345 = !{!333}
// CHECK:STDOUT: !346 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !347 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !346, line: 64, type: !321, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !355)
// CHECK:STDOUT: !348 = !DILocalVariable(arg: 1, scope: !347, type: !6)
// CHECK:STDOUT: !352 = !DILocation(line: 66, column: 17, scope: !347)
// CHECK:STDOUT: !354 = !DILocation(line: 66, column: 5, scope: !347)
// CHECK:STDOUT: !355 = !{!348}
// CHECK:STDOUT: !356 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2", scope: null, file: !16, line: 131, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !369)
// CHECK:STDOUT: !357 = !DILocalVariable(arg: 1, scope: !356, type: !6)
// CHECK:STDOUT: !365 = !DILocation(line: 137, column: 5, scope: !356)
// CHECK:STDOUT: !366 = !DILocation(line: 137, column: 28, scope: !356)
// CHECK:STDOUT: !367 = !DILocation(line: 138, column: 5, scope: !356)
// CHECK:STDOUT: !368 = !DILocation(line: 139, column: 5, scope: !356)
// CHECK:STDOUT: !369 = !{!357}
// CHECK:STDOUT: !370 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !346, line: 59, type: !321, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !377)
// CHECK:STDOUT: !371 = !DILocalVariable(arg: 1, scope: !370, type: !6)
// CHECK:STDOUT: !376 = !DILocation(line: 59, column: 28, scope: !370)
// CHECK:STDOUT: !377 = !{!371}
// CHECK:STDOUT: !378 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !379 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4", scope: null, file: !378, line: 23, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !387)
// CHECK:STDOUT: !380 = !DILocalVariable(arg: 1, scope: !379, type: !6)
// CHECK:STDOUT: !386 = !DILocation(line: 23, column: 25, scope: !379)
// CHECK:STDOUT: !387 = !{!380}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !390 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !392
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb3dd30fa3a4aa62:core.Destroy.Core"(ptr %self) #0 !dbg !394 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !396
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !398 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31"(ptr %return, ptr %c), !dbg !403
// CHECK:STDOUT:   ret void, !dbg !404
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !406 {
// CHECK:STDOUT:   call void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return, ptr %self), !dbg !412
// CHECK:STDOUT:   ret void, !dbg !413
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !415 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %value), !dbg !421
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !422
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !423
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return, ptr %value1), !dbg !424
// CHECK:STDOUT:   ret void, !dbg !425
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return), !dbg !426
// CHECK:STDOUT:   ret void, !dbg !427
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %value) #0 !dbg !429 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !434
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !434
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !434
// CHECK:STDOUT:   ret i1 %2, !dbg !435
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !437 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !446
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !447
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !448
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !448
// CHECK:STDOUT:   ret void, !dbg !449
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return) #0 !dbg !451 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !458
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !458
// CHECK:STDOUT:   ret void, !dbg !459
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!389}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !12 = !{!7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !16 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !42 = !{!7}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !388 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !389 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !388, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !390 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !388, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !389, retainedNodes: !393)
// CHECK:STDOUT: !391 = !DILocalVariable(arg: 1, scope: !390, type: !7)
// CHECK:STDOUT: !392 = !DILocation(line: 9, column: 9, scope: !390)
// CHECK:STDOUT: !393 = !{!391}
// CHECK:STDOUT: !394 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb3dd30fa3a4aa62:core.Destroy.Core", scope: null, file: !388, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !389, retainedNodes: !397)
// CHECK:STDOUT: !395 = !DILocalVariable(arg: 1, scope: !394, type: !7)
// CHECK:STDOUT: !396 = !DILocation(line: 9, column: 9, scope: !394)
// CHECK:STDOUT: !397 = !{!395}
// CHECK:STDOUT: !398 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !388, line: 9, type: !13, spFlags: DISPFlagDefinition, unit: !389, retainedNodes: !405)
// CHECK:STDOUT: !399 = !DILocalVariable(arg: 1, scope: !398, type: !7)
// CHECK:STDOUT: !403 = !DILocation(line: 11, column: 10, scope: !398)
// CHECK:STDOUT: !404 = !DILocation(line: 11, column: 3, scope: !398)
// CHECK:STDOUT: !405 = !{!399}
// CHECK:STDOUT: !406 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31", scope: null, file: !16, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !389, retainedNodes: !414)
// CHECK:STDOUT: !407 = !DILocalVariable(arg: 1, scope: !406, type: !7)
// CHECK:STDOUT: !412 = !DILocation(line: 51, column: 12, scope: !406)
// CHECK:STDOUT: !413 = !DILocation(line: 51, column: 5, scope: !406)
// CHECK:STDOUT: !414 = !{!407}
// CHECK:STDOUT: !415 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 147, type: !13, spFlags: DISPFlagDefinition, unit: !389, retainedNodes: !428)
// CHECK:STDOUT: !416 = !DILocalVariable(arg: 1, scope: !415, type: !7)
// CHECK:STDOUT: !421 = !DILocation(line: 148, column: 9, scope: !415)
// CHECK:STDOUT: !422 = !DILocation(line: 148, column: 8, scope: !415)
// CHECK:STDOUT: !423 = !DILocation(line: 149, column: 15, scope: !415)
// CHECK:STDOUT: !424 = !DILocation(line: 149, column: 14, scope: !415)
// CHECK:STDOUT: !425 = !DILocation(line: 149, column: 7, scope: !415)
// CHECK:STDOUT: !426 = !DILocation(line: 151, column: 14, scope: !415)
// CHECK:STDOUT: !427 = !DILocation(line: 151, column: 7, scope: !415)
// CHECK:STDOUT: !428 = !{!416}
// CHECK:STDOUT: !429 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 141, type: !13, spFlags: DISPFlagDefinition, unit: !389, retainedNodes: !436)
// CHECK:STDOUT: !430 = !DILocalVariable(arg: 1, scope: !429, type: !7)
// CHECK:STDOUT: !434 = !DILocation(line: 142, column: 12, scope: !429)
// CHECK:STDOUT: !435 = !DILocation(line: 142, column: 5, scope: !429)
// CHECK:STDOUT: !436 = !{!430}
// CHECK:STDOUT: !437 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !389, retainedNodes: !450)
// CHECK:STDOUT: !438 = !DILocalVariable(arg: 1, scope: !437, type: !7)
// CHECK:STDOUT: !446 = !DILocation(line: 137, column: 5, scope: !437)
// CHECK:STDOUT: !447 = !DILocation(line: 137, column: 28, scope: !437)
// CHECK:STDOUT: !448 = !DILocation(line: 138, column: 5, scope: !437)
// CHECK:STDOUT: !449 = !DILocation(line: 139, column: 5, scope: !437)
// CHECK:STDOUT: !450 = !{!438}
// CHECK:STDOUT: !451 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 125, type: !43, spFlags: DISPFlagDefinition, unit: !389)
// CHECK:STDOUT: !458 = !DILocation(line: 128, column: 5, scope: !451)
// CHECK:STDOUT: !459 = !DILocation(line: 129, column: 5, scope: !451)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %self) #0 !dbg !462 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !464
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !466 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %c), !dbg !471
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !472
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %self) #0 !dbg !474 {
// CHECK:STDOUT:   ret ptr %self, !dbg !481
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!461}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !12 = !{!7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !16 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !460 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !461 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !460, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !462 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b633cf020343e69:core.Destroy.Core", scope: null, file: !16, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !461, retainedNodes: !465)
// CHECK:STDOUT: !463 = !DILocalVariable(arg: 1, scope: !462, type: !7)
// CHECK:STDOUT: !464 = !DILocation(line: 164, column: 19, scope: !462)
// CHECK:STDOUT: !465 = !{!463}
// CHECK:STDOUT: !466 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !460, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !461, retainedNodes: !473)
// CHECK:STDOUT: !467 = !DILocalVariable(arg: 1, scope: !466, type: !7)
// CHECK:STDOUT: !471 = !DILocation(line: 6, column: 10, scope: !466)
// CHECK:STDOUT: !472 = !DILocation(line: 6, column: 3, scope: !466)
// CHECK:STDOUT: !473 = !{!467}
// CHECK:STDOUT: !474 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608", scope: null, file: !16, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !461, retainedNodes: !482)
// CHECK:STDOUT: !475 = !DILocalVariable(arg: 1, scope: !474, type: !7)
// CHECK:STDOUT: !481 = !DILocation(line: 51, column: 5, scope: !474)
// CHECK:STDOUT: !482 = !{!475}
// CHECK:STDOUT:
