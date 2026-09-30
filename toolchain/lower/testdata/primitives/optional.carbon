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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %o), !dbg !30
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %o), !dbg !32
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !33
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %.loc5_12.2), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.3530343734a6ea27(ptr %return), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 4, !dbg !47
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 4, !dbg !50
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 4, !dbg !53
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 4, !dbg !56
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 4, !dbg !59
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 4, !dbg !62
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 4, !dbg !65
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 4, !dbg !68
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !47
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %_.var.loc11, i32 %a), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !50
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr %_.var.loc12, i32 %a), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !53
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %_.var.loc13, i32 %a), !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !56
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr %_.var.loc14, i32 %a), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !59
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465"(ptr %_.var.loc15, i32 %b), !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !62
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821"(ptr %_.var.loc16, i32 %b), !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !65
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96"(ptr %_.var.loc17, i32 %b), !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !68
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77"(ptr %_.var.loc18, i32 %b), !dbg !68
// CHECK:STDOUT:   call void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %_.var.loc18), !dbg !68
// CHECK:STDOUT:   call void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %_.var.loc17), !dbg !65
// CHECK:STDOUT:   call void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %_.var.loc16), !dbg !62
// CHECK:STDOUT:   call void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %_.var.loc15), !dbg !59
// CHECK:STDOUT:   call void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %_.var.loc14), !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %_.var.loc13), !dbg !53
// CHECK:STDOUT:   call void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %_.var.loc12), !dbg !50
// CHECK:STDOUT:   call void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %_.var.loc11), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.ef6922493c513f55:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !95
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc18_3.1.has_value), !dbg !95
// CHECK:STDOUT:   %.loc18_3.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !95
// CHECK:STDOUT:   call void @"_CSelfDestruct.ef6922493c513f55:core.Destroy.Core"(ptr %.loc18_3.2.value), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6854f843fd23bc45:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core"(ptr %self), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d81dc56afe745c8e:core.Destroy.Core"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core"(ptr %self), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.51569357b34b54dc:core.Destroy.Core"(ptr %self) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core"(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !131
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc16_3.1.has_value), !dbg !131
// CHECK:STDOUT:   %.loc16_3.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !131
// CHECK:STDOUT:   call void @"_CSelfDestruct.51569357b34b54dc:core.Destroy.Core"(ptr %.loc16_3.2.value), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f6a63055ea5a1824:core.Destroy.Core"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core"(ptr %self), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core"(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.9f155f365c9e4267:core.Destroy.Core"(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core"(ptr %self), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %self) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !153 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !159
// CHECK:STDOUT:   ret i1 %1, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !162 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !168
// CHECK:STDOUT:   ret ptr %1, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !171 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27"(ptr %return, i32 %self), !dbg !176
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.3530343734a6ea27(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !179 {
// CHECK:STDOUT:   call void @"_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr %return), !dbg !182
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !185 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %self), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !193 {
// CHECK:STDOUT:   call void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7"(ptr %return, i32 %self), !dbg !198
// CHECK:STDOUT:   ret void, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !201 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %return, i32 %self), !dbg !206
// CHECK:STDOUT:   ret void, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !209 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %self), !dbg !215
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !218 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr %return, i32 %self), !dbg !224
// CHECK:STDOUT:   ret void, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !227 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %return, i32 %self), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !234
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !236 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr %return, i32 %self), !dbg !242
// CHECK:STDOUT:   ret void, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !245 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !251
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !252
// CHECK:STDOUT:   ret i1 %2, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !255 {
// CHECK:STDOUT:   ret ptr %value, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !263 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.3530343734a6ea27(ptr %return, i32 %self), !dbg !269
// CHECK:STDOUT:   ret void, !dbg !270
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !272 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !279
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !279
// CHECK:STDOUT:   ret void, !dbg !280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !281 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !286
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !286
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self), !dbg !286
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !286
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !286
// CHECK:STDOUT:   call void @_CSome.Optional.Core.00b00ea46fa3b046(ptr %return, i32 %2), !dbg !287
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core"(ptr %temp), !dbg !286
// CHECK:STDOUT:   ret void, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.3530343734a6ea27(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !290 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr %return, i32 %value), !dbg !295
// CHECK:STDOUT:   ret void, !dbg !296
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self) #0 !dbg !298 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self), !dbg !303
// CHECK:STDOUT:   ret i32 %1, !dbg !306
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.00b00ea46fa3b046(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !308 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2"(ptr %return, i32 %value), !dbg !313
// CHECK:STDOUT:   ret void, !dbg !314
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !316 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !325
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !325
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !327
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !327
// CHECK:STDOUT:   ret void, !dbg !328
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !331 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !336
// CHECK:STDOUT:   ret i32 %1, !dbg !338
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !340 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !349
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self), !dbg !350
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !349
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !351
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !351
// CHECK:STDOUT:   ret void, !dbg !352
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !354 {
// CHECK:STDOUT:   ret i32 %self, !dbg !360
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self) #0 !dbg !363 {
// CHECK:STDOUT:   ret i32 %self, !dbg !370
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core", { 1, 0 }
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
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !24)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !6)
// CHECK:STDOUT: !23 = !DILocation(line: 3, column: 39, scope: !21)
// CHECK:STDOUT: !24 = !{!22}
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !4, line: 3, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 4, column: 7, scope: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 4, column: 6, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 5, column: 13, scope: !25)
// CHECK:STDOUT: !33 = !DILocation(line: 5, column: 12, scope: !25)
// CHECK:STDOUT: !34 = !DILocation(line: 5, column: 5, scope: !25)
// CHECK:STDOUT: !35 = !{!7, !6}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 7, column: 10, scope: !25)
// CHECK:STDOUT: !38 = !{!7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 7, column: 3, scope: !25)
// CHECK:STDOUT: !41 = !{!26}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !4, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !72)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !6)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 2, scope: !42, type: !6)
// CHECK:STDOUT: !47 = !DILocation(line: 11, column: 3, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 12, column: 3, scope: !42)
// CHECK:STDOUT: !53 = !DILocation(line: 13, column: 3, scope: !42)
// CHECK:STDOUT: !56 = !DILocation(line: 14, column: 3, scope: !42)
// CHECK:STDOUT: !59 = !DILocation(line: 15, column: 3, scope: !42)
// CHECK:STDOUT: !62 = !DILocation(line: 16, column: 3, scope: !42)
// CHECK:STDOUT: !65 = !DILocation(line: 17, column: 3, scope: !42)
// CHECK:STDOUT: !68 = !DILocation(line: 18, column: 3, scope: !42)
// CHECK:STDOUT: !71 = !DILocation(line: 10, column: 1, scope: !42)
// CHECK:STDOUT: !72 = !{!43, !44}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core", scope: null, file: !4, line: 13, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !75 = !DILocation(line: 13, column: 10, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !7)
// CHECK:STDOUT: !79 = !DILocation(line: 18, column: 3, scope: !77)
// CHECK:STDOUT: !80 = !{!78}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.96a347e4ca9c10e1:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !83 = !DILocation(line: 18, column: 3, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.ef6922493c513f55:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !7)
// CHECK:STDOUT: !87 = !DILocation(line: 18, column: 3, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !7)
// CHECK:STDOUT: !91 = !DILocation(line: 18, column: 3, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 18, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6854f843fd23bc45:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !99 = !DILocation(line: 18, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !7)
// CHECK:STDOUT: !103 = !DILocation(line: 18, column: 3, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d81dc56afe745c8e:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 18, column: 3, scope: !105)
// CHECK:STDOUT: !108 = !{!106}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b90699303d3716f5:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !111 = !DILocation(line: 18, column: 3, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.60848ede08716e91:core.Destroy.Core", scope: null, file: !4, line: 17, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !116)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !115 = !DILocation(line: 17, column: 3, scope: !113)
// CHECK:STDOUT: !116 = !{!114}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !120)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 16, column: 3, scope: !117)
// CHECK:STDOUT: !120 = !{!118}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !124)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !7)
// CHECK:STDOUT: !123 = !DILocation(line: 16, column: 3, scope: !121)
// CHECK:STDOUT: !124 = !{!122}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.51569357b34b54dc:core.Destroy.Core", scope: null, file: !4, line: 16, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !128)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !6)
// CHECK:STDOUT: !127 = !DILocation(line: 16, column: 3, scope: !125)
// CHECK:STDOUT: !128 = !{!126}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !7)
// CHECK:STDOUT: !131 = !DILocation(line: 16, column: 3, scope: !129)
// CHECK:STDOUT: !132 = !{!130}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f6a63055ea5a1824:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !136)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !7)
// CHECK:STDOUT: !135 = !DILocation(line: 16, column: 3, scope: !133)
// CHECK:STDOUT: !136 = !{!134}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !7)
// CHECK:STDOUT: !139 = !DILocation(line: 16, column: 3, scope: !137)
// CHECK:STDOUT: !140 = !{!138}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.9f155f365c9e4267:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !144)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !7)
// CHECK:STDOUT: !143 = !DILocation(line: 16, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !{!142}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !7)
// CHECK:STDOUT: !147 = !DILocation(line: 16, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !{!146}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core", scope: null, file: !4, line: 15, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !7)
// CHECK:STDOUT: !151 = !DILocation(line: 15, column: 3, scope: !149)
// CHECK:STDOUT: !152 = !{!150}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.6e9e957697159b53", scope: null, file: !16, line: 36, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !161)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !7)
// CHECK:STDOUT: !159 = !DILocation(line: 37, column: 12, scope: !153)
// CHECK:STDOUT: !160 = !DILocation(line: 37, column: 5, scope: !153)
// CHECK:STDOUT: !161 = !{!154}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.6e9e957697159b53", scope: null, file: !16, line: 39, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !170)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !7)
// CHECK:STDOUT: !168 = !DILocation(line: 40, column: 12, scope: !162)
// CHECK:STDOUT: !169 = !DILocation(line: 40, column: 5, scope: !162)
// CHECK:STDOUT: !170 = !{!163}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843", scope: null, file: !16, line: 105, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !178)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !6)
// CHECK:STDOUT: !176 = !DILocation(line: 106, column: 12, scope: !171)
// CHECK:STDOUT: !177 = !DILocation(line: 106, column: 5, scope: !171)
// CHECK:STDOUT: !178 = !{!172}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.3530343734a6ea27", scope: null, file: !16, line: 30, type: !39, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !182 = !DILocation(line: 31, column: 12, scope: !179)
// CHECK:STDOUT: !183 = !DILocation(line: 31, column: 5, scope: !179)
// CHECK:STDOUT: !184 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465", scope: null, file: !184, line: 43, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !192)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !6)
// CHECK:STDOUT: !190 = !DILocation(line: 43, column: 40, scope: !185)
// CHECK:STDOUT: !191 = !DILocation(line: 43, column: 33, scope: !185)
// CHECK:STDOUT: !192 = !{!186}
// CHECK:STDOUT: !193 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224", scope: null, file: !16, line: 105, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !200)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !193, type: !6)
// CHECK:STDOUT: !198 = !DILocation(line: 106, column: 12, scope: !193)
// CHECK:STDOUT: !199 = !DILocation(line: 106, column: 5, scope: !193)
// CHECK:STDOUT: !200 = !{!194}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96", scope: null, file: !184, line: 43, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !208)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !6)
// CHECK:STDOUT: !206 = !DILocation(line: 43, column: 40, scope: !201)
// CHECK:STDOUT: !207 = !DILocation(line: 43, column: 33, scope: !201)
// CHECK:STDOUT: !208 = !{!202}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465", scope: null, file: !184, line: 47, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !217)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !6)
// CHECK:STDOUT: !215 = !DILocation(line: 47, column: 34, scope: !209)
// CHECK:STDOUT: !216 = !DILocation(line: 47, column: 27, scope: !209)
// CHECK:STDOUT: !217 = !{!210}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821", scope: null, file: !184, line: 47, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !226)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !6)
// CHECK:STDOUT: !224 = !DILocation(line: 47, column: 34, scope: !218)
// CHECK:STDOUT: !225 = !DILocation(line: 47, column: 27, scope: !218)
// CHECK:STDOUT: !226 = !{!219}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96", scope: null, file: !184, line: 47, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !235)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !6)
// CHECK:STDOUT: !233 = !DILocation(line: 47, column: 34, scope: !227)
// CHECK:STDOUT: !234 = !DILocation(line: 47, column: 27, scope: !227)
// CHECK:STDOUT: !235 = !{!228}
// CHECK:STDOUT: !236 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77", scope: null, file: !184, line: 47, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !244)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 1, scope: !236, type: !6)
// CHECK:STDOUT: !242 = !DILocation(line: 47, column: 34, scope: !236)
// CHECK:STDOUT: !243 = !DILocation(line: 47, column: 27, scope: !236)
// CHECK:STDOUT: !244 = !{!237}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !16, line: 172, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !254)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !7)
// CHECK:STDOUT: !251 = !DILocation(line: 173, column: 16, scope: !245)
// CHECK:STDOUT: !252 = !DILocation(line: 173, column: 12, scope: !245)
// CHECK:STDOUT: !253 = !DILocation(line: 173, column: 5, scope: !245)
// CHECK:STDOUT: !254 = !{!246}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !16, line: 175, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !262)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !7)
// CHECK:STDOUT: !261 = !DILocation(line: 176, column: 5, scope: !255)
// CHECK:STDOUT: !262 = !{!256}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27", scope: null, file: !16, line: 80, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !271)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !6)
// CHECK:STDOUT: !269 = !DILocation(line: 81, column: 12, scope: !263)
// CHECK:STDOUT: !270 = !DILocation(line: 81, column: 5, scope: !263)
// CHECK:STDOUT: !271 = !{!264}
// CHECK:STDOUT: !272 = distinct !DISubprogram(name: "None", linkageName: "_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc", scope: null, file: !16, line: 125, type: !39, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !279 = !DILocation(line: 128, column: 5, scope: !272)
// CHECK:STDOUT: !280 = !DILocation(line: 129, column: 5, scope: !272)
// CHECK:STDOUT: !281 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7", scope: null, file: !16, line: 87, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !289)
// CHECK:STDOUT: !282 = !DILocalVariable(arg: 1, scope: !281, type: !6)
// CHECK:STDOUT: !286 = !DILocation(line: 88, column: 29, scope: !281)
// CHECK:STDOUT: !287 = !DILocation(line: 88, column: 12, scope: !281)
// CHECK:STDOUT: !288 = !DILocation(line: 88, column: 5, scope: !281)
// CHECK:STDOUT: !289 = !{!282}
// CHECK:STDOUT: !290 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.3530343734a6ea27", scope: null, file: !16, line: 33, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !297)
// CHECK:STDOUT: !291 = !DILocalVariable(arg: 1, scope: !290, type: !6)
// CHECK:STDOUT: !295 = !DILocation(line: 34, column: 12, scope: !290)
// CHECK:STDOUT: !296 = !DILocation(line: 34, column: 5, scope: !290)
// CHECK:STDOUT: !297 = !{!291}
// CHECK:STDOUT: !298 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2", scope: null, file: !184, line: 43, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !307)
// CHECK:STDOUT: !299 = !DILocalVariable(arg: 1, scope: !298, type: !6)
// CHECK:STDOUT: !303 = !DILocation(line: 43, column: 40, scope: !298)
// CHECK:STDOUT: !304 = !{!6, !6}
// CHECK:STDOUT: !305 = !DISubroutineType(types: !304)
// CHECK:STDOUT: !306 = !DILocation(line: 43, column: 33, scope: !298)
// CHECK:STDOUT: !307 = !{!299}
// CHECK:STDOUT: !308 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.00b00ea46fa3b046", scope: null, file: !16, line: 33, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !315)
// CHECK:STDOUT: !309 = !DILocalVariable(arg: 1, scope: !308, type: !6)
// CHECK:STDOUT: !313 = !DILocation(line: 34, column: 12, scope: !308)
// CHECK:STDOUT: !314 = !DILocation(line: 34, column: 5, scope: !308)
// CHECK:STDOUT: !315 = !{!309}
// CHECK:STDOUT: !316 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc", scope: null, file: !16, line: 131, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !329)
// CHECK:STDOUT: !317 = !DILocalVariable(arg: 1, scope: !316, type: !6)
// CHECK:STDOUT: !325 = !DILocation(line: 137, column: 5, scope: !316)
// CHECK:STDOUT: !327 = !DILocation(line: 138, column: 5, scope: !316)
// CHECK:STDOUT: !328 = !DILocation(line: 139, column: 5, scope: !316)
// CHECK:STDOUT: !329 = !{!317}
// CHECK:STDOUT: !330 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !331 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !330, line: 64, type: !305, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !339)
// CHECK:STDOUT: !332 = !DILocalVariable(arg: 1, scope: !331, type: !6)
// CHECK:STDOUT: !336 = !DILocation(line: 66, column: 17, scope: !331)
// CHECK:STDOUT: !338 = !DILocation(line: 66, column: 5, scope: !331)
// CHECK:STDOUT: !339 = !{!332}
// CHECK:STDOUT: !340 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2", scope: null, file: !16, line: 131, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !353)
// CHECK:STDOUT: !341 = !DILocalVariable(arg: 1, scope: !340, type: !6)
// CHECK:STDOUT: !349 = !DILocation(line: 137, column: 5, scope: !340)
// CHECK:STDOUT: !350 = !DILocation(line: 137, column: 28, scope: !340)
// CHECK:STDOUT: !351 = !DILocation(line: 138, column: 5, scope: !340)
// CHECK:STDOUT: !352 = !DILocation(line: 139, column: 5, scope: !340)
// CHECK:STDOUT: !353 = !{!341}
// CHECK:STDOUT: !354 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !330, line: 59, type: !305, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !361)
// CHECK:STDOUT: !355 = !DILocalVariable(arg: 1, scope: !354, type: !6)
// CHECK:STDOUT: !360 = !DILocation(line: 59, column: 28, scope: !354)
// CHECK:STDOUT: !361 = !{!355}
// CHECK:STDOUT: !362 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !363 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4", scope: null, file: !362, line: 23, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !371)
// CHECK:STDOUT: !364 = !DILocalVariable(arg: 1, scope: !363, type: !6)
// CHECK:STDOUT: !370 = !DILocation(line: 23, column: 25, scope: !363)
// CHECK:STDOUT: !371 = !{!364}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !374 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !376
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb3dd30fa3a4aa62:core.Destroy.Core"(ptr %self) #0 !dbg !378 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !380
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !382 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31"(ptr %return, ptr %c), !dbg !387
// CHECK:STDOUT:   ret void, !dbg !388
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !390 {
// CHECK:STDOUT:   call void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return, ptr %self), !dbg !396
// CHECK:STDOUT:   ret void, !dbg !397
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !399 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %value), !dbg !405
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !406
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !407
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return, ptr %value1), !dbg !408
// CHECK:STDOUT:   ret void, !dbg !409
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return), !dbg !410
// CHECK:STDOUT:   ret void, !dbg !411
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %value) #0 !dbg !413 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !418
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !418
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !418
// CHECK:STDOUT:   ret i1 %2, !dbg !419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !421 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !430
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !431
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !432
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !432
// CHECK:STDOUT:   ret void, !dbg !433
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return) #0 !dbg !435 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !442
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !442
// CHECK:STDOUT:   ret void, !dbg !443
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!373}
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
// CHECK:STDOUT: !38 = !{!7}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !372 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !373 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !372, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !374 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !372, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !377)
// CHECK:STDOUT: !375 = !DILocalVariable(arg: 1, scope: !374, type: !7)
// CHECK:STDOUT: !376 = !DILocation(line: 9, column: 9, scope: !374)
// CHECK:STDOUT: !377 = !{!375}
// CHECK:STDOUT: !378 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb3dd30fa3a4aa62:core.Destroy.Core", scope: null, file: !372, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !381)
// CHECK:STDOUT: !379 = !DILocalVariable(arg: 1, scope: !378, type: !7)
// CHECK:STDOUT: !380 = !DILocation(line: 9, column: 9, scope: !378)
// CHECK:STDOUT: !381 = !{!379}
// CHECK:STDOUT: !382 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !372, line: 9, type: !13, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !389)
// CHECK:STDOUT: !383 = !DILocalVariable(arg: 1, scope: !382, type: !7)
// CHECK:STDOUT: !387 = !DILocation(line: 11, column: 10, scope: !382)
// CHECK:STDOUT: !388 = !DILocation(line: 11, column: 3, scope: !382)
// CHECK:STDOUT: !389 = !{!383}
// CHECK:STDOUT: !390 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31", scope: null, file: !16, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !398)
// CHECK:STDOUT: !391 = !DILocalVariable(arg: 1, scope: !390, type: !7)
// CHECK:STDOUT: !396 = !DILocation(line: 51, column: 12, scope: !390)
// CHECK:STDOUT: !397 = !DILocation(line: 51, column: 5, scope: !390)
// CHECK:STDOUT: !398 = !{!391}
// CHECK:STDOUT: !399 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 147, type: !13, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !412)
// CHECK:STDOUT: !400 = !DILocalVariable(arg: 1, scope: !399, type: !7)
// CHECK:STDOUT: !405 = !DILocation(line: 148, column: 9, scope: !399)
// CHECK:STDOUT: !406 = !DILocation(line: 148, column: 8, scope: !399)
// CHECK:STDOUT: !407 = !DILocation(line: 149, column: 15, scope: !399)
// CHECK:STDOUT: !408 = !DILocation(line: 149, column: 14, scope: !399)
// CHECK:STDOUT: !409 = !DILocation(line: 149, column: 7, scope: !399)
// CHECK:STDOUT: !410 = !DILocation(line: 151, column: 14, scope: !399)
// CHECK:STDOUT: !411 = !DILocation(line: 151, column: 7, scope: !399)
// CHECK:STDOUT: !412 = !{!400}
// CHECK:STDOUT: !413 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 141, type: !13, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !420)
// CHECK:STDOUT: !414 = !DILocalVariable(arg: 1, scope: !413, type: !7)
// CHECK:STDOUT: !418 = !DILocation(line: 142, column: 12, scope: !413)
// CHECK:STDOUT: !419 = !DILocation(line: 142, column: 5, scope: !413)
// CHECK:STDOUT: !420 = !{!414}
// CHECK:STDOUT: !421 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !373, retainedNodes: !434)
// CHECK:STDOUT: !422 = !DILocalVariable(arg: 1, scope: !421, type: !7)
// CHECK:STDOUT: !430 = !DILocation(line: 137, column: 5, scope: !421)
// CHECK:STDOUT: !431 = !DILocation(line: 137, column: 28, scope: !421)
// CHECK:STDOUT: !432 = !DILocation(line: 138, column: 5, scope: !421)
// CHECK:STDOUT: !433 = !DILocation(line: 139, column: 5, scope: !421)
// CHECK:STDOUT: !434 = !{!422}
// CHECK:STDOUT: !435 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 125, type: !39, spFlags: DISPFlagDefinition, unit: !373)
// CHECK:STDOUT: !442 = !DILocation(line: 128, column: 5, scope: !435)
// CHECK:STDOUT: !443 = !DILocation(line: 129, column: 5, scope: !435)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %self) #0 !dbg !446 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !448
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !450 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %c), !dbg !455
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !456
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %self) #0 !dbg !458 {
// CHECK:STDOUT:   ret ptr %self, !dbg !465
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!445}
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
// CHECK:STDOUT: !444 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !445 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !444, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !446 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b633cf020343e69:core.Destroy.Core", scope: null, file: !16, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !445, retainedNodes: !449)
// CHECK:STDOUT: !447 = !DILocalVariable(arg: 1, scope: !446, type: !7)
// CHECK:STDOUT: !448 = !DILocation(line: 164, column: 19, scope: !446)
// CHECK:STDOUT: !449 = !{!447}
// CHECK:STDOUT: !450 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !444, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !445, retainedNodes: !457)
// CHECK:STDOUT: !451 = !DILocalVariable(arg: 1, scope: !450, type: !7)
// CHECK:STDOUT: !455 = !DILocation(line: 6, column: 10, scope: !450)
// CHECK:STDOUT: !456 = !DILocation(line: 6, column: 3, scope: !450)
// CHECK:STDOUT: !457 = !{!451}
// CHECK:STDOUT: !458 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608", scope: null, file: !16, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !445, retainedNodes: !466)
// CHECK:STDOUT: !459 = !DILocalVariable(arg: 1, scope: !458, type: !7)
// CHECK:STDOUT: !465 = !DILocation(line: 51, column: 5, scope: !458)
// CHECK:STDOUT: !466 = !{!459}
// CHECK:STDOUT:
