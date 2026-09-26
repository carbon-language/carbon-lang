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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %o), !dbg !38
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !39
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %o), !dbg !40
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !41
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %.loc5_12.2), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.3530343734a6ea27(ptr %return), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 4, !dbg !55
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 4, !dbg !58
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 4, !dbg !61
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 4, !dbg !64
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 4, !dbg !67
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 4, !dbg !70
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 4, !dbg !73
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 4, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !55
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %_.var.loc11, i32 %a), !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !58
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr %_.var.loc12, i32 %a), !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !61
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %_.var.loc13, i32 %a), !dbg !61
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !64
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr %_.var.loc14, i32 %a), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !67
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465"(ptr %_.var.loc15, i32 %b), !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !70
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821"(ptr %_.var.loc16, i32 %b), !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !73
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96"(ptr %_.var.loc17, i32 %b), !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !76
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77"(ptr %_.var.loc18, i32 %b), !dbg !76
// CHECK:STDOUT:   call void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %_.var.loc18), !dbg !76
// CHECK:STDOUT:   call void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %_.var.loc17), !dbg !73
// CHECK:STDOUT:   call void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %_.var.loc16), !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %_.var.loc15), !dbg !67
// CHECK:STDOUT:   call void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %_.var.loc14), !dbg !64
// CHECK:STDOUT:   call void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %_.var.loc13), !dbg !61
// CHECK:STDOUT:   call void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %_.var.loc12), !dbg !58
// CHECK:STDOUT:   call void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %_.var.loc11), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core"(ptr %self), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ef6922493c513f55:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.ef6922493c513f55:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.ef6922493c513f55:core.Destroy.Core"(ptr %self), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !111
// CHECK:STDOUT:   call void @"_CSelfDestruct.ef6922493c513f55:core.Destroy.Core"(ptr %.loc18_3.1.value), !dbg !111
// CHECK:STDOUT:   %.loc18_3.2.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !111
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc18_3.2.has_value), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6854f843fd23bc45:core.Destroy.Core"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core"(ptr %self), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core"(ptr %self) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d81dc56afe745c8e:core.Destroy.Core"(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core"(ptr %self), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.60848ede08716e91:core.Destroy.Core"(ptr %self) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.60848ede08716e91:core.Destroy.Core"(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.60848ede08716e91:core.Destroy.Core"(ptr %self), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b90699303d3716f5:core.Destroy.Core"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b90699303d3716f5:core.Destroy.Core"(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b90699303d3716f5:core.Destroy.Core"(ptr %self), !dbg !139
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.51569357b34b54dc:core.Destroy.Core"(ptr %self) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.51569357b34b54dc:core.Destroy.Core"(ptr %self) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.51569357b34b54dc:core.Destroy.Core"(ptr %self), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core"(ptr %self) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !159
// CHECK:STDOUT:   call void @"_CSelfDestruct.51569357b34b54dc:core.Destroy.Core"(ptr %.loc16_3.1.value), !dbg !159
// CHECK:STDOUT:   %.loc16_3.2.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !159
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc16_3.2.has_value), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f6a63055ea5a1824:core.Destroy.Core"(ptr %self) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core"(ptr %self), !dbg !163
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core"(ptr %self) #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.9f155f365c9e4267:core.Destroy.Core"(ptr %self) #0 !dbg !169 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core"(ptr %self), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0acefe5e4ad33694:core.Destroy.Core"(ptr %self) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core"(ptr %self) #0 !dbg !177 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0acefe5e4ad33694:core.Destroy.Core"(ptr %self), !dbg !179
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.65ac38c7f841b05c:core.Destroy.Core"(ptr %self) #0 !dbg !181 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core"(ptr %self) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.65ac38c7f841b05c:core.Destroy.Core"(ptr %self), !dbg !187
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !189 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !195
// CHECK:STDOUT:   ret i1 %1, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !198 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !204
// CHECK:STDOUT:   ret ptr %1, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !207 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27"(ptr %return, i32 %self), !dbg !212
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.3530343734a6ea27(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !215 {
// CHECK:STDOUT:   call void @"_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr %return), !dbg !218
// CHECK:STDOUT:   ret void, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !221 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %self), !dbg !226
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !229 {
// CHECK:STDOUT:   call void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7"(ptr %return, i32 %self), !dbg !234
// CHECK:STDOUT:   ret void, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !237 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %return, i32 %self), !dbg !242
// CHECK:STDOUT:   ret void, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !245 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843"(ptr %return, i32 %self), !dbg !251
// CHECK:STDOUT:   ret void, !dbg !252
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !254 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465"(ptr %return, i32 %self), !dbg !260
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !263 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224"(ptr %return, i32 %self), !dbg !269
// CHECK:STDOUT:   ret void, !dbg !270
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !272 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96"(ptr %return, i32 %self), !dbg !278
// CHECK:STDOUT:   ret void, !dbg !279
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !281 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !287
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !288
// CHECK:STDOUT:   ret i1 %2, !dbg !289
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !291 {
// CHECK:STDOUT:   ret ptr %value, !dbg !297
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !299 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.3530343734a6ea27(ptr %return, i32 %self), !dbg !305
// CHECK:STDOUT:   ret void, !dbg !306
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !308 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !315
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !315
// CHECK:STDOUT:   ret void, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !317 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !322
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !322
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self), !dbg !322
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !322
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !322
// CHECK:STDOUT:   call void @_CSome.Optional.Core.00b00ea46fa3b046(ptr %return, i32 %2), !dbg !323
// CHECK:STDOUT:   call void @"_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core"(ptr %temp), !dbg !322
// CHECK:STDOUT:   ret void, !dbg !324
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.3530343734a6ea27(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !326 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr %return, i32 %value), !dbg !331
// CHECK:STDOUT:   ret void, !dbg !332
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self) #0 !dbg !334 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self), !dbg !339
// CHECK:STDOUT:   ret i32 %1, !dbg !342
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.00b00ea46fa3b046(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !344 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2"(ptr %return, i32 %value), !dbg !349
// CHECK:STDOUT:   ret void, !dbg !350
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !352 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !361
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !361
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !363
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !363
// CHECK:STDOUT:   ret void, !dbg !364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !367 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !372
// CHECK:STDOUT:   ret i32 %1, !dbg !374
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !376 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !385
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self), !dbg !386
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !385
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !387
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !387
// CHECK:STDOUT:   ret void, !dbg !388
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !390 {
// CHECK:STDOUT:   ret i32 %self, !dbg !396
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self) #0 !dbg !399 {
// CHECK:STDOUT:   ret i32 %self, !dbg !406
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
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !28)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !6)
// CHECK:STDOUT: !27 = !DILocation(line: 3, column: 39, scope: !25)
// CHECK:STDOUT: !28 = !{!26}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !32)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !6)
// CHECK:STDOUT: !31 = !DILocation(line: 3, column: 39, scope: !29)
// CHECK:STDOUT: !32 = !{!30}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !4, line: 3, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !38 = !DILocation(line: 4, column: 7, scope: !33)
// CHECK:STDOUT: !39 = !DILocation(line: 4, column: 6, scope: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 5, column: 13, scope: !33)
// CHECK:STDOUT: !41 = !DILocation(line: 5, column: 12, scope: !33)
// CHECK:STDOUT: !42 = !DILocation(line: 5, column: 5, scope: !33)
// CHECK:STDOUT: !43 = !{!7, !6}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 7, column: 10, scope: !33)
// CHECK:STDOUT: !46 = !{!7}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !DILocation(line: 7, column: 3, scope: !33)
// CHECK:STDOUT: !49 = !{!34}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !4, line: 10, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !6)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !50, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 11, column: 3, scope: !50)
// CHECK:STDOUT: !58 = !DILocation(line: 12, column: 3, scope: !50)
// CHECK:STDOUT: !61 = !DILocation(line: 13, column: 3, scope: !50)
// CHECK:STDOUT: !64 = !DILocation(line: 14, column: 3, scope: !50)
// CHECK:STDOUT: !67 = !DILocation(line: 15, column: 3, scope: !50)
// CHECK:STDOUT: !70 = !DILocation(line: 16, column: 3, scope: !50)
// CHECK:STDOUT: !73 = !DILocation(line: 17, column: 3, scope: !50)
// CHECK:STDOUT: !76 = !DILocation(line: 18, column: 3, scope: !50)
// CHECK:STDOUT: !79 = !DILocation(line: 10, column: 1, scope: !50)
// CHECK:STDOUT: !80 = !{!51, !52}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core", scope: null, file: !4, line: 13, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !83 = !DILocation(line: 13, column: 10, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6a8d754fe1453eab:core.Destroy.Core", scope: null, file: !4, line: 13, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !7)
// CHECK:STDOUT: !87 = !DILocation(line: 13, column: 10, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !7)
// CHECK:STDOUT: !91 = !DILocation(line: 18, column: 3, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.96a347e4ca9c10e1:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 18, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ef6922493c513f55:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !99 = !DILocation(line: 18, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.ef6922493c513f55:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !7)
// CHECK:STDOUT: !103 = !DILocation(line: 18, column: 3, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 18, column: 3, scope: !105)
// CHECK:STDOUT: !108 = !{!106}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6854f843fd23bc45:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !111 = !DILocation(line: 18, column: 3, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6854f843fd23bc45:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !116)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !115 = !DILocation(line: 18, column: 3, scope: !113)
// CHECK:STDOUT: !116 = !{!114}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d81dc56afe745c8e:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !120)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 18, column: 3, scope: !117)
// CHECK:STDOUT: !120 = !{!118}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d81dc56afe745c8e:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !124)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !7)
// CHECK:STDOUT: !123 = !DILocation(line: 18, column: 3, scope: !121)
// CHECK:STDOUT: !124 = !{!122}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.60848ede08716e91:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !128)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !7)
// CHECK:STDOUT: !127 = !DILocation(line: 18, column: 3, scope: !125)
// CHECK:STDOUT: !128 = !{!126}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.60848ede08716e91:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !7)
// CHECK:STDOUT: !131 = !DILocation(line: 18, column: 3, scope: !129)
// CHECK:STDOUT: !132 = !{!130}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b90699303d3716f5:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !136)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !7)
// CHECK:STDOUT: !135 = !DILocation(line: 18, column: 3, scope: !133)
// CHECK:STDOUT: !136 = !{!134}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b90699303d3716f5:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !7)
// CHECK:STDOUT: !139 = !DILocation(line: 18, column: 3, scope: !137)
// CHECK:STDOUT: !140 = !{!138}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !144)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !7)
// CHECK:STDOUT: !143 = !DILocation(line: 16, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !{!142}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !7)
// CHECK:STDOUT: !147 = !DILocation(line: 16, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !{!146}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.51569357b34b54dc:core.Destroy.Core", scope: null, file: !4, line: 16, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !6)
// CHECK:STDOUT: !151 = !DILocation(line: 16, column: 3, scope: !149)
// CHECK:STDOUT: !152 = !{!150}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.51569357b34b54dc:core.Destroy.Core", scope: null, file: !4, line: 16, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !6)
// CHECK:STDOUT: !155 = !DILocation(line: 16, column: 3, scope: !153)
// CHECK:STDOUT: !156 = !{!154}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f6a63055ea5a1824:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !160)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !7)
// CHECK:STDOUT: !159 = !DILocation(line: 16, column: 3, scope: !157)
// CHECK:STDOUT: !160 = !{!158}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f6a63055ea5a1824:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !164)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !7)
// CHECK:STDOUT: !163 = !DILocation(line: 16, column: 3, scope: !161)
// CHECK:STDOUT: !164 = !{!162}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.9f155f365c9e4267:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !168)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !7)
// CHECK:STDOUT: !167 = !DILocation(line: 16, column: 3, scope: !165)
// CHECK:STDOUT: !168 = !{!166}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.9f155f365c9e4267:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !172)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !7)
// CHECK:STDOUT: !171 = !DILocation(line: 16, column: 3, scope: !169)
// CHECK:STDOUT: !172 = !{!170}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0acefe5e4ad33694:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !176)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !7)
// CHECK:STDOUT: !175 = !DILocation(line: 16, column: 3, scope: !173)
// CHECK:STDOUT: !176 = !{!174}
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0acefe5e4ad33694:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !180)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !177, type: !7)
// CHECK:STDOUT: !179 = !DILocation(line: 16, column: 3, scope: !177)
// CHECK:STDOUT: !180 = !{!178}
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.65ac38c7f841b05c:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !184)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !7)
// CHECK:STDOUT: !183 = !DILocation(line: 16, column: 3, scope: !181)
// CHECK:STDOUT: !184 = !{!182}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.65ac38c7f841b05c:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !188)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !7)
// CHECK:STDOUT: !187 = !DILocation(line: 16, column: 3, scope: !185)
// CHECK:STDOUT: !188 = !{!186}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.6e9e957697159b53", scope: null, file: !16, line: 36, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !197)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !7)
// CHECK:STDOUT: !195 = !DILocation(line: 37, column: 12, scope: !189)
// CHECK:STDOUT: !196 = !DILocation(line: 37, column: 5, scope: !189)
// CHECK:STDOUT: !197 = !{!190}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.6e9e957697159b53", scope: null, file: !16, line: 39, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !206)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !7)
// CHECK:STDOUT: !204 = !DILocation(line: 40, column: 12, scope: !198)
// CHECK:STDOUT: !205 = !DILocation(line: 40, column: 5, scope: !198)
// CHECK:STDOUT: !206 = !{!199}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.9eebcd25770b8843", scope: null, file: !16, line: 105, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !214)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !6)
// CHECK:STDOUT: !212 = !DILocation(line: 106, column: 12, scope: !207)
// CHECK:STDOUT: !213 = !DILocation(line: 106, column: 5, scope: !207)
// CHECK:STDOUT: !214 = !{!208}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.3530343734a6ea27", scope: null, file: !16, line: 30, type: !47, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !218 = !DILocation(line: 31, column: 12, scope: !215)
// CHECK:STDOUT: !219 = !DILocation(line: 31, column: 5, scope: !215)
// CHECK:STDOUT: !220 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.27a2d1ef6dd68465", scope: null, file: !220, line: 43, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !228)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !6)
// CHECK:STDOUT: !226 = !DILocation(line: 43, column: 40, scope: !221)
// CHECK:STDOUT: !227 = !DILocation(line: 43, column: 33, scope: !221)
// CHECK:STDOUT: !228 = !{!222}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.5cec6441aa655224", scope: null, file: !16, line: 105, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !236)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !6)
// CHECK:STDOUT: !234 = !DILocation(line: 106, column: 12, scope: !229)
// CHECK:STDOUT: !235 = !DILocation(line: 106, column: 5, scope: !229)
// CHECK:STDOUT: !236 = !{!230}
// CHECK:STDOUT: !237 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.bc0e935618dc3a96", scope: null, file: !220, line: 43, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !244)
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 1, scope: !237, type: !6)
// CHECK:STDOUT: !242 = !DILocation(line: 43, column: 40, scope: !237)
// CHECK:STDOUT: !243 = !DILocation(line: 43, column: 33, scope: !237)
// CHECK:STDOUT: !244 = !{!238}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.27a2d1ef6dd68465", scope: null, file: !220, line: 47, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !253)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !6)
// CHECK:STDOUT: !251 = !DILocation(line: 47, column: 34, scope: !245)
// CHECK:STDOUT: !252 = !DILocation(line: 47, column: 27, scope: !245)
// CHECK:STDOUT: !253 = !{!246}
// CHECK:STDOUT: !254 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.9f04190beae22821", scope: null, file: !220, line: 47, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !262)
// CHECK:STDOUT: !255 = !DILocalVariable(arg: 1, scope: !254, type: !6)
// CHECK:STDOUT: !260 = !DILocation(line: 47, column: 34, scope: !254)
// CHECK:STDOUT: !261 = !DILocation(line: 47, column: 27, scope: !254)
// CHECK:STDOUT: !262 = !{!255}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bc0e935618dc3a96", scope: null, file: !220, line: 47, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !271)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !6)
// CHECK:STDOUT: !269 = !DILocation(line: 47, column: 34, scope: !263)
// CHECK:STDOUT: !270 = !DILocation(line: 47, column: 27, scope: !263)
// CHECK:STDOUT: !271 = !{!264}
// CHECK:STDOUT: !272 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.342679e541375c77", scope: null, file: !220, line: 47, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !280)
// CHECK:STDOUT: !273 = !DILocalVariable(arg: 1, scope: !272, type: !6)
// CHECK:STDOUT: !278 = !DILocation(line: 47, column: 34, scope: !272)
// CHECK:STDOUT: !279 = !DILocation(line: 47, column: 27, scope: !272)
// CHECK:STDOUT: !280 = !{!273}
// CHECK:STDOUT: !281 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !16, line: 172, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !290)
// CHECK:STDOUT: !282 = !DILocalVariable(arg: 1, scope: !281, type: !7)
// CHECK:STDOUT: !287 = !DILocation(line: 173, column: 16, scope: !281)
// CHECK:STDOUT: !288 = !DILocation(line: 173, column: 12, scope: !281)
// CHECK:STDOUT: !289 = !DILocation(line: 173, column: 5, scope: !281)
// CHECK:STDOUT: !290 = !{!282}
// CHECK:STDOUT: !291 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !16, line: 175, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !298)
// CHECK:STDOUT: !292 = !DILocalVariable(arg: 1, scope: !291, type: !7)
// CHECK:STDOUT: !297 = !DILocation(line: 176, column: 5, scope: !291)
// CHECK:STDOUT: !298 = !{!292}
// CHECK:STDOUT: !299 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.3530343734a6ea27", scope: null, file: !16, line: 80, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !307)
// CHECK:STDOUT: !300 = !DILocalVariable(arg: 1, scope: !299, type: !6)
// CHECK:STDOUT: !305 = !DILocation(line: 81, column: 12, scope: !299)
// CHECK:STDOUT: !306 = !DILocation(line: 81, column: 5, scope: !299)
// CHECK:STDOUT: !307 = !{!300}
// CHECK:STDOUT: !308 = distinct !DISubprogram(name: "None", linkageName: "_CNone.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc", scope: null, file: !16, line: 125, type: !47, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !315 = !DILocation(line: 128, column: 5, scope: !308)
// CHECK:STDOUT: !316 = !DILocation(line: 129, column: 5, scope: !308)
// CHECK:STDOUT: !317 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.dd5a2a74150e1fc7", scope: null, file: !16, line: 87, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !325)
// CHECK:STDOUT: !318 = !DILocalVariable(arg: 1, scope: !317, type: !6)
// CHECK:STDOUT: !322 = !DILocation(line: 88, column: 29, scope: !317)
// CHECK:STDOUT: !323 = !DILocation(line: 88, column: 12, scope: !317)
// CHECK:STDOUT: !324 = !DILocation(line: 88, column: 5, scope: !317)
// CHECK:STDOUT: !325 = !{!318}
// CHECK:STDOUT: !326 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.3530343734a6ea27", scope: null, file: !16, line: 33, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !333)
// CHECK:STDOUT: !327 = !DILocalVariable(arg: 1, scope: !326, type: !6)
// CHECK:STDOUT: !331 = !DILocation(line: 34, column: 12, scope: !326)
// CHECK:STDOUT: !332 = !DILocation(line: 34, column: 5, scope: !326)
// CHECK:STDOUT: !333 = !{!327}
// CHECK:STDOUT: !334 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2", scope: null, file: !220, line: 43, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !343)
// CHECK:STDOUT: !335 = !DILocalVariable(arg: 1, scope: !334, type: !6)
// CHECK:STDOUT: !339 = !DILocation(line: 43, column: 40, scope: !334)
// CHECK:STDOUT: !340 = !{!6, !6}
// CHECK:STDOUT: !341 = !DISubroutineType(types: !340)
// CHECK:STDOUT: !342 = !DILocation(line: 43, column: 33, scope: !334)
// CHECK:STDOUT: !343 = !{!335}
// CHECK:STDOUT: !344 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.00b00ea46fa3b046", scope: null, file: !16, line: 33, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !351)
// CHECK:STDOUT: !345 = !DILocalVariable(arg: 1, scope: !344, type: !6)
// CHECK:STDOUT: !349 = !DILocation(line: 34, column: 12, scope: !344)
// CHECK:STDOUT: !350 = !DILocation(line: 34, column: 5, scope: !344)
// CHECK:STDOUT: !351 = !{!345}
// CHECK:STDOUT: !352 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.d4057e3ae48c02cc", scope: null, file: !16, line: 131, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !365)
// CHECK:STDOUT: !353 = !DILocalVariable(arg: 1, scope: !352, type: !6)
// CHECK:STDOUT: !361 = !DILocation(line: 137, column: 5, scope: !352)
// CHECK:STDOUT: !363 = !DILocation(line: 138, column: 5, scope: !352)
// CHECK:STDOUT: !364 = !DILocation(line: 139, column: 5, scope: !352)
// CHECK:STDOUT: !365 = !{!353}
// CHECK:STDOUT: !366 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !367 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !366, line: 64, type: !341, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !375)
// CHECK:STDOUT: !368 = !DILocalVariable(arg: 1, scope: !367, type: !6)
// CHECK:STDOUT: !372 = !DILocation(line: 66, column: 17, scope: !367)
// CHECK:STDOUT: !374 = !DILocation(line: 66, column: 5, scope: !367)
// CHECK:STDOUT: !375 = !{!368}
// CHECK:STDOUT: !376 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.cd77d7fbcb14a7c2", scope: null, file: !16, line: 131, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !389)
// CHECK:STDOUT: !377 = !DILocalVariable(arg: 1, scope: !376, type: !6)
// CHECK:STDOUT: !385 = !DILocation(line: 137, column: 5, scope: !376)
// CHECK:STDOUT: !386 = !DILocation(line: 137, column: 28, scope: !376)
// CHECK:STDOUT: !387 = !DILocation(line: 138, column: 5, scope: !376)
// CHECK:STDOUT: !388 = !DILocation(line: 139, column: 5, scope: !376)
// CHECK:STDOUT: !389 = !{!377}
// CHECK:STDOUT: !390 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !366, line: 59, type: !341, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !397)
// CHECK:STDOUT: !391 = !DILocalVariable(arg: 1, scope: !390, type: !6)
// CHECK:STDOUT: !396 = !DILocation(line: 59, column: 28, scope: !390)
// CHECK:STDOUT: !397 = !{!391}
// CHECK:STDOUT: !398 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !399 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4", scope: null, file: !398, line: 23, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !407)
// CHECK:STDOUT: !400 = !DILocalVariable(arg: 1, scope: !399, type: !6)
// CHECK:STDOUT: !406 = !DILocation(line: 23, column: 25, scope: !399)
// CHECK:STDOUT: !407 = !{!400}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !410 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !412
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bb3dd30fa3a4aa62:core.Destroy.Core"(ptr %self) #0 !dbg !414 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !416
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb3dd30fa3a4aa62:core.Destroy.Core"(ptr %self) #0 !dbg !418 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.bb3dd30fa3a4aa62:core.Destroy.Core"(ptr %self), !dbg !420
// CHECK:STDOUT:   ret void, !dbg !420
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !422 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31"(ptr %return, ptr %c), !dbg !427
// CHECK:STDOUT:   ret void, !dbg !428
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !430 {
// CHECK:STDOUT:   call void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return, ptr %self), !dbg !436
// CHECK:STDOUT:   ret void, !dbg !437
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !439 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %value), !dbg !445
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !446
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !447
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return, ptr %value1), !dbg !448
// CHECK:STDOUT:   ret void, !dbg !449
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return), !dbg !450
// CHECK:STDOUT:   ret void, !dbg !451
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %value) #0 !dbg !453 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !458
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !458
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !458
// CHECK:STDOUT:   ret i1 %2, !dbg !459
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !461 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !470
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !471
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !472
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !472
// CHECK:STDOUT:   ret void, !dbg !473
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return) #0 !dbg !475 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !482
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !482
// CHECK:STDOUT:   ret void, !dbg !483
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!409}
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
// CHECK:STDOUT: !46 = !{!7}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !408 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !409 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !408, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !410 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !408, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !409, retainedNodes: !413)
// CHECK:STDOUT: !411 = !DILocalVariable(arg: 1, scope: !410, type: !7)
// CHECK:STDOUT: !412 = !DILocation(line: 9, column: 9, scope: !410)
// CHECK:STDOUT: !413 = !{!411}
// CHECK:STDOUT: !414 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bb3dd30fa3a4aa62:core.Destroy.Core", scope: null, file: !408, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !409, retainedNodes: !417)
// CHECK:STDOUT: !415 = !DILocalVariable(arg: 1, scope: !414, type: !7)
// CHECK:STDOUT: !416 = !DILocation(line: 9, column: 9, scope: !414)
// CHECK:STDOUT: !417 = !{!415}
// CHECK:STDOUT: !418 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb3dd30fa3a4aa62:core.Destroy.Core", scope: null, file: !408, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !409, retainedNodes: !421)
// CHECK:STDOUT: !419 = !DILocalVariable(arg: 1, scope: !418, type: !7)
// CHECK:STDOUT: !420 = !DILocation(line: 9, column: 9, scope: !418)
// CHECK:STDOUT: !421 = !{!419}
// CHECK:STDOUT: !422 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !408, line: 9, type: !13, spFlags: DISPFlagDefinition, unit: !409, retainedNodes: !429)
// CHECK:STDOUT: !423 = !DILocalVariable(arg: 1, scope: !422, type: !7)
// CHECK:STDOUT: !427 = !DILocation(line: 11, column: 10, scope: !422)
// CHECK:STDOUT: !428 = !DILocation(line: 11, column: 3, scope: !422)
// CHECK:STDOUT: !429 = !{!423}
// CHECK:STDOUT: !430 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31", scope: null, file: !16, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !409, retainedNodes: !438)
// CHECK:STDOUT: !431 = !DILocalVariable(arg: 1, scope: !430, type: !7)
// CHECK:STDOUT: !436 = !DILocation(line: 51, column: 12, scope: !430)
// CHECK:STDOUT: !437 = !DILocation(line: 51, column: 5, scope: !430)
// CHECK:STDOUT: !438 = !{!431}
// CHECK:STDOUT: !439 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 147, type: !13, spFlags: DISPFlagDefinition, unit: !409, retainedNodes: !452)
// CHECK:STDOUT: !440 = !DILocalVariable(arg: 1, scope: !439, type: !7)
// CHECK:STDOUT: !445 = !DILocation(line: 148, column: 9, scope: !439)
// CHECK:STDOUT: !446 = !DILocation(line: 148, column: 8, scope: !439)
// CHECK:STDOUT: !447 = !DILocation(line: 149, column: 15, scope: !439)
// CHECK:STDOUT: !448 = !DILocation(line: 149, column: 14, scope: !439)
// CHECK:STDOUT: !449 = !DILocation(line: 149, column: 7, scope: !439)
// CHECK:STDOUT: !450 = !DILocation(line: 151, column: 14, scope: !439)
// CHECK:STDOUT: !451 = !DILocation(line: 151, column: 7, scope: !439)
// CHECK:STDOUT: !452 = !{!440}
// CHECK:STDOUT: !453 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 141, type: !13, spFlags: DISPFlagDefinition, unit: !409, retainedNodes: !460)
// CHECK:STDOUT: !454 = !DILocalVariable(arg: 1, scope: !453, type: !7)
// CHECK:STDOUT: !458 = !DILocation(line: 142, column: 12, scope: !453)
// CHECK:STDOUT: !459 = !DILocation(line: 142, column: 5, scope: !453)
// CHECK:STDOUT: !460 = !{!454}
// CHECK:STDOUT: !461 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !409, retainedNodes: !474)
// CHECK:STDOUT: !462 = !DILocalVariable(arg: 1, scope: !461, type: !7)
// CHECK:STDOUT: !470 = !DILocation(line: 137, column: 5, scope: !461)
// CHECK:STDOUT: !471 = !DILocation(line: 137, column: 28, scope: !461)
// CHECK:STDOUT: !472 = !DILocation(line: 138, column: 5, scope: !461)
// CHECK:STDOUT: !473 = !DILocation(line: 139, column: 5, scope: !461)
// CHECK:STDOUT: !474 = !{!462}
// CHECK:STDOUT: !475 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 125, type: !47, spFlags: DISPFlagDefinition, unit: !409)
// CHECK:STDOUT: !482 = !DILocation(line: 128, column: 5, scope: !475)
// CHECK:STDOUT: !483 = !DILocation(line: 129, column: 5, scope: !475)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %self) #0 !dbg !486 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !488
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !490 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %c), !dbg !495
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !496
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %self) #0 !dbg !498 {
// CHECK:STDOUT:   ret ptr %self, !dbg !505
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!485}
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
// CHECK:STDOUT: !484 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !485 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !484, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !486 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b633cf020343e69:core.Destroy.Core", scope: null, file: !16, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !485, retainedNodes: !489)
// CHECK:STDOUT: !487 = !DILocalVariable(arg: 1, scope: !486, type: !7)
// CHECK:STDOUT: !488 = !DILocation(line: 164, column: 19, scope: !486)
// CHECK:STDOUT: !489 = !{!487}
// CHECK:STDOUT: !490 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !484, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !485, retainedNodes: !497)
// CHECK:STDOUT: !491 = !DILocalVariable(arg: 1, scope: !490, type: !7)
// CHECK:STDOUT: !495 = !DILocation(line: 6, column: 10, scope: !490)
// CHECK:STDOUT: !496 = !DILocation(line: 6, column: 3, scope: !490)
// CHECK:STDOUT: !497 = !{!491}
// CHECK:STDOUT: !498 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608", scope: null, file: !16, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !485, retainedNodes: !506)
// CHECK:STDOUT: !499 = !DILocalVariable(arg: 1, scope: !498, type: !7)
// CHECK:STDOUT: !505 = !DILocation(line: 51, column: 5, scope: !498)
// CHECK:STDOUT: !506 = !{!499}
// CHECK:STDOUT:
