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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %o), !dbg !25
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %o), !dbg !27
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !28
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.b3636752c9f41bbb"(ptr %return, i32 %.loc5_12.2), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.6cf9bfa9f9823eab(ptr %return), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 4, !dbg !42
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 4, !dbg !45
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 4, !dbg !48
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 4, !dbg !51
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 4, !dbg !54
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 4, !dbg !57
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 4, !dbg !60
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 4, !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !42
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.b3636752c9f41bbb"(ptr %_.var.loc11, i32 %a), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !45
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7caccf9eafba4078"(ptr %_.var.loc12, i32 %a), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !48
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.4dd33f7ee0974917"(ptr %_.var.loc13, i32 %a), !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !51
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.f224713919c7eac2"(ptr %_.var.loc14, i32 %a), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !54
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7caccf9eafba4078"(ptr %_.var.loc15, i32 %b), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !57
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.06ecddaeb9f6a433"(ptr %_.var.loc16, i32 %b), !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !60
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.f224713919c7eac2"(ptr %_.var.loc17, i32 %b), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !63
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.1aff9865d83c56a8"(ptr %_.var.loc18, i32 %b), !dbg !63
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.b1865cba8b89abd4(ptr %_.var.loc18), !dbg !63
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.d1a7c96d9b74df0c(ptr %_.var.loc17), !dbg !60
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63f9afd237509454(ptr %_.var.loc16), !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ef06cf32c76f6714(ptr %_.var.loc15), !dbg !54
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.b1865cba8b89abd4(ptr %_.var.loc14), !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.d1a7c96d9b74df0c(ptr %_.var.loc13), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63f9afd237509454(ptr %_.var.loc12), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ef06cf32c76f6714(ptr %_.var.loc11), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0539534c005421db:core.Destroy.Core"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d36a871cb8b5f003:core.Destroy.Core"(ptr %self) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.9a4bad3383e4eb27:core.Destroy.Core"(ptr %self) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0bbdcd5c51cc7e98:core.Destroy.Core"(ptr %self) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b78d24c5d904d0ce:core.Destroy.Core"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.32ec81fcfb7ba5dc:core.Destroy.Core"(ptr %self) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3be8338012ae399:core.Destroy.Core"(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e411fa475c7e35b4:core.Destroy.Core"(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.783c7572445b78b4:core.Destroy.Core"(ptr %self) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.772c3bc04656e8a7:core.Destroy.Core"(ptr %self) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !127
// CHECK:STDOUT:   ret i1 %1, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.6e9e957697159b53(ptr %self) #0 !dbg !130 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !136
// CHECK:STDOUT:   ret ptr %1, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.b3636752c9f41bbb"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !139 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.6cf9bfa9f9823eab"(ptr %return, i32 %self), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.6cf9bfa9f9823eab(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !147 {
// CHECK:STDOUT:   call void @"_CNone.74becf39533834ee:OptionalStorage.Core.b23f311f81e06010"(ptr %return), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7caccf9eafba4078"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !153 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.b3636752c9f41bbb"(ptr %return, i32 %self), !dbg !158
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.4dd33f7ee0974917"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !161 {
// CHECK:STDOUT:   call void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.9d96311e3f0f3a1f"(ptr %return, i32 %self), !dbg !166
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.f224713919c7eac2"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !169 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.4dd33f7ee0974917"(ptr %return, i32 %self), !dbg !174
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7caccf9eafba4078"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !177 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.b3636752c9f41bbb"(ptr %return, i32 %self), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.06ecddaeb9f6a433"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !186 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7caccf9eafba4078"(ptr %return, i32 %self), !dbg !192
// CHECK:STDOUT:   ret void, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.f224713919c7eac2"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !195 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.4dd33f7ee0974917"(ptr %return, i32 %self), !dbg !201
// CHECK:STDOUT:   ret void, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.1aff9865d83c56a8"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !204 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.f224713919c7eac2"(ptr %return, i32 %self), !dbg !210
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.b1865cba8b89abd4(ptr %self) #0 !dbg !214 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b78d24c5d904d0ce:core.Destroy.Core"(ptr %self), !dbg !218
// CHECK:STDOUT:   ret void, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.d1a7c96d9b74df0c(ptr %self) #0 !dbg !221 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0bbdcd5c51cc7e98:core.Destroy.Core"(ptr %self), !dbg !225
// CHECK:STDOUT:   ret void, !dbg !226
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.63f9afd237509454(ptr %self) #0 !dbg !228 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.772c3bc04656e8a7:core.Destroy.Core"(ptr %self), !dbg !232
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ef06cf32c76f6714(ptr %self) #0 !dbg !235 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.783c7572445b78b4:core.Destroy.Core"(ptr %self), !dbg !239
// CHECK:STDOUT:   ret void, !dbg !240
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !242 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !248
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !249
// CHECK:STDOUT:   ret i1 %2, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !252 {
// CHECK:STDOUT:   ret ptr %value, !dbg !258
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.6cf9bfa9f9823eab"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !260 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.6cf9bfa9f9823eab(ptr %return, i32 %self), !dbg !266
// CHECK:STDOUT:   ret void, !dbg !267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.74becf39533834ee:OptionalStorage.Core.b23f311f81e06010"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !269 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !276
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !276
// CHECK:STDOUT:   ret void, !dbg !277
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.9d96311e3f0f3a1f"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !278 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !283
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !283
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self), !dbg !283
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !283
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !283
// CHECK:STDOUT:   call void @_CSome.Optional.Core.faf9bdf124620749(ptr %return, i32 %2), !dbg !284
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.702b280ef75398d8(ptr %temp), !dbg !283
// CHECK:STDOUT:   ret void, !dbg !285
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.6cf9bfa9f9823eab(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !287 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.b23f311f81e06010"(ptr %return, i32 %value), !dbg !292
// CHECK:STDOUT:   ret void, !dbg !293
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2"(i32 %self) #0 !dbg !295 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self), !dbg !300
// CHECK:STDOUT:   ret i32 %1, !dbg !303
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.faf9bdf124620749(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !305 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.b994256de8d01720"(ptr %return, i32 %value), !dbg !310
// CHECK:STDOUT:   ret void, !dbg !311
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.702b280ef75398d8(ptr %self) #0 !dbg !313 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core"(ptr %self), !dbg !317
// CHECK:STDOUT:   ret void, !dbg !318
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.b23f311f81e06010"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !320 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !329
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !329
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !331
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !331
// CHECK:STDOUT:   ret void, !dbg !332
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !335 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !340
// CHECK:STDOUT:   ret i32 %1, !dbg !342
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.b994256de8d01720"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !344 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !353
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self), !dbg !354
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !353
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !355
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !355
// CHECK:STDOUT:   ret void, !dbg !356
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !358 {
// CHECK:STDOUT:   ret i32 %self, !dbg !364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4"(i32 %self) #0 !dbg !367 {
// CHECK:STDOUT:   ret i32 %self, !dbg !374
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.b3636752c9f41bbb", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 8, 7, 6, 5, 4, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.b1865cba8b89abd4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.d1a7c96d9b74df0c, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.63f9afd237509454, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.ef06cf32c76f6714, { 1, 0 }
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
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{!9, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !9}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 3, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !19)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !16, type: !6)
// CHECK:STDOUT: !18 = !DILocation(line: 3, column: 39, scope: !16)
// CHECK:STDOUT: !19 = !{!17}
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !36)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !20, type: !9)
// CHECK:STDOUT: !25 = !DILocation(line: 4, column: 7, scope: !20)
// CHECK:STDOUT: !26 = !DILocation(line: 4, column: 6, scope: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 5, column: 13, scope: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 5, column: 12, scope: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 5, column: 5, scope: !20)
// CHECK:STDOUT: !30 = !{!9, !6}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 7, column: 10, scope: !20)
// CHECK:STDOUT: !33 = !{!9}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 7, column: 3, scope: !20)
// CHECK:STDOUT: !36 = !{!21}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !4, line: 10, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !67)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !6)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !37, type: !6)
// CHECK:STDOUT: !42 = !DILocation(line: 11, column: 3, scope: !37)
// CHECK:STDOUT: !45 = !DILocation(line: 12, column: 3, scope: !37)
// CHECK:STDOUT: !48 = !DILocation(line: 13, column: 3, scope: !37)
// CHECK:STDOUT: !51 = !DILocation(line: 14, column: 3, scope: !37)
// CHECK:STDOUT: !54 = !DILocation(line: 15, column: 3, scope: !37)
// CHECK:STDOUT: !57 = !DILocation(line: 16, column: 3, scope: !37)
// CHECK:STDOUT: !60 = !DILocation(line: 17, column: 3, scope: !37)
// CHECK:STDOUT: !63 = !DILocation(line: 18, column: 3, scope: !37)
// CHECK:STDOUT: !66 = !DILocation(line: 10, column: 1, scope: !37)
// CHECK:STDOUT: !67 = !{!38, !39}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6a8d754fe1453eab:core.Destroy.Core", scope: null, file: !4, line: 13, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !9)
// CHECK:STDOUT: !70 = !DILocation(line: 13, column: 10, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.96a347e4ca9c10e1:core.Destroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !9)
// CHECK:STDOUT: !74 = !DILocation(line: 18, column: 3, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0539534c005421db:core.Destroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !79)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !9)
// CHECK:STDOUT: !78 = !DILocation(line: 18, column: 3, scope: !76)
// CHECK:STDOUT: !79 = !{!77}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d36a871cb8b5f003:core.Destroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !83)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !9)
// CHECK:STDOUT: !82 = !DILocation(line: 18, column: 3, scope: !80)
// CHECK:STDOUT: !83 = !{!81}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.9a4bad3383e4eb27:core.Destroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !9)
// CHECK:STDOUT: !86 = !DILocation(line: 18, column: 3, scope: !84)
// CHECK:STDOUT: !87 = !{!85}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0bbdcd5c51cc7e98:core.Destroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !91)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !9)
// CHECK:STDOUT: !90 = !DILocation(line: 18, column: 3, scope: !88)
// CHECK:STDOUT: !91 = !{!89}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b78d24c5d904d0ce:core.Destroy.Core", scope: null, file: !4, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !9)
// CHECK:STDOUT: !94 = !DILocation(line: 18, column: 3, scope: !92)
// CHECK:STDOUT: !95 = !{!93}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 16, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !9)
// CHECK:STDOUT: !98 = !DILocation(line: 16, column: 3, scope: !96)
// CHECK:STDOUT: !99 = !{!97}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.32ec81fcfb7ba5dc:core.Destroy.Core", scope: null, file: !4, line: 16, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !103)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !6)
// CHECK:STDOUT: !102 = !DILocation(line: 16, column: 3, scope: !100)
// CHECK:STDOUT: !103 = !{!101}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3be8338012ae399:core.Destroy.Core", scope: null, file: !4, line: 16, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !107)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !9)
// CHECK:STDOUT: !106 = !DILocation(line: 16, column: 3, scope: !104)
// CHECK:STDOUT: !107 = !{!105}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e411fa475c7e35b4:core.Destroy.Core", scope: null, file: !4, line: 16, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !111)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !9)
// CHECK:STDOUT: !110 = !DILocation(line: 16, column: 3, scope: !108)
// CHECK:STDOUT: !111 = !{!109}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.783c7572445b78b4:core.Destroy.Core", scope: null, file: !4, line: 16, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !9)
// CHECK:STDOUT: !114 = !DILocation(line: 16, column: 3, scope: !112)
// CHECK:STDOUT: !115 = !{!113}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.772c3bc04656e8a7:core.Destroy.Core", scope: null, file: !4, line: 16, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !119)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !9)
// CHECK:STDOUT: !118 = !DILocation(line: 16, column: 3, scope: !116)
// CHECK:STDOUT: !119 = !{!117}
// CHECK:STDOUT: !120 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.6e9e957697159b53", scope: null, file: !120, line: 36, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !129)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !9)
// CHECK:STDOUT: !127 = !DILocation(line: 37, column: 12, scope: !121)
// CHECK:STDOUT: !128 = !DILocation(line: 37, column: 5, scope: !121)
// CHECK:STDOUT: !129 = !{!122}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.6e9e957697159b53", scope: null, file: !120, line: 39, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !138)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !9)
// CHECK:STDOUT: !136 = !DILocation(line: 40, column: 12, scope: !130)
// CHECK:STDOUT: !137 = !DILocation(line: 40, column: 5, scope: !130)
// CHECK:STDOUT: !138 = !{!131}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.b3636752c9f41bbb", scope: null, file: !120, line: 105, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !146)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !6)
// CHECK:STDOUT: !144 = !DILocation(line: 106, column: 12, scope: !139)
// CHECK:STDOUT: !145 = !DILocation(line: 106, column: 5, scope: !139)
// CHECK:STDOUT: !146 = !{!140}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.6cf9bfa9f9823eab", scope: null, file: !120, line: 30, type: !34, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !150 = !DILocation(line: 31, column: 12, scope: !147)
// CHECK:STDOUT: !151 = !DILocation(line: 31, column: 5, scope: !147)
// CHECK:STDOUT: !152 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7caccf9eafba4078", scope: null, file: !152, line: 43, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !160)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !6)
// CHECK:STDOUT: !158 = !DILocation(line: 43, column: 40, scope: !153)
// CHECK:STDOUT: !159 = !DILocation(line: 43, column: 33, scope: !153)
// CHECK:STDOUT: !160 = !{!154}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.4dd33f7ee0974917", scope: null, file: !120, line: 105, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !168)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !6)
// CHECK:STDOUT: !166 = !DILocation(line: 106, column: 12, scope: !161)
// CHECK:STDOUT: !167 = !DILocation(line: 106, column: 5, scope: !161)
// CHECK:STDOUT: !168 = !{!162}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.f224713919c7eac2", scope: null, file: !152, line: 43, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !176)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !6)
// CHECK:STDOUT: !174 = !DILocation(line: 43, column: 40, scope: !169)
// CHECK:STDOUT: !175 = !DILocation(line: 43, column: 33, scope: !169)
// CHECK:STDOUT: !176 = !{!170}
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7caccf9eafba4078", scope: null, file: !152, line: 47, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !185)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !177, type: !6)
// CHECK:STDOUT: !183 = !DILocation(line: 47, column: 34, scope: !177)
// CHECK:STDOUT: !184 = !DILocation(line: 47, column: 27, scope: !177)
// CHECK:STDOUT: !185 = !{!178}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.06ecddaeb9f6a433", scope: null, file: !152, line: 47, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !194)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !6)
// CHECK:STDOUT: !192 = !DILocation(line: 47, column: 34, scope: !186)
// CHECK:STDOUT: !193 = !DILocation(line: 47, column: 27, scope: !186)
// CHECK:STDOUT: !194 = !{!187}
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.f224713919c7eac2", scope: null, file: !152, line: 47, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !203)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !6)
// CHECK:STDOUT: !201 = !DILocation(line: 47, column: 34, scope: !195)
// CHECK:STDOUT: !202 = !DILocation(line: 47, column: 27, scope: !195)
// CHECK:STDOUT: !203 = !{!196}
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.1aff9865d83c56a8", scope: null, file: !152, line: 47, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !212)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 1, scope: !204, type: !6)
// CHECK:STDOUT: !210 = !DILocation(line: 47, column: 34, scope: !204)
// CHECK:STDOUT: !211 = !DILocation(line: 47, column: 27, scope: !204)
// CHECK:STDOUT: !212 = !{!205}
// CHECK:STDOUT: !213 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.b1865cba8b89abd4", scope: null, file: !213, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !220)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !9)
// CHECK:STDOUT: !218 = !DILocation(line: 28, column: 5, scope: !214)
// CHECK:STDOUT: !219 = !DILocation(line: 26, column: 3, scope: !214)
// CHECK:STDOUT: !220 = !{!215}
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.d1a7c96d9b74df0c", scope: null, file: !213, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !227)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !9)
// CHECK:STDOUT: !225 = !DILocation(line: 28, column: 5, scope: !221)
// CHECK:STDOUT: !226 = !DILocation(line: 26, column: 3, scope: !221)
// CHECK:STDOUT: !227 = !{!222}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.63f9afd237509454", scope: null, file: !213, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !234)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !9)
// CHECK:STDOUT: !232 = !DILocation(line: 28, column: 5, scope: !228)
// CHECK:STDOUT: !233 = !DILocation(line: 26, column: 3, scope: !228)
// CHECK:STDOUT: !234 = !{!229}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ef06cf32c76f6714", scope: null, file: !213, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !241)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !9)
// CHECK:STDOUT: !239 = !DILocation(line: 28, column: 5, scope: !235)
// CHECK:STDOUT: !240 = !DILocation(line: 26, column: 3, scope: !235)
// CHECK:STDOUT: !241 = !{!236}
// CHECK:STDOUT: !242 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !120, line: 172, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !251)
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 1, scope: !242, type: !9)
// CHECK:STDOUT: !248 = !DILocation(line: 173, column: 16, scope: !242)
// CHECK:STDOUT: !249 = !DILocation(line: 173, column: 12, scope: !242)
// CHECK:STDOUT: !250 = !DILocation(line: 173, column: 5, scope: !242)
// CHECK:STDOUT: !251 = !{!243}
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !120, line: 175, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !259)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 1, scope: !252, type: !9)
// CHECK:STDOUT: !258 = !DILocation(line: 176, column: 5, scope: !252)
// CHECK:STDOUT: !259 = !{!253}
// CHECK:STDOUT: !260 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.6cf9bfa9f9823eab", scope: null, file: !120, line: 80, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !268)
// CHECK:STDOUT: !261 = !DILocalVariable(arg: 1, scope: !260, type: !6)
// CHECK:STDOUT: !266 = !DILocation(line: 81, column: 12, scope: !260)
// CHECK:STDOUT: !267 = !DILocation(line: 81, column: 5, scope: !260)
// CHECK:STDOUT: !268 = !{!261}
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "None", linkageName: "_CNone.74becf39533834ee:OptionalStorage.Core.b23f311f81e06010", scope: null, file: !120, line: 125, type: !34, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !276 = !DILocation(line: 128, column: 5, scope: !269)
// CHECK:STDOUT: !277 = !DILocation(line: 129, column: 5, scope: !269)
// CHECK:STDOUT: !278 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.9d96311e3f0f3a1f", scope: null, file: !120, line: 87, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !286)
// CHECK:STDOUT: !279 = !DILocalVariable(arg: 1, scope: !278, type: !6)
// CHECK:STDOUT: !283 = !DILocation(line: 88, column: 29, scope: !278)
// CHECK:STDOUT: !284 = !DILocation(line: 88, column: 12, scope: !278)
// CHECK:STDOUT: !285 = !DILocation(line: 88, column: 5, scope: !278)
// CHECK:STDOUT: !286 = !{!279}
// CHECK:STDOUT: !287 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.6cf9bfa9f9823eab", scope: null, file: !120, line: 33, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !294)
// CHECK:STDOUT: !288 = !DILocalVariable(arg: 1, scope: !287, type: !6)
// CHECK:STDOUT: !292 = !DILocation(line: 34, column: 12, scope: !287)
// CHECK:STDOUT: !293 = !DILocation(line: 34, column: 5, scope: !287)
// CHECK:STDOUT: !294 = !{!288}
// CHECK:STDOUT: !295 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.ba7abeb9eee545e2", scope: null, file: !152, line: 43, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !304)
// CHECK:STDOUT: !296 = !DILocalVariable(arg: 1, scope: !295, type: !6)
// CHECK:STDOUT: !300 = !DILocation(line: 43, column: 40, scope: !295)
// CHECK:STDOUT: !301 = !{!6, !6}
// CHECK:STDOUT: !302 = !DISubroutineType(types: !301)
// CHECK:STDOUT: !303 = !DILocation(line: 43, column: 33, scope: !295)
// CHECK:STDOUT: !304 = !{!296}
// CHECK:STDOUT: !305 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.faf9bdf124620749", scope: null, file: !120, line: 33, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !312)
// CHECK:STDOUT: !306 = !DILocalVariable(arg: 1, scope: !305, type: !6)
// CHECK:STDOUT: !310 = !DILocation(line: 34, column: 12, scope: !305)
// CHECK:STDOUT: !311 = !DILocation(line: 34, column: 5, scope: !305)
// CHECK:STDOUT: !312 = !{!306}
// CHECK:STDOUT: !313 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.702b280ef75398d8", scope: null, file: !213, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !319)
// CHECK:STDOUT: !314 = !DILocalVariable(arg: 1, scope: !313, type: !9)
// CHECK:STDOUT: !317 = !DILocation(line: 28, column: 5, scope: !313)
// CHECK:STDOUT: !318 = !DILocation(line: 26, column: 3, scope: !313)
// CHECK:STDOUT: !319 = !{!314}
// CHECK:STDOUT: !320 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.b23f311f81e06010", scope: null, file: !120, line: 131, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !333)
// CHECK:STDOUT: !321 = !DILocalVariable(arg: 1, scope: !320, type: !6)
// CHECK:STDOUT: !329 = !DILocation(line: 137, column: 5, scope: !320)
// CHECK:STDOUT: !331 = !DILocation(line: 138, column: 5, scope: !320)
// CHECK:STDOUT: !332 = !DILocation(line: 139, column: 5, scope: !320)
// CHECK:STDOUT: !333 = !{!321}
// CHECK:STDOUT: !334 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !335 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !334, line: 64, type: !302, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !343)
// CHECK:STDOUT: !336 = !DILocalVariable(arg: 1, scope: !335, type: !6)
// CHECK:STDOUT: !340 = !DILocation(line: 66, column: 17, scope: !335)
// CHECK:STDOUT: !342 = !DILocation(line: 66, column: 5, scope: !335)
// CHECK:STDOUT: !343 = !{!336}
// CHECK:STDOUT: !344 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.b994256de8d01720", scope: null, file: !120, line: 131, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !357)
// CHECK:STDOUT: !345 = !DILocalVariable(arg: 1, scope: !344, type: !6)
// CHECK:STDOUT: !353 = !DILocation(line: 137, column: 5, scope: !344)
// CHECK:STDOUT: !354 = !DILocation(line: 137, column: 28, scope: !344)
// CHECK:STDOUT: !355 = !DILocation(line: 138, column: 5, scope: !344)
// CHECK:STDOUT: !356 = !DILocation(line: 139, column: 5, scope: !344)
// CHECK:STDOUT: !357 = !{!345}
// CHECK:STDOUT: !358 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !334, line: 59, type: !302, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !365)
// CHECK:STDOUT: !359 = !DILocalVariable(arg: 1, scope: !358, type: !6)
// CHECK:STDOUT: !364 = !DILocation(line: 59, column: 28, scope: !358)
// CHECK:STDOUT: !365 = !{!359}
// CHECK:STDOUT: !366 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !367 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.4d6ceefeb7f772f4", scope: null, file: !366, line: 23, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !375)
// CHECK:STDOUT: !368 = !DILocalVariable(arg: 1, scope: !367, type: !6)
// CHECK:STDOUT: !374 = !DILocation(line: 23, column: 25, scope: !367)
// CHECK:STDOUT: !375 = !{!368}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bb3dd30fa3a4aa62:core.Destroy.Core"(ptr %self) #0 !dbg !378 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !380
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !382 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.0dba8744366cba59"(ptr %return, ptr %c), !dbg !387
// CHECK:STDOUT:   ret void, !dbg !388
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.0dba8744366cba59"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !390 {
// CHECK:STDOUT:   call void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944"(ptr %return, ptr %self), !dbg !396
// CHECK:STDOUT:   ret void, !dbg !397
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !399 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944"(ptr %value), !dbg !405
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !406
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !407
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944"(ptr %return, ptr %value1), !dbg !408
// CHECK:STDOUT:   ret void, !dbg !409
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944"(ptr %return), !dbg !410
// CHECK:STDOUT:   ret void, !dbg !411
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944"(ptr %value) #0 !dbg !413 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !418
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !418
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !418
// CHECK:STDOUT:   ret i1 %2, !dbg !419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !421 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !430
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !431
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !432
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !432
// CHECK:STDOUT:   ret void, !dbg !433
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944"(ptr sret({ {}, i1 }) %return) #0 !dbg !435 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !442
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !442
// CHECK:STDOUT:   ret void, !dbg !443
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!377}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{!9, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !14 = !{null, !9}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !33 = !{!9}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !120 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !376 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !377 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !376, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !378 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bb3dd30fa3a4aa62:core.Destroy.Core", scope: null, file: !376, line: 9, type: !15, spFlags: DISPFlagDefinition, unit: !377, retainedNodes: !381)
// CHECK:STDOUT: !379 = !DILocalVariable(arg: 1, scope: !378, type: !9)
// CHECK:STDOUT: !380 = !DILocation(line: 9, column: 9, scope: !378)
// CHECK:STDOUT: !381 = !{!379}
// CHECK:STDOUT: !382 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !376, line: 9, type: !11, spFlags: DISPFlagDefinition, unit: !377, retainedNodes: !389)
// CHECK:STDOUT: !383 = !DILocalVariable(arg: 1, scope: !382, type: !9)
// CHECK:STDOUT: !387 = !DILocation(line: 11, column: 10, scope: !382)
// CHECK:STDOUT: !388 = !DILocation(line: 11, column: 3, scope: !382)
// CHECK:STDOUT: !389 = !{!383}
// CHECK:STDOUT: !390 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.0dba8744366cba59", scope: null, file: !120, line: 50, type: !11, spFlags: DISPFlagDefinition, unit: !377, retainedNodes: !398)
// CHECK:STDOUT: !391 = !DILocalVariable(arg: 1, scope: !390, type: !9)
// CHECK:STDOUT: !396 = !DILocation(line: 51, column: 12, scope: !390)
// CHECK:STDOUT: !397 = !DILocation(line: 51, column: 5, scope: !390)
// CHECK:STDOUT: !398 = !{!391}
// CHECK:STDOUT: !399 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944", scope: null, file: !120, line: 147, type: !11, spFlags: DISPFlagDefinition, unit: !377, retainedNodes: !412)
// CHECK:STDOUT: !400 = !DILocalVariable(arg: 1, scope: !399, type: !9)
// CHECK:STDOUT: !405 = !DILocation(line: 148, column: 9, scope: !399)
// CHECK:STDOUT: !406 = !DILocation(line: 148, column: 8, scope: !399)
// CHECK:STDOUT: !407 = !DILocation(line: 149, column: 15, scope: !399)
// CHECK:STDOUT: !408 = !DILocation(line: 149, column: 14, scope: !399)
// CHECK:STDOUT: !409 = !DILocation(line: 149, column: 7, scope: !399)
// CHECK:STDOUT: !410 = !DILocation(line: 151, column: 14, scope: !399)
// CHECK:STDOUT: !411 = !DILocation(line: 151, column: 7, scope: !399)
// CHECK:STDOUT: !412 = !{!400}
// CHECK:STDOUT: !413 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944", scope: null, file: !120, line: 141, type: !11, spFlags: DISPFlagDefinition, unit: !377, retainedNodes: !420)
// CHECK:STDOUT: !414 = !DILocalVariable(arg: 1, scope: !413, type: !9)
// CHECK:STDOUT: !418 = !DILocation(line: 142, column: 12, scope: !413)
// CHECK:STDOUT: !419 = !DILocation(line: 142, column: 5, scope: !413)
// CHECK:STDOUT: !420 = !{!414}
// CHECK:STDOUT: !421 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944", scope: null, file: !120, line: 131, type: !11, spFlags: DISPFlagDefinition, unit: !377, retainedNodes: !434)
// CHECK:STDOUT: !422 = !DILocalVariable(arg: 1, scope: !421, type: !9)
// CHECK:STDOUT: !430 = !DILocation(line: 137, column: 5, scope: !421)
// CHECK:STDOUT: !431 = !DILocation(line: 137, column: 28, scope: !421)
// CHECK:STDOUT: !432 = !DILocation(line: 138, column: 5, scope: !421)
// CHECK:STDOUT: !433 = !DILocation(line: 139, column: 5, scope: !421)
// CHECK:STDOUT: !434 = !{!422}
// CHECK:STDOUT: !435 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.efddd55507031944", scope: null, file: !120, line: 125, type: !34, spFlags: DISPFlagDefinition, unit: !377)
// CHECK:STDOUT: !442 = !DILocation(line: 128, column: 5, scope: !435)
// CHECK:STDOUT: !443 = !DILocation(line: 129, column: 5, scope: !435)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !446 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %c), !dbg !451
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !452
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %self) #0 !dbg !454 {
// CHECK:STDOUT:   ret ptr %self, !dbg !461
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!445}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{!9, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !120 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !444 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !445 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !444, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !446 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !444, line: 5, type: !11, spFlags: DISPFlagDefinition, unit: !445, retainedNodes: !453)
// CHECK:STDOUT: !447 = !DILocalVariable(arg: 1, scope: !446, type: !9)
// CHECK:STDOUT: !451 = !DILocation(line: 6, column: 10, scope: !446)
// CHECK:STDOUT: !452 = !DILocation(line: 6, column: 3, scope: !446)
// CHECK:STDOUT: !453 = !{!447}
// CHECK:STDOUT: !454 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608", scope: null, file: !120, line: 50, type: !11, spFlags: DISPFlagDefinition, unit: !445, retainedNodes: !462)
// CHECK:STDOUT: !455 = !DILocalVariable(arg: 1, scope: !454, type: !9)
// CHECK:STDOUT: !461 = !DILocation(line: 51, column: 5, scope: !454)
// CHECK:STDOUT: !462 = !{!455}
// CHECK:STDOUT:
