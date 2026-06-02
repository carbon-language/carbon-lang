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
    fn Op[self: Self]() -> Self;
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

// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1fa36cd3e044ac37:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.8e119863da11ee41(ptr %o), !dbg !17
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.8e119863da11ee41(ptr %o), !dbg !19
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.d584e9c3abeb69c0"(ptr %return, i32 %.loc5_12.2), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.8abaa694b657a602(ptr %return), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 4, !dbg !30
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 4, !dbg !31
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 4, !dbg !32
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 4, !dbg !33
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 4, !dbg !34
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 4, !dbg !35
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 4, !dbg !36
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 4, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !30
// CHECK:STDOUT:   call void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.d584e9c3abeb69c0"(ptr %_.var.loc11, i32 %a), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !31
// CHECK:STDOUT:   call void @"_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.4462757cf1994b1f"(ptr %_.var.loc12, i32 %a), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !32
// CHECK:STDOUT:   call void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.5301bd2e5265063e"(ptr %_.var.loc13, i32 %a), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !33
// CHECK:STDOUT:   call void @"_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.bae947e6cb644eb3"(ptr %_.var.loc14, i32 %a), !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !34
// CHECK:STDOUT:   call void @"_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.4462757cf1994b1f"(ptr %_.var.loc15, i32 %b), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !35
// CHECK:STDOUT:   call void @"_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.d312635c3eabfce6"(ptr %_.var.loc16, i32 %b), !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !36
// CHECK:STDOUT:   call void @"_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.bae947e6cb644eb3"(ptr %_.var.loc17, i32 %b), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !37
// CHECK:STDOUT:   call void @"_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.4e32707a29fe420f"(ptr %_.var.loc18, i32 %b), !dbg !37
// CHECK:STDOUT:   call void @"_COp.40c46f69cfcfbfe0:core.Destroy.Core"(ptr %_.var.loc18), !dbg !37
// CHECK:STDOUT:   call void @"_COp.482d7f78e894e7e4:core.Destroy.Core"(ptr %_.var.loc17), !dbg !36
// CHECK:STDOUT:   call void @"_COp.772e6391bfdaa681:core.Destroy.Core"(ptr %_.var.loc16), !dbg !35
// CHECK:STDOUT:   call void @"_COp.06418948197cfd5d:core.Destroy.Core"(ptr %_.var.loc15), !dbg !34
// CHECK:STDOUT:   call void @"_COp.40c46f69cfcfbfe0:core.Destroy.Core"(ptr %_.var.loc14), !dbg !33
// CHECK:STDOUT:   call void @"_COp.482d7f78e894e7e4:core.Destroy.Core"(ptr %_.var.loc13), !dbg !32
// CHECK:STDOUT:   call void @"_COp.772e6391bfdaa681:core.Destroy.Core"(ptr %_.var.loc12), !dbg !31
// CHECK:STDOUT:   call void @"_COp.06418948197cfd5d:core.Destroy.Core"(ptr %_.var.loc11), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e3bae0f289bcf878:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.15c3613c1fae3fe8:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7a29c69bca4c0f8d:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4430ba11e9d8c243:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.482d7f78e894e7e4:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.40c46f69cfcfbfe0:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bc4985fcac159a9a:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6d11ad784a4a1da0:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.084abe736cf37c46:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.06418948197cfd5d:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.772e6391bfdaa681:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.8e119863da11ee41(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.5e5386123d4b6287"(ptr %self), !dbg !89
// CHECK:STDOUT:   ret i1 %1, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.8e119863da11ee41(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.5e5386123d4b6287"(ptr %self), !dbg !94
// CHECK:STDOUT:   ret ptr %1, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.d584e9c3abeb69c0"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !96 {
// CHECK:STDOUT:   call void @"_CConvert.a0c6f87edec590a2:OptionalAs.3a7c842b5a6bca51.Core.8abaa694b657a602"(ptr %return, i32 %self), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.8abaa694b657a602(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !103 {
// CHECK:STDOUT:   call void @"_CNone.c70f21296ce3b72c:OptionalStorage.Core.7ddaa82882073726"(ptr %return), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.4462757cf1994b1f"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !108 {
// CHECK:STDOUT:   call void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.d584e9c3abeb69c0"(ptr %return, i32 %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.5301bd2e5265063e"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !114 {
// CHECK:STDOUT:   call void @"_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.5c5ece92cc304396"(ptr %return, i32 %self), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.bae947e6cb644eb3"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !119 {
// CHECK:STDOUT:   call void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.5301bd2e5265063e"(ptr %return, i32 %self), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.4462757cf1994b1f"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !124 {
// CHECK:STDOUT:   call void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.d584e9c3abeb69c0"(ptr %return, i32 %self), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.d312635c3eabfce6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !129 {
// CHECK:STDOUT:   call void @"_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.4462757cf1994b1f"(ptr %return, i32 %self), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.bae947e6cb644eb3"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !134 {
// CHECK:STDOUT:   call void @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.5301bd2e5265063e"(ptr %return, i32 %self), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.4e32707a29fe420f"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !139 {
// CHECK:STDOUT:   call void @"_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.bae947e6cb644eb3"(ptr %return, i32 %self), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.5e5386123d4b6287"(ptr %value) #0 !dbg !144 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !147
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !148
// CHECK:STDOUT:   ret i1 %2, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.5e5386123d4b6287"(ptr %value) #0 !dbg !150 {
// CHECK:STDOUT:   ret ptr %value, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.a0c6f87edec590a2:OptionalAs.3a7c842b5a6bca51.Core.8abaa694b657a602"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !154 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.8abaa694b657a602(ptr %return, i32 %self), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c70f21296ce3b72c:OptionalStorage.Core.7ddaa82882073726"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !159 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !160
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.5c5ece92cc304396"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !162 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !165
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !165
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.8c6f821d602abaa4"(i32 %self), !dbg !165
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !165
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !165
// CHECK:STDOUT:   call void @_CSome.Optional.Core.664c9635efe58704(ptr %return, i32 %2), !dbg !166
// CHECK:STDOUT:   call void @"_COp.e3bae0f289bcf878:core.Destroy.Core"(ptr %temp), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.8abaa694b657a602(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !168 {
// CHECK:STDOUT:   call void @"_CSome.c70f21296ce3b72c:OptionalStorage.Core.7ddaa82882073726"(ptr %return, i32 %value), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.8c6f821d602abaa4"(i32 %self) #0 !dbg !173 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.667d1d3a0ad99849:ImplicitAs.8b551f392b5250ee.Core.4a088ada889550fc"(i32 %self), !dbg !176
// CHECK:STDOUT:   ret i32 %1, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.664c9635efe58704(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !178 {
// CHECK:STDOUT:   call void @"_CSome.c70f21296ce3b72c:OptionalStorage.Core.66877768e8b2b97c"(ptr %return, i32 %value), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c70f21296ce3b72c:OptionalStorage.Core.7ddaa82882073726"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !183 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !186
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !186
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !187
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !187
// CHECK:STDOUT:   ret void, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.667d1d3a0ad99849:ImplicitAs.8b551f392b5250ee.Core.4a088ada889550fc"(i32 %self) #0 !dbg !189 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !195
// CHECK:STDOUT:   ret i32 %1, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c70f21296ce3b72c:OptionalStorage.Core.66877768e8b2b97c"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !197 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !200
// CHECK:STDOUT:   %1 = call i32 @"_COp.99377c0e95f681c6:Copy.Core.4bee6dcdea6ba756"(i32 %self), !dbg !201
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !200
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !202
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !202
// CHECK:STDOUT:   ret void, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !204 {
// CHECK:STDOUT:   ret i32 %self, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.99377c0e95f681c6:Copy.Core.4bee6dcdea6ba756"(i32 %self) #0 !dbg !208 {
// CHECK:STDOUT:   ret i32 %self, !dbg !212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.482d7f78e894e7e4:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.40c46f69cfcfbfe0:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.06418948197cfd5d:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.772e6391bfdaa681:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.d584e9c3abeb69c0", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 8, 7, 6, 5, 4, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1fa36cd3e044ac37:core.Destroy.Core", scope: null, file: !3, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 3, column: 39, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !3, line: 3, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 4, column: 7, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 4, column: 6, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 5, column: 13, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 5, column: 12, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 5, column: 5, scope: !11)
// CHECK:STDOUT: !22 = !DILocation(line: 7, column: 10, scope: !11)
// CHECK:STDOUT: !23 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !3, line: 10, type: !25, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !25 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !26 = !{null, !7, !7}
// CHECK:STDOUT: !27 = !{!28, !29}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !24, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 11, column: 3, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 12, column: 3, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 13, column: 3, scope: !24)
// CHECK:STDOUT: !33 = !DILocation(line: 14, column: 3, scope: !24)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 3, scope: !24)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 3, scope: !24)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 3, scope: !24)
// CHECK:STDOUT: !37 = !DILocation(line: 18, column: 3, scope: !24)
// CHECK:STDOUT: !38 = !DILocation(line: 10, column: 1, scope: !24)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e3bae0f289bcf878:core.Destroy.Core", scope: null, file: !3, line: 13, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{null, !14}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !39, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 13, column: 10, scope: !39)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.15c3613c1fae3fe8:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !14)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 3, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7a29c69bca4c0f8d:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !14)
// CHECK:STDOUT: !52 = !DILocation(line: 18, column: 3, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4430ba11e9d8c243:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !14)
// CHECK:STDOUT: !56 = !DILocation(line: 18, column: 3, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.482d7f78e894e7e4:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !14)
// CHECK:STDOUT: !60 = !DILocation(line: 18, column: 3, scope: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Op", linkageName: "_COp.40c46f69cfcfbfe0:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !14)
// CHECK:STDOUT: !64 = !DILocation(line: 18, column: 3, scope: !61)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bc4985fcac159a9a:core.Destroy.Core", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !66)
// CHECK:STDOUT: !66 = !{!67}
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !65, type: !7)
// CHECK:STDOUT: !68 = !DILocation(line: 16, column: 3, scope: !65)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6d11ad784a4a1da0:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !14)
// CHECK:STDOUT: !72 = !DILocation(line: 16, column: 3, scope: !69)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Op", linkageName: "_COp.084abe736cf37c46:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !14)
// CHECK:STDOUT: !76 = !DILocation(line: 16, column: 3, scope: !73)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Op", linkageName: "_COp.06418948197cfd5d:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !77, type: !14)
// CHECK:STDOUT: !80 = !DILocation(line: 16, column: 3, scope: !77)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Op", linkageName: "_COp.772e6391bfdaa681:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !14)
// CHECK:STDOUT: !84 = !DILocation(line: 16, column: 3, scope: !81)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.8e119863da11ee41", scope: null, file: !86, line: 36, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !87)
// CHECK:STDOUT: !86 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !87 = !{!88}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !85, type: !14)
// CHECK:STDOUT: !89 = !DILocation(line: 37, column: 12, scope: !85)
// CHECK:STDOUT: !90 = !DILocation(line: 37, column: 5, scope: !85)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.8e119863da11ee41", scope: null, file: !86, line: 39, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !92)
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !91, type: !14)
// CHECK:STDOUT: !94 = !DILocation(line: 40, column: 12, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 40, column: 5, scope: !91)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.d584e9c3abeb69c0", scope: null, file: !86, line: 105, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DISubroutineType(types: !98)
// CHECK:STDOUT: !98 = !{!14, !7}
// CHECK:STDOUT: !99 = !{!100}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !96, type: !7)
// CHECK:STDOUT: !101 = !DILocation(line: 106, column: 12, scope: !96)
// CHECK:STDOUT: !102 = !DILocation(line: 106, column: 5, scope: !96)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.8abaa694b657a602", scope: null, file: !86, line: 30, type: !104, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !104 = !DISubroutineType(types: !105)
// CHECK:STDOUT: !105 = !{!14}
// CHECK:STDOUT: !106 = !DILocation(line: 31, column: 12, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 31, column: 5, scope: !103)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.4462757cf1994b1f", scope: null, file: !109, line: 40, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !110)
// CHECK:STDOUT: !109 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !110 = !{!111}
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !108, type: !7)
// CHECK:STDOUT: !112 = !DILocation(line: 40, column: 45, scope: !108)
// CHECK:STDOUT: !113 = !DILocation(line: 40, column: 38, scope: !108)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.0a82a05610538fbd:ImplicitAs.dfe7f37cd2946174.Core.5301bd2e5265063e", scope: null, file: !86, line: 105, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !115)
// CHECK:STDOUT: !115 = !{!116}
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !114, type: !7)
// CHECK:STDOUT: !117 = !DILocation(line: 106, column: 12, scope: !114)
// CHECK:STDOUT: !118 = !DILocation(line: 106, column: 5, scope: !114)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.bae947e6cb644eb3", scope: null, file: !109, line: 40, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !120)
// CHECK:STDOUT: !120 = !{!121}
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !119, type: !7)
// CHECK:STDOUT: !122 = !DILocation(line: 40, column: 45, scope: !119)
// CHECK:STDOUT: !123 = !DILocation(line: 40, column: 38, scope: !119)
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.4462757cf1994b1f", scope: null, file: !109, line: 44, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !125)
// CHECK:STDOUT: !125 = !{!126}
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !124, type: !7)
// CHECK:STDOUT: !127 = !DILocation(line: 44, column: 45, scope: !124)
// CHECK:STDOUT: !128 = !DILocation(line: 44, column: 38, scope: !124)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.d312635c3eabfce6", scope: null, file: !109, line: 44, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !130)
// CHECK:STDOUT: !130 = !{!131}
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !129, type: !7)
// CHECK:STDOUT: !132 = !DILocation(line: 44, column: 45, scope: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 44, column: 38, scope: !129)
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.bae947e6cb644eb3", scope: null, file: !109, line: 44, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !135)
// CHECK:STDOUT: !135 = !{!136}
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !134, type: !7)
// CHECK:STDOUT: !137 = !DILocation(line: 44, column: 45, scope: !134)
// CHECK:STDOUT: !138 = !DILocation(line: 44, column: 38, scope: !134)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.874935b394d32264:ImplicitAs.eb057aa32837c84e.Core.4e32707a29fe420f", scope: null, file: !109, line: 44, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !140)
// CHECK:STDOUT: !140 = !{!141}
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !139, type: !7)
// CHECK:STDOUT: !142 = !DILocation(line: 44, column: 45, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 44, column: 38, scope: !139)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.5e5386123d4b6287", scope: null, file: !86, line: 172, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !145)
// CHECK:STDOUT: !145 = !{!146}
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !144, type: !14)
// CHECK:STDOUT: !147 = !DILocation(line: 173, column: 16, scope: !144)
// CHECK:STDOUT: !148 = !DILocation(line: 173, column: 12, scope: !144)
// CHECK:STDOUT: !149 = !DILocation(line: 173, column: 5, scope: !144)
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.5e5386123d4b6287", scope: null, file: !86, line: 175, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !151)
// CHECK:STDOUT: !151 = !{!152}
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !150, type: !14)
// CHECK:STDOUT: !153 = !DILocation(line: 176, column: 5, scope: !150)
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a0c6f87edec590a2:OptionalAs.3a7c842b5a6bca51.Core.8abaa694b657a602", scope: null, file: !86, line: 80, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !155)
// CHECK:STDOUT: !155 = !{!156}
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !154, type: !7)
// CHECK:STDOUT: !157 = !DILocation(line: 81, column: 12, scope: !154)
// CHECK:STDOUT: !158 = !DILocation(line: 81, column: 5, scope: !154)
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c70f21296ce3b72c:OptionalStorage.Core.7ddaa82882073726", scope: null, file: !86, line: 125, type: !104, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !160 = !DILocation(line: 128, column: 5, scope: !159)
// CHECK:STDOUT: !161 = !DILocation(line: 129, column: 5, scope: !159)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6bb01b3b35e74e55:OptionalAs.db69f09c2307f298.Core.5c5ece92cc304396", scope: null, file: !86, line: 87, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !163)
// CHECK:STDOUT: !163 = !{!164}
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !162, type: !7)
// CHECK:STDOUT: !165 = !DILocation(line: 88, column: 29, scope: !162)
// CHECK:STDOUT: !166 = !DILocation(line: 88, column: 12, scope: !162)
// CHECK:STDOUT: !167 = !DILocation(line: 88, column: 5, scope: !162)
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.8abaa694b657a602", scope: null, file: !86, line: 33, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !169)
// CHECK:STDOUT: !169 = !{!170}
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !168, type: !7)
// CHECK:STDOUT: !171 = !DILocation(line: 34, column: 12, scope: !168)
// CHECK:STDOUT: !172 = !DILocation(line: 34, column: 5, scope: !168)
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8c14fbb190d2c4a2:ImplicitAs.ad22d1bbc0605210.Core.8c6f821d602abaa4", scope: null, file: !109, line: 40, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !174)
// CHECK:STDOUT: !174 = !{!175}
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !173, type: !7)
// CHECK:STDOUT: !176 = !DILocation(line: 40, column: 45, scope: !173)
// CHECK:STDOUT: !177 = !DILocation(line: 40, column: 38, scope: !173)
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.664c9635efe58704", scope: null, file: !86, line: 33, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !179)
// CHECK:STDOUT: !179 = !{!180}
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !178, type: !7)
// CHECK:STDOUT: !181 = !DILocation(line: 34, column: 12, scope: !178)
// CHECK:STDOUT: !182 = !DILocation(line: 34, column: 5, scope: !178)
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c70f21296ce3b72c:OptionalStorage.Core.7ddaa82882073726", scope: null, file: !86, line: 131, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !184)
// CHECK:STDOUT: !184 = !{!185}
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !183, type: !7)
// CHECK:STDOUT: !186 = !DILocation(line: 137, column: 5, scope: !183)
// CHECK:STDOUT: !187 = !DILocation(line: 138, column: 5, scope: !183)
// CHECK:STDOUT: !188 = !DILocation(line: 139, column: 5, scope: !183)
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.667d1d3a0ad99849:ImplicitAs.8b551f392b5250ee.Core.4a088ada889550fc", scope: null, file: !190, line: 62, type: !191, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !193)
// CHECK:STDOUT: !190 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !191 = !DISubroutineType(types: !192)
// CHECK:STDOUT: !192 = !{!7, !7}
// CHECK:STDOUT: !193 = !{!194}
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !189, type: !7)
// CHECK:STDOUT: !195 = !DILocation(line: 64, column: 17, scope: !189)
// CHECK:STDOUT: !196 = !DILocation(line: 64, column: 5, scope: !189)
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c70f21296ce3b72c:OptionalStorage.Core.66877768e8b2b97c", scope: null, file: !86, line: 131, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !198)
// CHECK:STDOUT: !198 = !{!199}
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !197, type: !7)
// CHECK:STDOUT: !200 = !DILocation(line: 137, column: 5, scope: !197)
// CHECK:STDOUT: !201 = !DILocation(line: 137, column: 28, scope: !197)
// CHECK:STDOUT: !202 = !DILocation(line: 138, column: 5, scope: !197)
// CHECK:STDOUT: !203 = !DILocation(line: 139, column: 5, scope: !197)
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !190, line: 57, type: !191, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !205)
// CHECK:STDOUT: !205 = !{!206}
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !204, type: !7)
// CHECK:STDOUT: !207 = !DILocation(line: 57, column: 36, scope: !204)
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "Op", linkageName: "_COp.99377c0e95f681c6:Copy.Core.4bee6dcdea6ba756", scope: null, file: !209, line: 19, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !210)
// CHECK:STDOUT: !209 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !210 = !{!211}
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 1, scope: !208, type: !7)
// CHECK:STDOUT: !212 = !DILocation(line: 19, column: 33, scope: !208)
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d1d1365d61e0d059:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.3a7c842b5a6bca51.Core:Copy.Core.b21360622e65ae5d"(ptr %return, ptr %c), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.3a7c842b5a6bca51.Core:Copy.Core.b21360622e65ae5d"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !18 {
// CHECK:STDOUT:   call void @"_CCopy.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987"(ptr %return, ptr %self), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !24 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987"(ptr %value), !dbg !27
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   call void @"_CSome.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987"(ptr %return, ptr %value1), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987"(ptr %return), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987"(ptr %value) #0 !dbg !34 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !37
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !37
// CHECK:STDOUT:   ret i1 %2, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !39 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !43
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !44
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !44
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987"(ptr sret({ {}, i1 }) %return) #0 !dbg !46 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d1d1365d61e0d059:core.Destroy.Core", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 9, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 9, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!7, !7}
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 11, column: 10, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 11, column: 3, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.3a7c842b5a6bca51.Core:Copy.Core.b21360622e65ae5d", scope: null, file: !19, line: 50, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 51, column: 12, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 51, column: 5, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987", scope: null, file: !19, line: 147, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 148, column: 9, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 148, column: 8, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 149, column: 15, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 149, column: 14, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 149, column: 7, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 151, column: 14, scope: !24)
// CHECK:STDOUT: !33 = !DILocation(line: 151, column: 7, scope: !24)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987", scope: null, file: !19, line: 141, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 142, column: 12, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 142, column: 5, scope: !34)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987", scope: null, file: !19, line: 131, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 137, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 137, column: 28, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 138, column: 5, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 139, column: 5, scope: !39)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "None", linkageName: "_CNone.2d6936f0ed82c0eb:OptionalStorage.Core.000de1c4710e9987", scope: null, file: !19, line: 125, type: !47, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!7}
// CHECK:STDOUT: !49 = !DILocation(line: 128, column: 5, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 129, column: 5, scope: !46)
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.3a7c842b5a6bca51.Core:Copy.Core.12f52bd835165b7d"(ptr %c), !dbg !10
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.3a7c842b5a6bca51.Core:Copy.Core.12f52bd835165b7d"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT:   ret ptr %self, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.3a7c842b5a6bca51.Core:Copy.Core.12f52bd835165b7d", scope: null, file: !13, line: 50, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !13 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 51, column: 5, scope: !12)
