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

// CHECK:STDOUT: ; ModuleID = 'optional.carbon'
// CHECK:STDOUT: source_filename = "optional.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.03e3348579103d79(ptr %o), !dbg !17
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.03e3348579103d79(ptr %o), !dbg !19
// CHECK:STDOUT:   %.loc18_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.5a0e7187546c44f4"(ptr %return, i32 %.loc18_12.2), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.1cf7a59e6b610f99(ptr %return), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc24 = alloca <{ i32, i1 }>, align 8, !dbg !30
// CHECK:STDOUT:   %_.var.loc25 = alloca <{ i32, i1 }>, align 8, !dbg !31
// CHECK:STDOUT:   %_.var.loc26 = alloca <{ i32, i1 }>, align 8, !dbg !32
// CHECK:STDOUT:   %_.var.loc27 = alloca <{ i32, i1 }>, align 8, !dbg !33
// CHECK:STDOUT:   %_.var.loc28 = alloca <{ i32, i1 }>, align 8, !dbg !34
// CHECK:STDOUT:   %_.var.loc29 = alloca <{ i32, i1 }>, align 8, !dbg !35
// CHECK:STDOUT:   %_.var.loc30 = alloca <{ i32, i1 }>, align 8, !dbg !36
// CHECK:STDOUT:   %_.var.loc31 = alloca <{ i32, i1 }>, align 8, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc24), !dbg !30
// CHECK:STDOUT:   call void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.5a0e7187546c44f4"(ptr %_.var.loc24, i32 %a), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc25), !dbg !31
// CHECK:STDOUT:   call void @"_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.82369a4a757fb6f9"(ptr %_.var.loc25, i32 %a), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc26), !dbg !32
// CHECK:STDOUT:   call void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.afa4bd38e7c54446"(ptr %_.var.loc26, i32 %a), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc27), !dbg !33
// CHECK:STDOUT:   call void @"_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.18692096e0cb24e2"(ptr %_.var.loc27, i32 %a), !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc28), !dbg !34
// CHECK:STDOUT:   call void @"_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.82369a4a757fb6f9"(ptr %_.var.loc28, i32 %b), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc29), !dbg !35
// CHECK:STDOUT:   call void @"_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.4455d2d83f18e89b"(ptr %_.var.loc29, i32 %b), !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc30), !dbg !36
// CHECK:STDOUT:   call void @"_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.18692096e0cb24e2"(ptr %_.var.loc30, i32 %b), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc31), !dbg !37
// CHECK:STDOUT:   call void @"_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.06eea6c30aacdc45"(ptr %_.var.loc31, i32 %b), !dbg !37
// CHECK:STDOUT:   call void @"_COp.0d1642a71859a6de:core.Destroy.Core"(ptr %_.var.loc31), !dbg !37
// CHECK:STDOUT:   call void @"_COp.a254d3e093c759d7:core.Destroy.Core"(ptr %_.var.loc30), !dbg !36
// CHECK:STDOUT:   call void @"_COp.f6c1a73754191bd4:core.Destroy.Core"(ptr %_.var.loc29), !dbg !35
// CHECK:STDOUT:   call void @"_COp.b6ef84402181ba6f:core.Destroy.Core"(ptr %_.var.loc28), !dbg !34
// CHECK:STDOUT:   call void @"_COp.0d1642a71859a6de:core.Destroy.Core"(ptr %_.var.loc27), !dbg !33
// CHECK:STDOUT:   call void @"_COp.a254d3e093c759d7:core.Destroy.Core"(ptr %_.var.loc26), !dbg !32
// CHECK:STDOUT:   call void @"_COp.f6c1a73754191bd4:core.Destroy.Core"(ptr %_.var.loc25), !dbg !31
// CHECK:STDOUT:   call void @"_COp.b6ef84402181ba6f:core.Destroy.Core"(ptr %_.var.loc24), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.957ebacb0fd15edb:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9ec466cd9fb9d70b:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.05fd1694057796b3:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.37b99eb8140c67dc:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a254d3e093c759d7:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0d1642a71859a6de:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b9f59d4d0f116c67:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.59d039ff34f2fb65:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.099822cde4c680b1:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b6ef84402181ba6f:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f6c1a73754191bd4:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.03e3348579103d79(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %self), !dbg !89
// CHECK:STDOUT:   ret i1 %1, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.03e3348579103d79(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %self), !dbg !94
// CHECK:STDOUT:   ret ptr %1, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.5a0e7187546c44f4"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !96 {
// CHECK:STDOUT:   call void @"_CConvert.a15aada625dab765:OptionalAs.0e326e799dad0c64.Core.1cf7a59e6b610f99"(ptr %return, i32 %self), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.1cf7a59e6b610f99(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !103 {
// CHECK:STDOUT:   call void @"_CNone.41c279c2a75ec21d:OptionalStorage.Core.bfb441135f07cbe1"(ptr %return), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.82369a4a757fb6f9"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !108 {
// CHECK:STDOUT:   call void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.5a0e7187546c44f4"(ptr %return, i32 %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.afa4bd38e7c54446"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !114 {
// CHECK:STDOUT:   call void @"_CConvert.81a1ecd9155a3498:OptionalAs.f58b2ba767714873.Core.4d66cd83b8a854d5"(ptr %return, i32 %self), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.18692096e0cb24e2"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !119 {
// CHECK:STDOUT:   call void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.afa4bd38e7c54446"(ptr %return, i32 %self), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.82369a4a757fb6f9"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !124 {
// CHECK:STDOUT:   call void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.5a0e7187546c44f4"(ptr %return, i32 %self), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.4455d2d83f18e89b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !129 {
// CHECK:STDOUT:   call void @"_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.82369a4a757fb6f9"(ptr %return, i32 %self), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.18692096e0cb24e2"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !134 {
// CHECK:STDOUT:   call void @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.afa4bd38e7c54446"(ptr %return, i32 %self), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.06eea6c30aacdc45"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !139 {
// CHECK:STDOUT:   call void @"_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.18692096e0cb24e2"(ptr %return, i32 %self), !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %value) #0 !dbg !144 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !147
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !148
// CHECK:STDOUT:   ret i1 %2, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %value) #0 !dbg !150 {
// CHECK:STDOUT:   ret ptr %value, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.a15aada625dab765:OptionalAs.0e326e799dad0c64.Core.1cf7a59e6b610f99"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !154 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.1cf7a59e6b610f99(ptr %return, i32 %self), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.41c279c2a75ec21d:OptionalStorage.Core.bfb441135f07cbe1"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !159 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !160
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.81a1ecd9155a3498:OptionalAs.f58b2ba767714873.Core.4d66cd83b8a854d5"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !162 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !165
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !165
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.bc6e7cc1f73fc0ac"(i32 %self), !dbg !165
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !165
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !165
// CHECK:STDOUT:   call void @_CSome.Optional.Core.7b25bb243f6b640f(ptr %return, i32 %2), !dbg !166
// CHECK:STDOUT:   call void @"_COp.957ebacb0fd15edb:core.Destroy.Core"(ptr %temp), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.1cf7a59e6b610f99(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !168 {
// CHECK:STDOUT:   call void @"_CSome.41c279c2a75ec21d:OptionalStorage.Core.bfb441135f07cbe1"(ptr %return, i32 %value), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.bc6e7cc1f73fc0ac"(i32 %self) #0 !dbg !173 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.1f02987e9673cc00:ImplicitAs.d993a5132e000405.Core.628184612c80324d"(i32 %self), !dbg !176
// CHECK:STDOUT:   ret i32 %1, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.7b25bb243f6b640f(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !178 {
// CHECK:STDOUT:   call void @"_CSome.41c279c2a75ec21d:OptionalStorage.Core.8a8bbee1ee7003f6"(ptr %return, i32 %value), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.41c279c2a75ec21d:OptionalStorage.Core.bfb441135f07cbe1"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !183 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !186
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !186
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !187
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !187
// CHECK:STDOUT:   ret void, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.1f02987e9673cc00:ImplicitAs.d993a5132e000405.Core.628184612c80324d"(i32 %self) #0 !dbg !189 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !195
// CHECK:STDOUT:   ret i32 %1, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.41c279c2a75ec21d:OptionalStorage.Core.8a8bbee1ee7003f6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !197 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !200
// CHECK:STDOUT:   %1 = call i32 @"_COp.356fbf842fce682d:Copy.Core.64ccbb8e5d9a0b8e"(i32 %self), !dbg !201
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
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.356fbf842fce682d:Copy.Core.64ccbb8e5d9a0b8e"(i32 %self) #0 !dbg !208 {
// CHECK:STDOUT:   ret i32 %self, !dbg !212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.a254d3e093c759d7:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.0d1642a71859a6de:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.b6ef84402181ba6f:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.f6c1a73754191bd4:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.5a0e7187546c44f4", { 0, 1, 3, 2 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "optional.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 16, column: 39, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !3, line: 16, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 7, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 6, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 13, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 12, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 5, scope: !11)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 10, scope: !11)
// CHECK:STDOUT: !23 = !DILocation(line: 20, column: 3, scope: !11)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !3, line: 23, type: !25, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !25 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !26 = !{null, !7, !7}
// CHECK:STDOUT: !27 = !{!28, !29}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !24, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 24, column: 3, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 25, column: 3, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 26, column: 3, scope: !24)
// CHECK:STDOUT: !33 = !DILocation(line: 27, column: 3, scope: !24)
// CHECK:STDOUT: !34 = !DILocation(line: 28, column: 3, scope: !24)
// CHECK:STDOUT: !35 = !DILocation(line: 29, column: 3, scope: !24)
// CHECK:STDOUT: !36 = !DILocation(line: 30, column: 3, scope: !24)
// CHECK:STDOUT: !37 = !DILocation(line: 31, column: 3, scope: !24)
// CHECK:STDOUT: !38 = !DILocation(line: 23, column: 1, scope: !24)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.957ebacb0fd15edb:core.Destroy.Core", scope: null, file: !3, line: 26, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{null, !14}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !39, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 10, scope: !39)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9ec466cd9fb9d70b:core.Destroy.Core", scope: null, file: !3, line: 31, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !14)
// CHECK:STDOUT: !48 = !DILocation(line: 31, column: 3, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.05fd1694057796b3:core.Destroy.Core", scope: null, file: !3, line: 31, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !14)
// CHECK:STDOUT: !52 = !DILocation(line: 31, column: 3, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.37b99eb8140c67dc:core.Destroy.Core", scope: null, file: !3, line: 31, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !14)
// CHECK:STDOUT: !56 = !DILocation(line: 31, column: 3, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a254d3e093c759d7:core.Destroy.Core", scope: null, file: !3, line: 31, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !14)
// CHECK:STDOUT: !60 = !DILocation(line: 31, column: 3, scope: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0d1642a71859a6de:core.Destroy.Core", scope: null, file: !3, line: 31, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !14)
// CHECK:STDOUT: !64 = !DILocation(line: 31, column: 3, scope: !61)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b9f59d4d0f116c67:core.Destroy.Core", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !66)
// CHECK:STDOUT: !66 = !{!67}
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !65, type: !7)
// CHECK:STDOUT: !68 = !DILocation(line: 29, column: 3, scope: !65)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Op", linkageName: "_COp.59d039ff34f2fb65:core.Destroy.Core", scope: null, file: !3, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !14)
// CHECK:STDOUT: !72 = !DILocation(line: 29, column: 3, scope: !69)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Op", linkageName: "_COp.099822cde4c680b1:core.Destroy.Core", scope: null, file: !3, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !14)
// CHECK:STDOUT: !76 = !DILocation(line: 29, column: 3, scope: !73)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b6ef84402181ba6f:core.Destroy.Core", scope: null, file: !3, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !77, type: !14)
// CHECK:STDOUT: !80 = !DILocation(line: 29, column: 3, scope: !77)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f6c1a73754191bd4:core.Destroy.Core", scope: null, file: !3, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !14)
// CHECK:STDOUT: !84 = !DILocation(line: 29, column: 3, scope: !81)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.03e3348579103d79", scope: null, file: !86, line: 33, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !87)
// CHECK:STDOUT: !86 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !87 = !{!88}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !85, type: !14)
// CHECK:STDOUT: !89 = !DILocation(line: 34, column: 12, scope: !85)
// CHECK:STDOUT: !90 = !DILocation(line: 34, column: 5, scope: !85)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.03e3348579103d79", scope: null, file: !86, line: 36, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !92)
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !91, type: !14)
// CHECK:STDOUT: !94 = !DILocation(line: 37, column: 12, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 37, column: 5, scope: !91)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.5a0e7187546c44f4", scope: null, file: !86, line: 96, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DISubroutineType(types: !98)
// CHECK:STDOUT: !98 = !{!14, !7}
// CHECK:STDOUT: !99 = !{!100}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !96, type: !7)
// CHECK:STDOUT: !101 = !DILocation(line: 97, column: 12, scope: !96)
// CHECK:STDOUT: !102 = !DILocation(line: 97, column: 5, scope: !96)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.1cf7a59e6b610f99", scope: null, file: !86, line: 27, type: !104, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !104 = !DISubroutineType(types: !105)
// CHECK:STDOUT: !105 = !{!14}
// CHECK:STDOUT: !106 = !DILocation(line: 28, column: 12, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 28, column: 5, scope: !103)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.82369a4a757fb6f9", scope: null, file: !109, line: 40, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !110)
// CHECK:STDOUT: !109 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !110 = !{!111}
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !108, type: !7)
// CHECK:STDOUT: !112 = !DILocation(line: 40, column: 45, scope: !108)
// CHECK:STDOUT: !113 = !DILocation(line: 40, column: 38, scope: !108)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.81f236ae340cbf98:ImplicitAs.0f95c9e18c91e00a.Core.afa4bd38e7c54446", scope: null, file: !86, line: 96, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !115)
// CHECK:STDOUT: !115 = !{!116}
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !114, type: !7)
// CHECK:STDOUT: !117 = !DILocation(line: 97, column: 12, scope: !114)
// CHECK:STDOUT: !118 = !DILocation(line: 97, column: 5, scope: !114)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.18692096e0cb24e2", scope: null, file: !109, line: 40, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !120)
// CHECK:STDOUT: !120 = !{!121}
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !119, type: !7)
// CHECK:STDOUT: !122 = !DILocation(line: 40, column: 45, scope: !119)
// CHECK:STDOUT: !123 = !DILocation(line: 40, column: 38, scope: !119)
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.82369a4a757fb6f9", scope: null, file: !109, line: 44, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !125)
// CHECK:STDOUT: !125 = !{!126}
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !124, type: !7)
// CHECK:STDOUT: !127 = !DILocation(line: 44, column: 45, scope: !124)
// CHECK:STDOUT: !128 = !DILocation(line: 44, column: 38, scope: !124)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.4455d2d83f18e89b", scope: null, file: !109, line: 44, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !130)
// CHECK:STDOUT: !130 = !{!131}
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !129, type: !7)
// CHECK:STDOUT: !132 = !DILocation(line: 44, column: 45, scope: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 44, column: 38, scope: !129)
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.18692096e0cb24e2", scope: null, file: !109, line: 44, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !135)
// CHECK:STDOUT: !135 = !{!136}
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !134, type: !7)
// CHECK:STDOUT: !137 = !DILocation(line: 44, column: 45, scope: !134)
// CHECK:STDOUT: !138 = !DILocation(line: 44, column: 38, scope: !134)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.b6295decd2c96c07:ImplicitAs.eb057aa32837c84e.Core.06eea6c30aacdc45", scope: null, file: !109, line: 44, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !140)
// CHECK:STDOUT: !140 = !{!141}
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !139, type: !7)
// CHECK:STDOUT: !142 = !DILocation(line: 44, column: 45, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 44, column: 38, scope: !139)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4", scope: null, file: !86, line: 156, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !145)
// CHECK:STDOUT: !145 = !{!146}
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !144, type: !14)
// CHECK:STDOUT: !147 = !DILocation(line: 157, column: 16, scope: !144)
// CHECK:STDOUT: !148 = !DILocation(line: 157, column: 12, scope: !144)
// CHECK:STDOUT: !149 = !DILocation(line: 157, column: 5, scope: !144)
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4", scope: null, file: !86, line: 159, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !151)
// CHECK:STDOUT: !151 = !{!152}
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !150, type: !14)
// CHECK:STDOUT: !153 = !DILocation(line: 160, column: 5, scope: !150)
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a15aada625dab765:OptionalAs.0e326e799dad0c64.Core.1cf7a59e6b610f99", scope: null, file: !86, line: 71, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !155)
// CHECK:STDOUT: !155 = !{!156}
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !154, type: !7)
// CHECK:STDOUT: !157 = !DILocation(line: 72, column: 12, scope: !154)
// CHECK:STDOUT: !158 = !DILocation(line: 72, column: 5, scope: !154)
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "None", linkageName: "_CNone.41c279c2a75ec21d:OptionalStorage.Core.bfb441135f07cbe1", scope: null, file: !86, line: 116, type: !104, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !160 = !DILocation(line: 119, column: 5, scope: !159)
// CHECK:STDOUT: !161 = !DILocation(line: 120, column: 5, scope: !159)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.81a1ecd9155a3498:OptionalAs.f58b2ba767714873.Core.4d66cd83b8a854d5", scope: null, file: !86, line: 78, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !163)
// CHECK:STDOUT: !163 = !{!164}
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !162, type: !7)
// CHECK:STDOUT: !165 = !DILocation(line: 79, column: 29, scope: !162)
// CHECK:STDOUT: !166 = !DILocation(line: 79, column: 12, scope: !162)
// CHECK:STDOUT: !167 = !DILocation(line: 79, column: 5, scope: !162)
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.1cf7a59e6b610f99", scope: null, file: !86, line: 30, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !169)
// CHECK:STDOUT: !169 = !{!170}
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !168, type: !7)
// CHECK:STDOUT: !171 = !DILocation(line: 31, column: 12, scope: !168)
// CHECK:STDOUT: !172 = !DILocation(line: 31, column: 5, scope: !168)
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.7f5e648ca99d8630:ImplicitAs.ad22d1bbc0605210.Core.bc6e7cc1f73fc0ac", scope: null, file: !109, line: 40, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !174)
// CHECK:STDOUT: !174 = !{!175}
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !173, type: !7)
// CHECK:STDOUT: !176 = !DILocation(line: 40, column: 45, scope: !173)
// CHECK:STDOUT: !177 = !DILocation(line: 40, column: 38, scope: !173)
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.7b25bb243f6b640f", scope: null, file: !86, line: 30, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !179)
// CHECK:STDOUT: !179 = !{!180}
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !178, type: !7)
// CHECK:STDOUT: !181 = !DILocation(line: 31, column: 12, scope: !178)
// CHECK:STDOUT: !182 = !DILocation(line: 31, column: 5, scope: !178)
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.41c279c2a75ec21d:OptionalStorage.Core.bfb441135f07cbe1", scope: null, file: !86, line: 122, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !184)
// CHECK:STDOUT: !184 = !{!185}
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !183, type: !7)
// CHECK:STDOUT: !186 = !DILocation(line: 128, column: 5, scope: !183)
// CHECK:STDOUT: !187 = !DILocation(line: 129, column: 5, scope: !183)
// CHECK:STDOUT: !188 = !DILocation(line: 130, column: 5, scope: !183)
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.1f02987e9673cc00:ImplicitAs.d993a5132e000405.Core.628184612c80324d", scope: null, file: !190, line: 62, type: !191, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !193)
// CHECK:STDOUT: !190 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !191 = !DISubroutineType(types: !192)
// CHECK:STDOUT: !192 = !{!7, !7}
// CHECK:STDOUT: !193 = !{!194}
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !189, type: !7)
// CHECK:STDOUT: !195 = !DILocation(line: 64, column: 17, scope: !189)
// CHECK:STDOUT: !196 = !DILocation(line: 64, column: 5, scope: !189)
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.41c279c2a75ec21d:OptionalStorage.Core.8a8bbee1ee7003f6", scope: null, file: !86, line: 122, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !198)
// CHECK:STDOUT: !198 = !{!199}
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !197, type: !7)
// CHECK:STDOUT: !200 = !DILocation(line: 128, column: 5, scope: !197)
// CHECK:STDOUT: !201 = !DILocation(line: 128, column: 28, scope: !197)
// CHECK:STDOUT: !202 = !DILocation(line: 129, column: 5, scope: !197)
// CHECK:STDOUT: !203 = !DILocation(line: 130, column: 5, scope: !197)
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !190, line: 57, type: !191, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !205)
// CHECK:STDOUT: !205 = !{!206}
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !204, type: !7)
// CHECK:STDOUT: !207 = !DILocation(line: 57, column: 36, scope: !204)
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "Op", linkageName: "_COp.356fbf842fce682d:Copy.Core.64ccbb8e5d9a0b8e", scope: null, file: !209, line: 19, type: !97, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !210)
// CHECK:STDOUT: !209 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !210 = !{!211}
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 1, scope: !208, type: !7)
// CHECK:STDOUT: !212 = !DILocation(line: 19, column: 33, scope: !208)
