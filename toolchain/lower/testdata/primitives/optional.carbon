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
// CHECK:STDOUT: define weak_odr void @"_COp.3fe7740c8ce31fb1:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5d78bb11848724da:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e6bd9626cf45e979:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.57bd9f6c8b0ddceb:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.c663fb3583da8fea(ptr %o), !dbg !36
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !37
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.c663fb3583da8fea(ptr %o), !dbg !38
// CHECK:STDOUT:   %.loc18_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !39
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.c79ad1588f12791e"(ptr %return, i32 %.loc18_12.2), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.0c81118a62a79398(ptr %return), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc24 = alloca <{ i32, i1 }>, align 8, !dbg !49
// CHECK:STDOUT:   %_.var.loc25 = alloca <{ i32, i1 }>, align 8, !dbg !50
// CHECK:STDOUT:   %_.var.loc26 = alloca <{ i32, i1 }>, align 8, !dbg !51
// CHECK:STDOUT:   %_.var.loc27 = alloca <{ i32, i1 }>, align 8, !dbg !52
// CHECK:STDOUT:   %_.var.loc28 = alloca <{ i32, i1 }>, align 8, !dbg !53
// CHECK:STDOUT:   %_.var.loc29 = alloca <{ i32, i1 }>, align 8, !dbg !54
// CHECK:STDOUT:   %_.var.loc30 = alloca <{ i32, i1 }>, align 8, !dbg !55
// CHECK:STDOUT:   %_.var.loc31 = alloca <{ i32, i1 }>, align 8, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc24), !dbg !49
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.c79ad1588f12791e"(ptr %_.var.loc24, i32 %a), !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc25), !dbg !50
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.8feec41b87e21c52"(ptr %_.var.loc25, i32 %a), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc26), !dbg !51
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.726ad7e8d60b79ba"(ptr %_.var.loc26, i32 %a), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc27), !dbg !52
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.368ba7a5021eee28"(ptr %_.var.loc27, i32 %a), !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc28), !dbg !53
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.8feec41b87e21c52"(ptr %_.var.loc28, i32 %b), !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc29), !dbg !54
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.a05a3000089e393b"(ptr %_.var.loc29, i32 %b), !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc30), !dbg !55
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.368ba7a5021eee28"(ptr %_.var.loc30, i32 %b), !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc31), !dbg !56
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.987e140306088de9"(ptr %_.var.loc31, i32 %b), !dbg !56
// CHECK:STDOUT:   call void @"_COp.814fcc5cfef7fb5a:core.Destroy.Core"(ptr %_.var.loc31), !dbg !56
// CHECK:STDOUT:   call void @"_COp.28f444f2f46732e4:core.Destroy.Core"(ptr %_.var.loc30), !dbg !55
// CHECK:STDOUT:   call void @"_COp.6d348bed09037202:core.Destroy.Core"(ptr %_.var.loc29), !dbg !54
// CHECK:STDOUT:   call void @"_COp.401b34a5965e6592:core.Destroy.Core"(ptr %_.var.loc28), !dbg !53
// CHECK:STDOUT:   call void @"_COp.814fcc5cfef7fb5a:core.Destroy.Core"(ptr %_.var.loc27), !dbg !52
// CHECK:STDOUT:   call void @"_COp.28f444f2f46732e4:core.Destroy.Core"(ptr %_.var.loc26), !dbg !51
// CHECK:STDOUT:   call void @"_COp.6d348bed09037202:core.Destroy.Core"(ptr %_.var.loc25), !dbg !50
// CHECK:STDOUT:   call void @"_COp.401b34a5965e6592:core.Destroy.Core"(ptr %_.var.loc24), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cf0fdf6659baaa34:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6d0c6cfaed549004:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ff8e65f476832154:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.83837874cf69a650:core.Destroy.Core"(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.28f444f2f46732e4:core.Destroy.Core"(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.814fcc5cfef7fb5a:core.Destroy.Core"(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.401b34a5965e6592:core.Destroy.Core"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6d348bed09037202:core.Destroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.c663fb3583da8fea(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %self), !dbg !93
// CHECK:STDOUT:   ret i1 %1, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.c663fb3583da8fea(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %self), !dbg !98
// CHECK:STDOUT:   ret ptr %1, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.c79ad1588f12791e"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !100 {
// CHECK:STDOUT:   call void @"_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.0c81118a62a79398"(ptr %return, i32 %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.0c81118a62a79398(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !107 {
// CHECK:STDOUT:   call void @"_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr %return), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.8feec41b87e21c52"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !112 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.c79ad1588f12791e"(ptr %return, i32 %self), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.726ad7e8d60b79ba"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !118 {
// CHECK:STDOUT:   call void @"_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.0d08c55dafae1072"(ptr %return, i32 %self), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.368ba7a5021eee28"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !123 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.726ad7e8d60b79ba"(ptr %return, i32 %self), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.8feec41b87e21c52"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !128 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.c79ad1588f12791e"(ptr %return, i32 %self), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.a05a3000089e393b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !133 {
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.8feec41b87e21c52"(ptr %return, i32 %self), !dbg !136
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.368ba7a5021eee28"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !138 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.726ad7e8d60b79ba"(ptr %return, i32 %self), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.987e140306088de9"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !143 {
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.368ba7a5021eee28"(ptr %return, i32 %self), !dbg !146
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %value) #0 !dbg !148 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !151
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !152
// CHECK:STDOUT:   ret i1 %2, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %value) #0 !dbg !154 {
// CHECK:STDOUT:   ret ptr %value, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.0c81118a62a79398"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !158 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.0c81118a62a79398(ptr %return, i32 %self), !dbg !161
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !163 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !164
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !164
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.0d08c55dafae1072"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !166 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !169
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !169
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a"(i32 %self), !dbg !169
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !169
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !169
// CHECK:STDOUT:   call void @_CSome.Optional.Core.c4a9a0e867e9217b(ptr %return, i32 %2), !dbg !170
// CHECK:STDOUT:   call void @"_COp.cf0fdf6659baaa34:core.Destroy.Core"(ptr %temp), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.0c81118a62a79398(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !172 {
// CHECK:STDOUT:   call void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr %return, i32 %value), !dbg !175
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a"(i32 %self) #0 !dbg !177 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401"(i32 %self), !dbg !180
// CHECK:STDOUT:   ret i32 %1, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.c4a9a0e867e9217b(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !182 {
// CHECK:STDOUT:   call void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d"(ptr %return, i32 %value), !dbg !185
// CHECK:STDOUT:   ret void, !dbg !186
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !187 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !190
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !190
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !191
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !191
// CHECK:STDOUT:   ret void, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401"(i32 %self) #0 !dbg !193 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !199
// CHECK:STDOUT:   ret i32 %1, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !201 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !204
// CHECK:STDOUT:   %1 = call i32 @"_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b"(i32 %self), !dbg !205
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !204
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !206
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !206
// CHECK:STDOUT:   ret void, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !208 {
// CHECK:STDOUT:   ret i32 %self, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b"(i32 %self) #0 !dbg !212 {
// CHECK:STDOUT:   ret i32 %self, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.28f444f2f46732e4:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.814fcc5cfef7fb5a:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.401b34a5965e6592:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.6d348bed09037202:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.c79ad1588f12791e", { 0, 1, 3, 2 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3fe7740c8ce31fb1:core.Destroy.Core", scope: null, file: !5, line: 148, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !6 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !7 = !{null, !8}
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 148, column: 19, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 16, column: 39, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5d78bb11848724da:core.Destroy.Core", scope: null, file: !5, line: 115, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 115, column: 19, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e6bd9626cf45e979:core.Destroy.Core", scope: null, file: !5, line: 115, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !8)
// CHECK:STDOUT: !26 = !DILocation(line: 115, column: 19, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.57bd9f6c8b0ddceb:core.Destroy.Core", scope: null, file: !5, line: 115, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !8)
// CHECK:STDOUT: !30 = !DILocation(line: 115, column: 19, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !3, line: 16, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !33 = !{!8, !8}
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !31, type: !8)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 7, scope: !31)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 6, scope: !31)
// CHECK:STDOUT: !38 = !DILocation(line: 18, column: 13, scope: !31)
// CHECK:STDOUT: !39 = !DILocation(line: 18, column: 12, scope: !31)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 5, scope: !31)
// CHECK:STDOUT: !41 = !DILocation(line: 20, column: 10, scope: !31)
// CHECK:STDOUT: !42 = !DILocation(line: 20, column: 3, scope: !31)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !3, line: 23, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !45 = !{null, !15, !15}
// CHECK:STDOUT: !46 = !{!47, !48}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !43, type: !15)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 2, scope: !43, type: !15)
// CHECK:STDOUT: !49 = !DILocation(line: 24, column: 3, scope: !43)
// CHECK:STDOUT: !50 = !DILocation(line: 25, column: 3, scope: !43)
// CHECK:STDOUT: !51 = !DILocation(line: 26, column: 3, scope: !43)
// CHECK:STDOUT: !52 = !DILocation(line: 27, column: 3, scope: !43)
// CHECK:STDOUT: !53 = !DILocation(line: 28, column: 3, scope: !43)
// CHECK:STDOUT: !54 = !DILocation(line: 29, column: 3, scope: !43)
// CHECK:STDOUT: !55 = !DILocation(line: 30, column: 3, scope: !43)
// CHECK:STDOUT: !56 = !DILocation(line: 31, column: 3, scope: !43)
// CHECK:STDOUT: !57 = !DILocation(line: 23, column: 1, scope: !43)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cf0fdf6659baaa34:core.Destroy.Core", scope: null, file: !3, line: 26, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !59)
// CHECK:STDOUT: !59 = !{!60}
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !58, type: !8)
// CHECK:STDOUT: !61 = !DILocation(line: 26, column: 10, scope: !58)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6d0c6cfaed549004:core.Destroy.Core", scope: null, file: !5, line: 115, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !8)
// CHECK:STDOUT: !65 = !DILocation(line: 115, column: 19, scope: !62)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ff8e65f476832154:core.Destroy.Core", scope: null, file: !5, line: 115, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !67)
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !66, type: !8)
// CHECK:STDOUT: !69 = !DILocation(line: 115, column: 19, scope: !66)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Op", linkageName: "_COp.83837874cf69a650:core.Destroy.Core", scope: null, file: !5, line: 115, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !71)
// CHECK:STDOUT: !71 = !{!72}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !70, type: !8)
// CHECK:STDOUT: !73 = !DILocation(line: 115, column: 19, scope: !70)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Op", linkageName: "_COp.28f444f2f46732e4:core.Destroy.Core", scope: null, file: !3, line: 31, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !75)
// CHECK:STDOUT: !75 = !{!76}
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !74, type: !8)
// CHECK:STDOUT: !77 = !DILocation(line: 31, column: 3, scope: !74)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Op", linkageName: "_COp.814fcc5cfef7fb5a:core.Destroy.Core", scope: null, file: !3, line: 31, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !79)
// CHECK:STDOUT: !79 = !{!80}
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !78, type: !8)
// CHECK:STDOUT: !81 = !DILocation(line: 31, column: 3, scope: !78)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Op", linkageName: "_COp.401b34a5965e6592:core.Destroy.Core", scope: null, file: !3, line: 29, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !83)
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !82, type: !8)
// CHECK:STDOUT: !85 = !DILocation(line: 29, column: 3, scope: !82)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6d348bed09037202:core.Destroy.Core", scope: null, file: !3, line: 29, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !87)
// CHECK:STDOUT: !87 = !{!88}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !86, type: !8)
// CHECK:STDOUT: !89 = !DILocation(line: 29, column: 3, scope: !86)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.c663fb3583da8fea", scope: null, file: !5, line: 33, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !91)
// CHECK:STDOUT: !91 = !{!92}
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !90, type: !8)
// CHECK:STDOUT: !93 = !DILocation(line: 34, column: 12, scope: !90)
// CHECK:STDOUT: !94 = !DILocation(line: 34, column: 5, scope: !90)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.c663fb3583da8fea", scope: null, file: !5, line: 36, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !96)
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !95, type: !8)
// CHECK:STDOUT: !98 = !DILocation(line: 37, column: 12, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 37, column: 5, scope: !95)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.c79ad1588f12791e", scope: null, file: !5, line: 96, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !103)
// CHECK:STDOUT: !101 = !DISubroutineType(types: !102)
// CHECK:STDOUT: !102 = !{!8, !15}
// CHECK:STDOUT: !103 = !{!104}
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !100, type: !15)
// CHECK:STDOUT: !105 = !DILocation(line: 97, column: 12, scope: !100)
// CHECK:STDOUT: !106 = !DILocation(line: 97, column: 5, scope: !100)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.0c81118a62a79398", scope: null, file: !5, line: 27, type: !108, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !108 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !109 = !{!8}
// CHECK:STDOUT: !110 = !DILocation(line: 28, column: 12, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 28, column: 5, scope: !107)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.8feec41b87e21c52", scope: null, file: !113, line: 40, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !114)
// CHECK:STDOUT: !113 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !114 = !{!115}
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !112, type: !15)
// CHECK:STDOUT: !116 = !DILocation(line: 40, column: 45, scope: !112)
// CHECK:STDOUT: !117 = !DILocation(line: 40, column: 38, scope: !112)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.726ad7e8d60b79ba", scope: null, file: !5, line: 96, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !119)
// CHECK:STDOUT: !119 = !{!120}
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !118, type: !15)
// CHECK:STDOUT: !121 = !DILocation(line: 97, column: 12, scope: !118)
// CHECK:STDOUT: !122 = !DILocation(line: 97, column: 5, scope: !118)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.368ba7a5021eee28", scope: null, file: !113, line: 40, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !124)
// CHECK:STDOUT: !124 = !{!125}
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !123, type: !15)
// CHECK:STDOUT: !126 = !DILocation(line: 40, column: 45, scope: !123)
// CHECK:STDOUT: !127 = !DILocation(line: 40, column: 38, scope: !123)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.8feec41b87e21c52", scope: null, file: !113, line: 44, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !129)
// CHECK:STDOUT: !129 = !{!130}
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !128, type: !15)
// CHECK:STDOUT: !131 = !DILocation(line: 44, column: 45, scope: !128)
// CHECK:STDOUT: !132 = !DILocation(line: 44, column: 38, scope: !128)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.a05a3000089e393b", scope: null, file: !113, line: 44, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !134)
// CHECK:STDOUT: !134 = !{!135}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !133, type: !15)
// CHECK:STDOUT: !136 = !DILocation(line: 44, column: 45, scope: !133)
// CHECK:STDOUT: !137 = !DILocation(line: 44, column: 38, scope: !133)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.368ba7a5021eee28", scope: null, file: !113, line: 44, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !139)
// CHECK:STDOUT: !139 = !{!140}
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !138, type: !15)
// CHECK:STDOUT: !141 = !DILocation(line: 44, column: 45, scope: !138)
// CHECK:STDOUT: !142 = !DILocation(line: 44, column: 38, scope: !138)
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.987e140306088de9", scope: null, file: !113, line: 44, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !144)
// CHECK:STDOUT: !144 = !{!145}
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !143, type: !15)
// CHECK:STDOUT: !146 = !DILocation(line: 44, column: 45, scope: !143)
// CHECK:STDOUT: !147 = !DILocation(line: 44, column: 38, scope: !143)
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70", scope: null, file: !5, line: 156, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !149)
// CHECK:STDOUT: !149 = !{!150}
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !148, type: !8)
// CHECK:STDOUT: !151 = !DILocation(line: 157, column: 16, scope: !148)
// CHECK:STDOUT: !152 = !DILocation(line: 157, column: 12, scope: !148)
// CHECK:STDOUT: !153 = !DILocation(line: 157, column: 5, scope: !148)
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70", scope: null, file: !5, line: 159, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !155)
// CHECK:STDOUT: !155 = !{!156}
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !154, type: !8)
// CHECK:STDOUT: !157 = !DILocation(line: 160, column: 5, scope: !154)
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.0c81118a62a79398", scope: null, file: !5, line: 71, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !159)
// CHECK:STDOUT: !159 = !{!160}
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !158, type: !15)
// CHECK:STDOUT: !161 = !DILocation(line: 72, column: 12, scope: !158)
// CHECK:STDOUT: !162 = !DILocation(line: 72, column: 5, scope: !158)
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "None", linkageName: "_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6", scope: null, file: !5, line: 116, type: !108, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !164 = !DILocation(line: 119, column: 5, scope: !163)
// CHECK:STDOUT: !165 = !DILocation(line: 120, column: 5, scope: !163)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.0d08c55dafae1072", scope: null, file: !5, line: 78, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !167)
// CHECK:STDOUT: !167 = !{!168}
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !166, type: !15)
// CHECK:STDOUT: !169 = !DILocation(line: 79, column: 29, scope: !166)
// CHECK:STDOUT: !170 = !DILocation(line: 79, column: 12, scope: !166)
// CHECK:STDOUT: !171 = !DILocation(line: 79, column: 5, scope: !166)
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.0c81118a62a79398", scope: null, file: !5, line: 30, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !173)
// CHECK:STDOUT: !173 = !{!174}
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !172, type: !15)
// CHECK:STDOUT: !175 = !DILocation(line: 31, column: 12, scope: !172)
// CHECK:STDOUT: !176 = !DILocation(line: 31, column: 5, scope: !172)
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a", scope: null, file: !113, line: 40, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !178)
// CHECK:STDOUT: !178 = !{!179}
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !177, type: !15)
// CHECK:STDOUT: !180 = !DILocation(line: 40, column: 45, scope: !177)
// CHECK:STDOUT: !181 = !DILocation(line: 40, column: 38, scope: !177)
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.c4a9a0e867e9217b", scope: null, file: !5, line: 30, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !183)
// CHECK:STDOUT: !183 = !{!184}
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !182, type: !15)
// CHECK:STDOUT: !185 = !DILocation(line: 31, column: 12, scope: !182)
// CHECK:STDOUT: !186 = !DILocation(line: 31, column: 5, scope: !182)
// CHECK:STDOUT: !187 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6", scope: null, file: !5, line: 122, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !188)
// CHECK:STDOUT: !188 = !{!189}
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 1, scope: !187, type: !15)
// CHECK:STDOUT: !190 = !DILocation(line: 128, column: 5, scope: !187)
// CHECK:STDOUT: !191 = !DILocation(line: 129, column: 5, scope: !187)
// CHECK:STDOUT: !192 = !DILocation(line: 130, column: 5, scope: !187)
// CHECK:STDOUT: !193 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401", scope: null, file: !194, line: 62, type: !195, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !197)
// CHECK:STDOUT: !194 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !195 = !DISubroutineType(types: !196)
// CHECK:STDOUT: !196 = !{!15, !15}
// CHECK:STDOUT: !197 = !{!198}
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !193, type: !15)
// CHECK:STDOUT: !199 = !DILocation(line: 64, column: 17, scope: !193)
// CHECK:STDOUT: !200 = !DILocation(line: 64, column: 5, scope: !193)
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d", scope: null, file: !5, line: 122, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !202)
// CHECK:STDOUT: !202 = !{!203}
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !201, type: !15)
// CHECK:STDOUT: !204 = !DILocation(line: 128, column: 5, scope: !201)
// CHECK:STDOUT: !205 = !DILocation(line: 128, column: 28, scope: !201)
// CHECK:STDOUT: !206 = !DILocation(line: 129, column: 5, scope: !201)
// CHECK:STDOUT: !207 = !DILocation(line: 130, column: 5, scope: !201)
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !194, line: 57, type: !195, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !209)
// CHECK:STDOUT: !209 = !{!210}
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !208, type: !15)
// CHECK:STDOUT: !211 = !DILocation(line: 57, column: 36, scope: !208)
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b", scope: null, file: !213, line: 19, type: !101, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !214)
// CHECK:STDOUT: !213 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !214 = !{!215}
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !212, type: !15)
// CHECK:STDOUT: !216 = !DILocation(line: 19, column: 33, scope: !212)
