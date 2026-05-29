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
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.36b54e287793a2ba(ptr %o), !dbg !17
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.36b54e287793a2ba(ptr %o), !dbg !19
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.d356e073ce1a5b35"(ptr %return, i32 %.loc5_12.2), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.cf11014f74122042(ptr %return), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 8, !dbg !30
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 8, !dbg !31
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 8, !dbg !32
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 8, !dbg !33
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 8, !dbg !34
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 8, !dbg !35
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 8, !dbg !36
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 8, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !30
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.d356e073ce1a5b35"(ptr %_.var.loc11, i32 %a), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !31
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.93e3eaa0d3eea476"(ptr %_.var.loc12, i32 %a), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !32
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.a06e83c323a306da"(ptr %_.var.loc13, i32 %a), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !33
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.183173080affdcb8"(ptr %_.var.loc14, i32 %a), !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !34
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.93e3eaa0d3eea476"(ptr %_.var.loc15, i32 %b), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !35
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.54bcc98de4b4d68e"(ptr %_.var.loc16, i32 %b), !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !36
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.183173080affdcb8"(ptr %_.var.loc17, i32 %b), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !37
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.53171a0379a201a0"(ptr %_.var.loc18, i32 %b), !dbg !37
// CHECK:STDOUT:   call void @"_COp.adbfbb7b79f63468:core.Destroy.Core"(ptr %_.var.loc18), !dbg !37
// CHECK:STDOUT:   call void @"_COp.8ed16437b3e9f3a5:core.Destroy.Core"(ptr %_.var.loc17), !dbg !36
// CHECK:STDOUT:   call void @"_COp.01d632db26744f11:core.Destroy.Core"(ptr %_.var.loc16), !dbg !35
// CHECK:STDOUT:   call void @"_COp.2867d2b01356b3cb:core.Destroy.Core"(ptr %_.var.loc15), !dbg !34
// CHECK:STDOUT:   call void @"_COp.adbfbb7b79f63468:core.Destroy.Core"(ptr %_.var.loc14), !dbg !33
// CHECK:STDOUT:   call void @"_COp.8ed16437b3e9f3a5:core.Destroy.Core"(ptr %_.var.loc13), !dbg !32
// CHECK:STDOUT:   call void @"_COp.01d632db26744f11:core.Destroy.Core"(ptr %_.var.loc12), !dbg !31
// CHECK:STDOUT:   call void @"_COp.2867d2b01356b3cb:core.Destroy.Core"(ptr %_.var.loc11), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cf0fdf6659baaa34:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ed16437b3e9f3a5:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.adbfbb7b79f63468:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2867d2b01356b3cb:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.01d632db26744f11:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.36b54e287793a2ba(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %self), !dbg !65
// CHECK:STDOUT:   ret i1 %1, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.36b54e287793a2ba(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret ptr %1, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.d356e073ce1a5b35"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !72 {
// CHECK:STDOUT:   call void @"_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.cf11014f74122042"(ptr %return, i32 %self), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.cf11014f74122042(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !79 {
// CHECK:STDOUT:   call void @"_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr %return), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.93e3eaa0d3eea476"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !84 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.d356e073ce1a5b35"(ptr %return, i32 %self), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.a06e83c323a306da"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !90 {
// CHECK:STDOUT:   call void @"_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.359fad345e4cc188"(ptr %return, i32 %self), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.183173080affdcb8"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !95 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.a06e83c323a306da"(ptr %return, i32 %self), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.93e3eaa0d3eea476"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !100 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.d356e073ce1a5b35"(ptr %return, i32 %self), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.54bcc98de4b4d68e"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !105 {
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.93e3eaa0d3eea476"(ptr %return, i32 %self), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.183173080affdcb8"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !110 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.a06e83c323a306da"(ptr %return, i32 %self), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.53171a0379a201a0"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !115 {
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.183173080affdcb8"(ptr %return, i32 %self), !dbg !118
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %value) #0 !dbg !120 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !123
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !124
// CHECK:STDOUT:   ret i1 %2, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %value) #0 !dbg !126 {
// CHECK:STDOUT:   ret ptr %value, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.cf11014f74122042"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !130 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.cf11014f74122042(ptr %return, i32 %self), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !135 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !136
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !136
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.359fad345e4cc188"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !138 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !141
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !141
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a"(i32 %self), !dbg !141
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !141
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !141
// CHECK:STDOUT:   call void @_CSome.Optional.Core.50482747a823e073(ptr %return, i32 %2), !dbg !142
// CHECK:STDOUT:   call void @"_COp.cf0fdf6659baaa34:core.Destroy.Core"(ptr %temp), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.cf11014f74122042(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !144 {
// CHECK:STDOUT:   call void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr %return, i32 %value), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a"(i32 %self) #0 !dbg !149 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401"(i32 %self), !dbg !152
// CHECK:STDOUT:   ret i32 %1, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.50482747a823e073(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !154 {
// CHECK:STDOUT:   call void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d"(ptr %return, i32 %value), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !159 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !162
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !162
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !163
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !163
// CHECK:STDOUT:   ret void, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401"(i32 %self) #0 !dbg !165 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !171
// CHECK:STDOUT:   ret i32 %1, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !173 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !176
// CHECK:STDOUT:   %1 = call i32 @"_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b"(i32 %self), !dbg !177
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !176
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !178
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !178
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !180 {
// CHECK:STDOUT:   ret i32 %self, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b"(i32 %self) #0 !dbg !184 {
// CHECK:STDOUT:   ret i32 %self, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.8ed16437b3e9f3a5:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.adbfbb7b79f63468:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.2867d2b01356b3cb:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.01d632db26744f11:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.d356e073ce1a5b35", { 0, 1, 3, 2 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
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
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cf0fdf6659baaa34:core.Destroy.Core", scope: null, file: !3, line: 13, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{null, !14}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !39, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 13, column: 10, scope: !39)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ed16437b3e9f3a5:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !14)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 3, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.adbfbb7b79f63468:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !14)
// CHECK:STDOUT: !52 = !DILocation(line: 18, column: 3, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2867d2b01356b3cb:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !14)
// CHECK:STDOUT: !56 = !DILocation(line: 16, column: 3, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.01d632db26744f11:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !14)
// CHECK:STDOUT: !60 = !DILocation(line: 16, column: 3, scope: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.36b54e287793a2ba", scope: null, file: !62, line: 36, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !63)
// CHECK:STDOUT: !62 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !61, type: !14)
// CHECK:STDOUT: !65 = !DILocation(line: 37, column: 12, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 37, column: 5, scope: !61)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.36b54e287793a2ba", scope: null, file: !62, line: 39, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !14)
// CHECK:STDOUT: !70 = !DILocation(line: 40, column: 12, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 40, column: 5, scope: !67)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.d356e073ce1a5b35", scope: null, file: !62, line: 105, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DISubroutineType(types: !74)
// CHECK:STDOUT: !74 = !{!14, !7}
// CHECK:STDOUT: !75 = !{!76}
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !72, type: !7)
// CHECK:STDOUT: !77 = !DILocation(line: 106, column: 12, scope: !72)
// CHECK:STDOUT: !78 = !DILocation(line: 106, column: 5, scope: !72)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.cf11014f74122042", scope: null, file: !62, line: 30, type: !80, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !80 = !DISubroutineType(types: !81)
// CHECK:STDOUT: !81 = !{!14}
// CHECK:STDOUT: !82 = !DILocation(line: 31, column: 12, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 31, column: 5, scope: !79)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.93e3eaa0d3eea476", scope: null, file: !85, line: 40, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !86)
// CHECK:STDOUT: !85 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !86 = !{!87}
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !84, type: !7)
// CHECK:STDOUT: !88 = !DILocation(line: 40, column: 45, scope: !84)
// CHECK:STDOUT: !89 = !DILocation(line: 40, column: 38, scope: !84)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.a06e83c323a306da", scope: null, file: !62, line: 105, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !91)
// CHECK:STDOUT: !91 = !{!92}
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !90, type: !7)
// CHECK:STDOUT: !93 = !DILocation(line: 106, column: 12, scope: !90)
// CHECK:STDOUT: !94 = !DILocation(line: 106, column: 5, scope: !90)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.183173080affdcb8", scope: null, file: !85, line: 40, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !96)
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !95, type: !7)
// CHECK:STDOUT: !98 = !DILocation(line: 40, column: 45, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 40, column: 38, scope: !95)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.93e3eaa0d3eea476", scope: null, file: !85, line: 44, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !7)
// CHECK:STDOUT: !103 = !DILocation(line: 44, column: 45, scope: !100)
// CHECK:STDOUT: !104 = !DILocation(line: 44, column: 38, scope: !100)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.54bcc98de4b4d68e", scope: null, file: !85, line: 44, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !106)
// CHECK:STDOUT: !106 = !{!107}
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !105, type: !7)
// CHECK:STDOUT: !108 = !DILocation(line: 44, column: 45, scope: !105)
// CHECK:STDOUT: !109 = !DILocation(line: 44, column: 38, scope: !105)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.183173080affdcb8", scope: null, file: !85, line: 44, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !111)
// CHECK:STDOUT: !111 = !{!112}
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !110, type: !7)
// CHECK:STDOUT: !113 = !DILocation(line: 44, column: 45, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 44, column: 38, scope: !110)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.53171a0379a201a0", scope: null, file: !85, line: 44, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !116)
// CHECK:STDOUT: !116 = !{!117}
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !115, type: !7)
// CHECK:STDOUT: !118 = !DILocation(line: 44, column: 45, scope: !115)
// CHECK:STDOUT: !119 = !DILocation(line: 44, column: 38, scope: !115)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70", scope: null, file: !62, line: 172, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !121)
// CHECK:STDOUT: !121 = !{!122}
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !120, type: !14)
// CHECK:STDOUT: !123 = !DILocation(line: 173, column: 16, scope: !120)
// CHECK:STDOUT: !124 = !DILocation(line: 173, column: 12, scope: !120)
// CHECK:STDOUT: !125 = !DILocation(line: 173, column: 5, scope: !120)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70", scope: null, file: !62, line: 175, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !127)
// CHECK:STDOUT: !127 = !{!128}
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !126, type: !14)
// CHECK:STDOUT: !129 = !DILocation(line: 176, column: 5, scope: !126)
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.cf11014f74122042", scope: null, file: !62, line: 80, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !131)
// CHECK:STDOUT: !131 = !{!132}
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !130, type: !7)
// CHECK:STDOUT: !133 = !DILocation(line: 81, column: 12, scope: !130)
// CHECK:STDOUT: !134 = !DILocation(line: 81, column: 5, scope: !130)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "None", linkageName: "_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6", scope: null, file: !62, line: 125, type: !80, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !136 = !DILocation(line: 128, column: 5, scope: !135)
// CHECK:STDOUT: !137 = !DILocation(line: 129, column: 5, scope: !135)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.359fad345e4cc188", scope: null, file: !62, line: 87, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !139)
// CHECK:STDOUT: !139 = !{!140}
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !138, type: !7)
// CHECK:STDOUT: !141 = !DILocation(line: 88, column: 29, scope: !138)
// CHECK:STDOUT: !142 = !DILocation(line: 88, column: 12, scope: !138)
// CHECK:STDOUT: !143 = !DILocation(line: 88, column: 5, scope: !138)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.cf11014f74122042", scope: null, file: !62, line: 33, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !145)
// CHECK:STDOUT: !145 = !{!146}
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !144, type: !7)
// CHECK:STDOUT: !147 = !DILocation(line: 34, column: 12, scope: !144)
// CHECK:STDOUT: !148 = !DILocation(line: 34, column: 5, scope: !144)
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a", scope: null, file: !85, line: 40, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !150)
// CHECK:STDOUT: !150 = !{!151}
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !149, type: !7)
// CHECK:STDOUT: !152 = !DILocation(line: 40, column: 45, scope: !149)
// CHECK:STDOUT: !153 = !DILocation(line: 40, column: 38, scope: !149)
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.50482747a823e073", scope: null, file: !62, line: 33, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !155)
// CHECK:STDOUT: !155 = !{!156}
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !154, type: !7)
// CHECK:STDOUT: !157 = !DILocation(line: 34, column: 12, scope: !154)
// CHECK:STDOUT: !158 = !DILocation(line: 34, column: 5, scope: !154)
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6", scope: null, file: !62, line: 131, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !160)
// CHECK:STDOUT: !160 = !{!161}
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !159, type: !7)
// CHECK:STDOUT: !162 = !DILocation(line: 137, column: 5, scope: !159)
// CHECK:STDOUT: !163 = !DILocation(line: 138, column: 5, scope: !159)
// CHECK:STDOUT: !164 = !DILocation(line: 139, column: 5, scope: !159)
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401", scope: null, file: !166, line: 62, type: !167, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !169)
// CHECK:STDOUT: !166 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !167 = !DISubroutineType(types: !168)
// CHECK:STDOUT: !168 = !{!7, !7}
// CHECK:STDOUT: !169 = !{!170}
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !165, type: !7)
// CHECK:STDOUT: !171 = !DILocation(line: 64, column: 17, scope: !165)
// CHECK:STDOUT: !172 = !DILocation(line: 64, column: 5, scope: !165)
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d", scope: null, file: !62, line: 131, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !174)
// CHECK:STDOUT: !174 = !{!175}
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !173, type: !7)
// CHECK:STDOUT: !176 = !DILocation(line: 137, column: 5, scope: !173)
// CHECK:STDOUT: !177 = !DILocation(line: 137, column: 28, scope: !173)
// CHECK:STDOUT: !178 = !DILocation(line: 138, column: 5, scope: !173)
// CHECK:STDOUT: !179 = !DILocation(line: 139, column: 5, scope: !173)
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !166, line: 57, type: !167, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !181)
// CHECK:STDOUT: !181 = !{!182}
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !180, type: !7)
// CHECK:STDOUT: !183 = !DILocation(line: 57, column: 36, scope: !180)
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b", scope: null, file: !185, line: 19, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !186)
// CHECK:STDOUT: !185 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !186 = !{!187}
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !184, type: !7)
// CHECK:STDOUT: !188 = !DILocation(line: 19, column: 33, scope: !184)
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fc5284d8d58d5626:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.c9e30f6f9b477949"(ptr %return, ptr %c), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.c9e30f6f9b477949"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !18 {
// CHECK:STDOUT:   call void @"_CCopy.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %return, ptr %self), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !24 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %value), !dbg !27
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   call void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %return, ptr %value1), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %return), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %value) #0 !dbg !34 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !37
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !37
// CHECK:STDOUT:   ret i1 %2, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !39 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !43
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !44
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !44
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr sret({ {}, i1 }) %return) #0 !dbg !46 {
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fc5284d8d58d5626:core.Destroy.Core", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
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
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.c9e30f6f9b477949", scope: null, file: !19, line: 50, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 51, column: 12, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 51, column: 5, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a", scope: null, file: !19, line: 147, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 148, column: 9, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 148, column: 8, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 149, column: 15, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 149, column: 14, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 149, column: 7, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 151, column: 14, scope: !24)
// CHECK:STDOUT: !33 = !DILocation(line: 151, column: 7, scope: !24)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a", scope: null, file: !19, line: 141, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 142, column: 12, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 142, column: 5, scope: !34)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a", scope: null, file: !19, line: 131, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 137, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 137, column: 28, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 138, column: 5, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 139, column: 5, scope: !39)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "None", linkageName: "_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a", scope: null, file: !19, line: 125, type: !47, spFlags: DISPFlagDefinition, unit: !2)
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
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.051b3610b095ceee"(ptr %c), !dbg !10
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.051b3610b095ceee"(ptr %self) #0 !dbg !12 {
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.051b3610b095ceee", scope: null, file: !13, line: 50, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !13 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 51, column: 5, scope: !12)
