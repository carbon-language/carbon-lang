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
// CHECK:STDOUT: define weak_odr void @"_COp.57bd9f6c8b0ddceb:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.e6115b4747d3a2b3(ptr %o), !dbg !28
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.e6115b4747d3a2b3(ptr %o), !dbg !30
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !31
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.1e3b2f935feb783d"(ptr %return, i32 %.loc5_12.2), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.15222d4c26531792(ptr %return), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc11 = alloca <{ i32, i1 }>, align 8, !dbg !41
// CHECK:STDOUT:   %_.var.loc12 = alloca <{ i32, i1 }>, align 8, !dbg !42
// CHECK:STDOUT:   %_.var.loc13 = alloca <{ i32, i1 }>, align 8, !dbg !43
// CHECK:STDOUT:   %_.var.loc14 = alloca <{ i32, i1 }>, align 8, !dbg !44
// CHECK:STDOUT:   %_.var.loc15 = alloca <{ i32, i1 }>, align 8, !dbg !45
// CHECK:STDOUT:   %_.var.loc16 = alloca <{ i32, i1 }>, align 8, !dbg !46
// CHECK:STDOUT:   %_.var.loc17 = alloca <{ i32, i1 }>, align 8, !dbg !47
// CHECK:STDOUT:   %_.var.loc18 = alloca <{ i32, i1 }>, align 8, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc11), !dbg !41
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.1e3b2f935feb783d"(ptr %_.var.loc11, i32 %a), !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !42
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.6a83624815bc8756"(ptr %_.var.loc12, i32 %a), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !43
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.26cce57ba1e65e96"(ptr %_.var.loc13, i32 %a), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !44
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.af766a6b51610133"(ptr %_.var.loc14, i32 %a), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !45
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.6a83624815bc8756"(ptr %_.var.loc15, i32 %b), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !46
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.13d80ab560a4ebc9"(ptr %_.var.loc16, i32 %b), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !47
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.af766a6b51610133"(ptr %_.var.loc17, i32 %b), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !48
// CHECK:STDOUT:   call void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.2654ad1e872e27ec"(ptr %_.var.loc18, i32 %b), !dbg !48
// CHECK:STDOUT:   call void @"_COp.fe89838e2e9a63da:core.Destroy.Core"(ptr %_.var.loc18), !dbg !48
// CHECK:STDOUT:   call void @"_COp.4e11a4480f8cbb96:core.Destroy.Core"(ptr %_.var.loc17), !dbg !47
// CHECK:STDOUT:   call void @"_COp.5074ca706f22983f:core.Destroy.Core"(ptr %_.var.loc16), !dbg !46
// CHECK:STDOUT:   call void @"_COp.50f61c2ea0ca4bcb:core.Destroy.Core"(ptr %_.var.loc15), !dbg !45
// CHECK:STDOUT:   call void @"_COp.fe89838e2e9a63da:core.Destroy.Core"(ptr %_.var.loc14), !dbg !44
// CHECK:STDOUT:   call void @"_COp.4e11a4480f8cbb96:core.Destroy.Core"(ptr %_.var.loc13), !dbg !43
// CHECK:STDOUT:   call void @"_COp.5074ca706f22983f:core.Destroy.Core"(ptr %_.var.loc12), !dbg !42
// CHECK:STDOUT:   call void @"_COp.50f61c2ea0ca4bcb:core.Destroy.Core"(ptr %_.var.loc11), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cf0fdf6659baaa34:core.Destroy.Core"(ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.83837874cf69a650:core.Destroy.Core"(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4e11a4480f8cbb96:core.Destroy.Core"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fe89838e2e9a63da:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.50f61c2ea0ca4bcb:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5074ca706f22983f:core.Destroy.Core"(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.e6115b4747d3a2b3(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %self), !dbg !77
// CHECK:STDOUT:   ret i1 %1, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.e6115b4747d3a2b3(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %self), !dbg !82
// CHECK:STDOUT:   ret ptr %1, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.1e3b2f935feb783d"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !84 {
// CHECK:STDOUT:   call void @"_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.15222d4c26531792"(ptr %return, i32 %self), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.15222d4c26531792(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !91 {
// CHECK:STDOUT:   call void @"_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr %return), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.6a83624815bc8756"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !96 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.1e3b2f935feb783d"(ptr %return, i32 %self), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.26cce57ba1e65e96"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !102 {
// CHECK:STDOUT:   call void @"_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.7d3466c9052fbbc7"(ptr %return, i32 %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.af766a6b51610133"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !107 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.26cce57ba1e65e96"(ptr %return, i32 %self), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.6a83624815bc8756"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !112 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.1e3b2f935feb783d"(ptr %return, i32 %self), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.13d80ab560a4ebc9"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !117 {
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.6a83624815bc8756"(ptr %return, i32 %self), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.af766a6b51610133"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !122 {
// CHECK:STDOUT:   call void @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.26cce57ba1e65e96"(ptr %return, i32 %self), !dbg !125
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.2654ad1e872e27ec"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !127 {
// CHECK:STDOUT:   call void @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.af766a6b51610133"(ptr %return, i32 %self), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %value) #0 !dbg !132 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !135
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !136
// CHECK:STDOUT:   ret i1 %2, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70"(ptr %value) #0 !dbg !138 {
// CHECK:STDOUT:   ret ptr %value, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.15222d4c26531792"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !142 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.15222d4c26531792(ptr %return, i32 %self), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !147 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !148
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !148
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.7d3466c9052fbbc7"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !150 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !153
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !153
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a"(i32 %self), !dbg !153
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !153
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !153
// CHECK:STDOUT:   call void @_CSome.Optional.Core.2930fc949346f0ad(ptr %return, i32 %2), !dbg !154
// CHECK:STDOUT:   call void @"_COp.cf0fdf6659baaa34:core.Destroy.Core"(ptr %temp), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.15222d4c26531792(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !156 {
// CHECK:STDOUT:   call void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr %return, i32 %value), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a"(i32 %self) #0 !dbg !161 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401"(i32 %self), !dbg !164
// CHECK:STDOUT:   ret i32 %1, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.2930fc949346f0ad(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !166 {
// CHECK:STDOUT:   call void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d"(ptr %return, i32 %value), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !171 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !174
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !174
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !175
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !175
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401"(i32 %self) #0 !dbg !177 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !183
// CHECK:STDOUT:   ret i32 %1, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !185 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !188
// CHECK:STDOUT:   %1 = call i32 @"_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b"(i32 %self), !dbg !189
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !188
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !190
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !190
// CHECK:STDOUT:   ret void, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !192 {
// CHECK:STDOUT:   ret i32 %self, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b"(i32 %self) #0 !dbg !196 {
// CHECK:STDOUT:   ret i32 %self, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.4e11a4480f8cbb96:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.fe89838e2e9a63da:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.50f61c2ea0ca4bcb:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.5074ca706f22983f:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.1e3b2f935feb783d", { 0, 1, 3, 2 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3fe7740c8ce31fb1:core.Destroy.Core", scope: null, file: !5, line: 164, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !6 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !7 = !{null, !8}
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 164, column: 19, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 3, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 3, column: 39, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.57bd9f6c8b0ddceb:core.Destroy.Core", scope: null, file: !5, line: 124, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !8)
// CHECK:STDOUT: !22 = !DILocation(line: 124, column: 19, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !3, line: 3, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !25 = !{!8, !8}
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !23, type: !8)
// CHECK:STDOUT: !28 = !DILocation(line: 4, column: 7, scope: !23)
// CHECK:STDOUT: !29 = !DILocation(line: 4, column: 6, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 5, column: 13, scope: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 5, column: 12, scope: !23)
// CHECK:STDOUT: !32 = !DILocation(line: 5, column: 5, scope: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 7, column: 10, scope: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 7, column: 3, scope: !23)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !3, line: 10, type: !36, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !37 = !{null, !15, !15}
// CHECK:STDOUT: !38 = !{!39, !40}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !35, type: !15)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 2, scope: !35, type: !15)
// CHECK:STDOUT: !41 = !DILocation(line: 11, column: 3, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 12, column: 3, scope: !35)
// CHECK:STDOUT: !43 = !DILocation(line: 13, column: 3, scope: !35)
// CHECK:STDOUT: !44 = !DILocation(line: 14, column: 3, scope: !35)
// CHECK:STDOUT: !45 = !DILocation(line: 15, column: 3, scope: !35)
// CHECK:STDOUT: !46 = !DILocation(line: 16, column: 3, scope: !35)
// CHECK:STDOUT: !47 = !DILocation(line: 17, column: 3, scope: !35)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 3, scope: !35)
// CHECK:STDOUT: !49 = !DILocation(line: 10, column: 1, scope: !35)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cf0fdf6659baaa34:core.Destroy.Core", scope: null, file: !3, line: 13, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !51)
// CHECK:STDOUT: !51 = !{!52}
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !50, type: !8)
// CHECK:STDOUT: !53 = !DILocation(line: 13, column: 10, scope: !50)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp.83837874cf69a650:core.Destroy.Core", scope: null, file: !5, line: 124, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !55)
// CHECK:STDOUT: !55 = !{!56}
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !54, type: !8)
// CHECK:STDOUT: !57 = !DILocation(line: 124, column: 19, scope: !54)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4e11a4480f8cbb96:core.Destroy.Core", scope: null, file: !3, line: 18, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !59)
// CHECK:STDOUT: !59 = !{!60}
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !58, type: !8)
// CHECK:STDOUT: !61 = !DILocation(line: 18, column: 3, scope: !58)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fe89838e2e9a63da:core.Destroy.Core", scope: null, file: !3, line: 18, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !8)
// CHECK:STDOUT: !65 = !DILocation(line: 18, column: 3, scope: !62)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Op", linkageName: "_COp.50f61c2ea0ca4bcb:core.Destroy.Core", scope: null, file: !3, line: 16, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !67)
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !66, type: !8)
// CHECK:STDOUT: !69 = !DILocation(line: 16, column: 3, scope: !66)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5074ca706f22983f:core.Destroy.Core", scope: null, file: !3, line: 16, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !71)
// CHECK:STDOUT: !71 = !{!72}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !70, type: !8)
// CHECK:STDOUT: !73 = !DILocation(line: 16, column: 3, scope: !70)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.e6115b4747d3a2b3", scope: null, file: !5, line: 36, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !75)
// CHECK:STDOUT: !75 = !{!76}
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !74, type: !8)
// CHECK:STDOUT: !77 = !DILocation(line: 37, column: 12, scope: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 37, column: 5, scope: !74)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.e6115b4747d3a2b3", scope: null, file: !5, line: 39, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !80)
// CHECK:STDOUT: !80 = !{!81}
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !79, type: !8)
// CHECK:STDOUT: !82 = !DILocation(line: 40, column: 12, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 40, column: 5, scope: !79)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.1e3b2f935feb783d", scope: null, file: !5, line: 105, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !87)
// CHECK:STDOUT: !85 = !DISubroutineType(types: !86)
// CHECK:STDOUT: !86 = !{!8, !15}
// CHECK:STDOUT: !87 = !{!88}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !84, type: !15)
// CHECK:STDOUT: !89 = !DILocation(line: 106, column: 12, scope: !84)
// CHECK:STDOUT: !90 = !DILocation(line: 106, column: 5, scope: !84)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.15222d4c26531792", scope: null, file: !5, line: 30, type: !92, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !92 = !DISubroutineType(types: !93)
// CHECK:STDOUT: !93 = !{!8}
// CHECK:STDOUT: !94 = !DILocation(line: 31, column: 12, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 31, column: 5, scope: !91)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.6a83624815bc8756", scope: null, file: !97, line: 40, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !98)
// CHECK:STDOUT: !97 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !98 = !{!99}
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !96, type: !15)
// CHECK:STDOUT: !100 = !DILocation(line: 40, column: 45, scope: !96)
// CHECK:STDOUT: !101 = !DILocation(line: 40, column: 38, scope: !96)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.26cce57ba1e65e96", scope: null, file: !5, line: 105, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !103)
// CHECK:STDOUT: !103 = !{!104}
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !102, type: !15)
// CHECK:STDOUT: !105 = !DILocation(line: 106, column: 12, scope: !102)
// CHECK:STDOUT: !106 = !DILocation(line: 106, column: 5, scope: !102)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.af766a6b51610133", scope: null, file: !97, line: 40, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !108)
// CHECK:STDOUT: !108 = !{!109}
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !107, type: !15)
// CHECK:STDOUT: !110 = !DILocation(line: 40, column: 45, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 40, column: 38, scope: !107)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.6a83624815bc8756", scope: null, file: !97, line: 44, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !113)
// CHECK:STDOUT: !113 = !{!114}
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !112, type: !15)
// CHECK:STDOUT: !115 = !DILocation(line: 44, column: 45, scope: !112)
// CHECK:STDOUT: !116 = !DILocation(line: 44, column: 38, scope: !112)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.13d80ab560a4ebc9", scope: null, file: !97, line: 44, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !118)
// CHECK:STDOUT: !118 = !{!119}
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !117, type: !15)
// CHECK:STDOUT: !120 = !DILocation(line: 44, column: 45, scope: !117)
// CHECK:STDOUT: !121 = !DILocation(line: 44, column: 38, scope: !117)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.af766a6b51610133", scope: null, file: !97, line: 44, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !123)
// CHECK:STDOUT: !123 = !{!124}
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !122, type: !15)
// CHECK:STDOUT: !125 = !DILocation(line: 44, column: 45, scope: !122)
// CHECK:STDOUT: !126 = !DILocation(line: 44, column: 38, scope: !122)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e767a09bf0821f52:ImplicitAs.eb057aa32837c84e.Core.2654ad1e872e27ec", scope: null, file: !97, line: 44, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !128)
// CHECK:STDOUT: !128 = !{!129}
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !127, type: !15)
// CHECK:STDOUT: !130 = !DILocation(line: 44, column: 45, scope: !127)
// CHECK:STDOUT: !131 = !DILocation(line: 44, column: 38, scope: !127)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70", scope: null, file: !5, line: 172, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !133)
// CHECK:STDOUT: !133 = !{!134}
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !132, type: !8)
// CHECK:STDOUT: !135 = !DILocation(line: 173, column: 16, scope: !132)
// CHECK:STDOUT: !136 = !DILocation(line: 173, column: 12, scope: !132)
// CHECK:STDOUT: !137 = !DILocation(line: 173, column: 5, scope: !132)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.c8fd090a2e741f70", scope: null, file: !5, line: 175, type: !24, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !139)
// CHECK:STDOUT: !139 = !{!140}
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !138, type: !8)
// CHECK:STDOUT: !141 = !DILocation(line: 176, column: 5, scope: !138)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.15222d4c26531792", scope: null, file: !5, line: 80, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !143)
// CHECK:STDOUT: !143 = !{!144}
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !142, type: !15)
// CHECK:STDOUT: !145 = !DILocation(line: 81, column: 12, scope: !142)
// CHECK:STDOUT: !146 = !DILocation(line: 81, column: 5, scope: !142)
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "None", linkageName: "_CNone.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6", scope: null, file: !5, line: 125, type: !92, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !148 = !DILocation(line: 128, column: 5, scope: !147)
// CHECK:STDOUT: !149 = !DILocation(line: 129, column: 5, scope: !147)
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.62d55c5523334675:OptionalAs.f5e7d4305ac1f327.Core.7d3466c9052fbbc7", scope: null, file: !5, line: 87, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !151)
// CHECK:STDOUT: !151 = !{!152}
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !150, type: !15)
// CHECK:STDOUT: !153 = !DILocation(line: 88, column: 29, scope: !150)
// CHECK:STDOUT: !154 = !DILocation(line: 88, column: 12, scope: !150)
// CHECK:STDOUT: !155 = !DILocation(line: 88, column: 5, scope: !150)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.15222d4c26531792", scope: null, file: !5, line: 33, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !157)
// CHECK:STDOUT: !157 = !{!158}
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !156, type: !15)
// CHECK:STDOUT: !159 = !DILocation(line: 34, column: 12, scope: !156)
// CHECK:STDOUT: !160 = !DILocation(line: 34, column: 5, scope: !156)
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.68cbef1dbba6ee88:ImplicitAs.ad22d1bbc0605210.Core.3d03a93e76942d8a", scope: null, file: !97, line: 40, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !162)
// CHECK:STDOUT: !162 = !{!163}
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !161, type: !15)
// CHECK:STDOUT: !164 = !DILocation(line: 40, column: 45, scope: !161)
// CHECK:STDOUT: !165 = !DILocation(line: 40, column: 38, scope: !161)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.2930fc949346f0ad", scope: null, file: !5, line: 33, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !167)
// CHECK:STDOUT: !167 = !{!168}
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !166, type: !15)
// CHECK:STDOUT: !169 = !DILocation(line: 34, column: 12, scope: !166)
// CHECK:STDOUT: !170 = !DILocation(line: 34, column: 5, scope: !166)
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.4f9853132d866dc6:OptionalStorage.Core.fe1b3f265f915ed6", scope: null, file: !5, line: 131, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !172)
// CHECK:STDOUT: !172 = !{!173}
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !171, type: !15)
// CHECK:STDOUT: !174 = !DILocation(line: 137, column: 5, scope: !171)
// CHECK:STDOUT: !175 = !DILocation(line: 138, column: 5, scope: !171)
// CHECK:STDOUT: !176 = !DILocation(line: 139, column: 5, scope: !171)
// CHECK:STDOUT: !177 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.b912251d28f69b6d:ImplicitAs.a629ad9d3aac3895.Core.245bb80fde7c8401", scope: null, file: !178, line: 62, type: !179, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !181)
// CHECK:STDOUT: !178 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !179 = !DISubroutineType(types: !180)
// CHECK:STDOUT: !180 = !{!15, !15}
// CHECK:STDOUT: !181 = !{!182}
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !177, type: !15)
// CHECK:STDOUT: !183 = !DILocation(line: 64, column: 17, scope: !177)
// CHECK:STDOUT: !184 = !DILocation(line: 64, column: 5, scope: !177)
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.4f9853132d866dc6:OptionalStorage.Core.9078267c53e6931d", scope: null, file: !5, line: 131, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !186)
// CHECK:STDOUT: !186 = !{!187}
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !185, type: !15)
// CHECK:STDOUT: !188 = !DILocation(line: 137, column: 5, scope: !185)
// CHECK:STDOUT: !189 = !DILocation(line: 137, column: 28, scope: !185)
// CHECK:STDOUT: !190 = !DILocation(line: 138, column: 5, scope: !185)
// CHECK:STDOUT: !191 = !DILocation(line: 139, column: 5, scope: !185)
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !178, line: 57, type: !179, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !193)
// CHECK:STDOUT: !193 = !{!194}
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !192, type: !15)
// CHECK:STDOUT: !195 = !DILocation(line: 57, column: 36, scope: !192)
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c4bf98205baa6bbe:Copy.Core.986e4fbf737c841b", scope: null, file: !197, line: 19, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !198)
// CHECK:STDOUT: !197 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !198 = !{!199}
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !196, type: !15)
// CHECK:STDOUT: !200 = !DILocation(line: 19, column: 33, scope: !196)
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
// CHECK:STDOUT: define weak_odr void @"_COp.b912baee4d396aae:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.59f9d1ab6e717102"(ptr %return, ptr %c), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.59f9d1ab6e717102"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !23 {
// CHECK:STDOUT:   call void @"_CCopy.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %return, ptr %self), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !28 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %value), !dbg !31
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   call void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %return, ptr %value1), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %return), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr %value) #0 !dbg !38 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !41
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !41
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !41
// CHECK:STDOUT:   ret i1 %2, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !43 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !47
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a"(ptr sret({ {}, i1 }) %return) #0 !dbg !50 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !53
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
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
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b912baee4d396aae:core.Destroy.Core", scope: null, file: !12, line: 124, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !13)
// CHECK:STDOUT: !12 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 124, column: 19, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 9, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{!7, !7}
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !16, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 11, column: 10, scope: !16)
// CHECK:STDOUT: !22 = !DILocation(line: 11, column: 3, scope: !16)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.59f9d1ab6e717102", scope: null, file: !12, line: 50, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 51, column: 12, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 51, column: 5, scope: !23)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a", scope: null, file: !12, line: 147, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !31 = !DILocation(line: 148, column: 9, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 148, column: 8, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 149, column: 15, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 149, column: 14, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 149, column: 7, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 151, column: 14, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 151, column: 7, scope: !28)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a", scope: null, file: !12, line: 141, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 142, column: 12, scope: !38)
// CHECK:STDOUT: !42 = !DILocation(line: 142, column: 5, scope: !38)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a", scope: null, file: !12, line: 131, type: !17, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !44)
// CHECK:STDOUT: !44 = !{!45}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !46 = !DILocation(line: 137, column: 5, scope: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 137, column: 28, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 138, column: 5, scope: !43)
// CHECK:STDOUT: !49 = !DILocation(line: 139, column: 5, scope: !43)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "None", linkageName: "_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.8e4d068dce7d104a", scope: null, file: !12, line: 125, type: !51, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !51 = !DISubroutineType(types: !52)
// CHECK:STDOUT: !52 = !{!7}
// CHECK:STDOUT: !53 = !DILocation(line: 128, column: 5, scope: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 129, column: 5, scope: !50)
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.660495384349c676:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.95419e75852273e5"(ptr %c), !dbg !17
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.95419e75852273e5"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT:   ret ptr %self, !dbg !22
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.660495384349c676:core.Destroy.Core", scope: null, file: !5, line: 164, type: !6, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !6 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !7 = !{null, !8}
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 164, column: 19, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!8, !8}
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !12, type: !8)
// CHECK:STDOUT: !17 = !DILocation(line: 6, column: 10, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 6, column: 3, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.7eef99cd03a74de4.Core:Copy.Core.95419e75852273e5", scope: null, file: !5, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !8)
// CHECK:STDOUT: !22 = !DILocation(line: 51, column: 5, scope: !19)
