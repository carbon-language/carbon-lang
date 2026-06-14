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

// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.b79ac2ab1b7701d6(ptr %o), !dbg !17
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.b79ac2ab1b7701d6(ptr %o), !dbg !19
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.a77ff6e8edaf9e8d"(ptr %return, i32 %.loc5_12.2), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.70a0b54a75854b7a(ptr %return), !dbg !22
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
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.a77ff6e8edaf9e8d"(ptr %_.var.loc11, i32 %a), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !31
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.36d191ed69f42abb"(ptr %_.var.loc12, i32 %a), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !32
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.6f97602f44c068f9"(ptr %_.var.loc13, i32 %a), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !33
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7e715d26c6bb9daa"(ptr %_.var.loc14, i32 %a), !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !34
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.36d191ed69f42abb"(ptr %_.var.loc15, i32 %b), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !35
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bd2b2818aef220bf"(ptr %_.var.loc16, i32 %b), !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !36
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7e715d26c6bb9daa"(ptr %_.var.loc17, i32 %b), !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !37
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7f331ac9d2b85055"(ptr %_.var.loc18, i32 %b), !dbg !37
// CHECK:STDOUT:   call void @"_COp.56bc00fa6756ea9e:core.Destroy.Core"(ptr %_.var.loc18), !dbg !37
// CHECK:STDOUT:   call void @"_COp.bc7b205814bb8542:core.Destroy.Core"(ptr %_.var.loc17), !dbg !36
// CHECK:STDOUT:   call void @"_COp.49575b3ccc332429:core.Destroy.Core"(ptr %_.var.loc16), !dbg !35
// CHECK:STDOUT:   call void @"_COp.30688d383ae72bab:core.Destroy.Core"(ptr %_.var.loc15), !dbg !34
// CHECK:STDOUT:   call void @"_COp.56bc00fa6756ea9e:core.Destroy.Core"(ptr %_.var.loc14), !dbg !33
// CHECK:STDOUT:   call void @"_COp.bc7b205814bb8542:core.Destroy.Core"(ptr %_.var.loc13), !dbg !32
// CHECK:STDOUT:   call void @"_COp.49575b3ccc332429:core.Destroy.Core"(ptr %_.var.loc12), !dbg !31
// CHECK:STDOUT:   call void @"_COp.30688d383ae72bab:core.Destroy.Core"(ptr %_.var.loc11), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.05c48b3d4126597e:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d4fdfc2912de527a:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.eb54d7735089d035:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.33aade76ef369ae0:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4352d4717a4da3c5:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bc7b205814bb8542:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.56bc00fa6756ea9e:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94ba1c30d421702e:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0fc9634857315db2:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.69700c1fc1b2744c:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.36bd54c278c044c5:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.30688d383ae72bab:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.49575b3ccc332429:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.b79ac2ab1b7701d6(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !97
// CHECK:STDOUT:   ret i1 %1, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.b79ac2ab1b7701d6(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %self), !dbg !102
// CHECK:STDOUT:   ret ptr %1, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.a77ff6e8edaf9e8d"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !104 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.70a0b54a75854b7a"(ptr %return, i32 %self), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.70a0b54a75854b7a(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !111 {
// CHECK:STDOUT:   call void @"_CNone.74becf39533834ee:OptionalStorage.Core.b99f4a084b13a83f"(ptr %return), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.36d191ed69f42abb"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !116 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.a77ff6e8edaf9e8d"(ptr %return, i32 %self), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.6f97602f44c068f9"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !122 {
// CHECK:STDOUT:   call void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.04776faa19f4241b"(ptr %return, i32 %self), !dbg !125
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7e715d26c6bb9daa"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !127 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.6f97602f44c068f9"(ptr %return, i32 %self), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.36d191ed69f42abb"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !132 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.a77ff6e8edaf9e8d"(ptr %return, i32 %self), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bd2b2818aef220bf"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !137 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.36d191ed69f42abb"(ptr %return, i32 %self), !dbg !140
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7e715d26c6bb9daa"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !142 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.6f97602f44c068f9"(ptr %return, i32 %self), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7f331ac9d2b85055"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !147 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7e715d26c6bb9daa"(ptr %return, i32 %self), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !152 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !155
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !156
// CHECK:STDOUT:   ret i1 %2, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba"(ptr %value) #0 !dbg !158 {
// CHECK:STDOUT:   ret ptr %value, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.70a0b54a75854b7a"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !162 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.70a0b54a75854b7a(ptr %return, i32 %self), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.74becf39533834ee:OptionalStorage.Core.b99f4a084b13a83f"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !167 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !168
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !168
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.04776faa19f4241b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !170 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !173
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !173
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.a1e92423fb5cda0f"(i32 %self), !dbg !173
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !173
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !173
// CHECK:STDOUT:   call void @_CSome.Optional.Core.442ff502447e2586(ptr %return, i32 %2), !dbg !174
// CHECK:STDOUT:   call void @"_COp.05c48b3d4126597e:core.Destroy.Core"(ptr %temp), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.70a0b54a75854b7a(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !176 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.b99f4a084b13a83f"(ptr %return, i32 %value), !dbg !179
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.a1e92423fb5cda0f"(i32 %self) #0 !dbg !181 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47"(i32 %self), !dbg !184
// CHECK:STDOUT:   ret i32 %1, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.442ff502447e2586(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !186 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.9e376a14fbf8fb3d"(ptr %return, i32 %value), !dbg !189
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.b99f4a084b13a83f"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !191 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !194
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !194
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !195
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !195
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47"(i32 %self) #0 !dbg !197 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !203
// CHECK:STDOUT:   ret i32 %1, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.9e376a14fbf8fb3d"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !205 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !208
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.5bd8d5767580997a"(i32 %self), !dbg !209
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !208
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !210
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !210
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !212 {
// CHECK:STDOUT:   ret i32 %self, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.5bd8d5767580997a"(i32 %self) #0 !dbg !216 {
// CHECK:STDOUT:   ret i32 %self, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.bc7b205814bb8542:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.56bc00fa6756ea9e:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.30688d383ae72bab:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.49575b3ccc332429:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.a77ff6e8edaf9e8d", { 0, 1, 3, 2 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
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
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.05c48b3d4126597e:core.Destroy.Core", scope: null, file: !3, line: 13, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{null, !14}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !39, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 13, column: 10, scope: !39)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d4fdfc2912de527a:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !14)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 3, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.eb54d7735089d035:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !14)
// CHECK:STDOUT: !52 = !DILocation(line: 18, column: 3, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.33aade76ef369ae0:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !14)
// CHECK:STDOUT: !56 = !DILocation(line: 18, column: 3, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4352d4717a4da3c5:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !14)
// CHECK:STDOUT: !60 = !DILocation(line: 18, column: 3, scope: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bc7b205814bb8542:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !14)
// CHECK:STDOUT: !64 = !DILocation(line: 18, column: 3, scope: !61)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.56bc00fa6756ea9e:core.Destroy.Core", scope: null, file: !3, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !66)
// CHECK:STDOUT: !66 = !{!67}
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !65, type: !14)
// CHECK:STDOUT: !68 = !DILocation(line: 18, column: 3, scope: !65)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94ba1c30d421702e:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !14)
// CHECK:STDOUT: !72 = !DILocation(line: 16, column: 3, scope: !69)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0fc9634857315db2:core.Destroy.Core", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !76 = !DILocation(line: 16, column: 3, scope: !73)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Op", linkageName: "_COp.69700c1fc1b2744c:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !77, type: !14)
// CHECK:STDOUT: !80 = !DILocation(line: 16, column: 3, scope: !77)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Op", linkageName: "_COp.36bd54c278c044c5:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !14)
// CHECK:STDOUT: !84 = !DILocation(line: 16, column: 3, scope: !81)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Op", linkageName: "_COp.30688d383ae72bab:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !86)
// CHECK:STDOUT: !86 = !{!87}
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !85, type: !14)
// CHECK:STDOUT: !88 = !DILocation(line: 16, column: 3, scope: !85)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Op", linkageName: "_COp.49575b3ccc332429:core.Destroy.Core", scope: null, file: !3, line: 16, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !90)
// CHECK:STDOUT: !90 = !{!91}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !89, type: !14)
// CHECK:STDOUT: !92 = !DILocation(line: 16, column: 3, scope: !89)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.b79ac2ab1b7701d6", scope: null, file: !94, line: 36, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !95)
// CHECK:STDOUT: !94 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !95 = !{!96}
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !93, type: !14)
// CHECK:STDOUT: !97 = !DILocation(line: 37, column: 12, scope: !93)
// CHECK:STDOUT: !98 = !DILocation(line: 37, column: 5, scope: !93)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.b79ac2ab1b7701d6", scope: null, file: !94, line: 39, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !100)
// CHECK:STDOUT: !100 = !{!101}
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !99, type: !14)
// CHECK:STDOUT: !102 = !DILocation(line: 40, column: 12, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 40, column: 5, scope: !99)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.a77ff6e8edaf9e8d", scope: null, file: !94, line: 105, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !107)
// CHECK:STDOUT: !105 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !106 = !{!14, !7}
// CHECK:STDOUT: !107 = !{!108}
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !104, type: !7)
// CHECK:STDOUT: !109 = !DILocation(line: 106, column: 12, scope: !104)
// CHECK:STDOUT: !110 = !DILocation(line: 106, column: 5, scope: !104)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.70a0b54a75854b7a", scope: null, file: !94, line: 30, type: !112, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !112 = !DISubroutineType(types: !113)
// CHECK:STDOUT: !113 = !{!14}
// CHECK:STDOUT: !114 = !DILocation(line: 31, column: 12, scope: !111)
// CHECK:STDOUT: !115 = !DILocation(line: 31, column: 5, scope: !111)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.36d191ed69f42abb", scope: null, file: !117, line: 43, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !118)
// CHECK:STDOUT: !117 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !118 = !{!119}
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !116, type: !7)
// CHECK:STDOUT: !120 = !DILocation(line: 43, column: 40, scope: !116)
// CHECK:STDOUT: !121 = !DILocation(line: 43, column: 33, scope: !116)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.6f97602f44c068f9", scope: null, file: !94, line: 105, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !123)
// CHECK:STDOUT: !123 = !{!124}
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !122, type: !7)
// CHECK:STDOUT: !125 = !DILocation(line: 106, column: 12, scope: !122)
// CHECK:STDOUT: !126 = !DILocation(line: 106, column: 5, scope: !122)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.7e715d26c6bb9daa", scope: null, file: !117, line: 43, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !128)
// CHECK:STDOUT: !128 = !{!129}
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !127, type: !7)
// CHECK:STDOUT: !130 = !DILocation(line: 43, column: 40, scope: !127)
// CHECK:STDOUT: !131 = !DILocation(line: 43, column: 33, scope: !127)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.36d191ed69f42abb", scope: null, file: !117, line: 47, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !133)
// CHECK:STDOUT: !133 = !{!134}
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !132, type: !7)
// CHECK:STDOUT: !135 = !DILocation(line: 47, column: 34, scope: !132)
// CHECK:STDOUT: !136 = !DILocation(line: 47, column: 27, scope: !132)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.bd2b2818aef220bf", scope: null, file: !117, line: 47, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !138)
// CHECK:STDOUT: !138 = !{!139}
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !137, type: !7)
// CHECK:STDOUT: !140 = !DILocation(line: 47, column: 34, scope: !137)
// CHECK:STDOUT: !141 = !DILocation(line: 47, column: 27, scope: !137)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7e715d26c6bb9daa", scope: null, file: !117, line: 47, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !143)
// CHECK:STDOUT: !143 = !{!144}
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !142, type: !7)
// CHECK:STDOUT: !145 = !DILocation(line: 47, column: 34, scope: !142)
// CHECK:STDOUT: !146 = !DILocation(line: 47, column: 27, scope: !142)
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.7f331ac9d2b85055", scope: null, file: !117, line: 47, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !148)
// CHECK:STDOUT: !148 = !{!149}
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !147, type: !7)
// CHECK:STDOUT: !150 = !DILocation(line: 47, column: 34, scope: !147)
// CHECK:STDOUT: !151 = !DILocation(line: 47, column: 27, scope: !147)
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !94, line: 172, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !153)
// CHECK:STDOUT: !153 = !{!154}
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !152, type: !14)
// CHECK:STDOUT: !155 = !DILocation(line: 173, column: 16, scope: !152)
// CHECK:STDOUT: !156 = !DILocation(line: 173, column: 12, scope: !152)
// CHECK:STDOUT: !157 = !DILocation(line: 173, column: 5, scope: !152)
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.84588f41d61dafba", scope: null, file: !94, line: 175, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !159)
// CHECK:STDOUT: !159 = !{!160}
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !158, type: !14)
// CHECK:STDOUT: !161 = !DILocation(line: 176, column: 5, scope: !158)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.70a0b54a75854b7a", scope: null, file: !94, line: 80, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !163)
// CHECK:STDOUT: !163 = !{!164}
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !162, type: !7)
// CHECK:STDOUT: !165 = !DILocation(line: 81, column: 12, scope: !162)
// CHECK:STDOUT: !166 = !DILocation(line: 81, column: 5, scope: !162)
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "None", linkageName: "_CNone.74becf39533834ee:OptionalStorage.Core.b99f4a084b13a83f", scope: null, file: !94, line: 125, type: !112, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !168 = !DILocation(line: 128, column: 5, scope: !167)
// CHECK:STDOUT: !169 = !DILocation(line: 129, column: 5, scope: !167)
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.04776faa19f4241b", scope: null, file: !94, line: 87, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !171)
// CHECK:STDOUT: !171 = !{!172}
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !170, type: !7)
// CHECK:STDOUT: !173 = !DILocation(line: 88, column: 29, scope: !170)
// CHECK:STDOUT: !174 = !DILocation(line: 88, column: 12, scope: !170)
// CHECK:STDOUT: !175 = !DILocation(line: 88, column: 5, scope: !170)
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.70a0b54a75854b7a", scope: null, file: !94, line: 33, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !177)
// CHECK:STDOUT: !177 = !{!178}
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 1, scope: !176, type: !7)
// CHECK:STDOUT: !179 = !DILocation(line: 34, column: 12, scope: !176)
// CHECK:STDOUT: !180 = !DILocation(line: 34, column: 5, scope: !176)
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.a1e92423fb5cda0f", scope: null, file: !117, line: 43, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !182)
// CHECK:STDOUT: !182 = !{!183}
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !181, type: !7)
// CHECK:STDOUT: !184 = !DILocation(line: 43, column: 40, scope: !181)
// CHECK:STDOUT: !185 = !DILocation(line: 43, column: 33, scope: !181)
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.442ff502447e2586", scope: null, file: !94, line: 33, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !187)
// CHECK:STDOUT: !187 = !{!188}
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 1, scope: !186, type: !7)
// CHECK:STDOUT: !189 = !DILocation(line: 34, column: 12, scope: !186)
// CHECK:STDOUT: !190 = !DILocation(line: 34, column: 5, scope: !186)
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.b99f4a084b13a83f", scope: null, file: !94, line: 131, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !192)
// CHECK:STDOUT: !192 = !{!193}
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !191, type: !7)
// CHECK:STDOUT: !194 = !DILocation(line: 137, column: 5, scope: !191)
// CHECK:STDOUT: !195 = !DILocation(line: 138, column: 5, scope: !191)
// CHECK:STDOUT: !196 = !DILocation(line: 139, column: 5, scope: !191)
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47", scope: null, file: !198, line: 64, type: !199, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !201)
// CHECK:STDOUT: !198 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !199 = !DISubroutineType(types: !200)
// CHECK:STDOUT: !200 = !{!7, !7}
// CHECK:STDOUT: !201 = !{!202}
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !197, type: !7)
// CHECK:STDOUT: !203 = !DILocation(line: 66, column: 17, scope: !197)
// CHECK:STDOUT: !204 = !DILocation(line: 66, column: 5, scope: !197)
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.9e376a14fbf8fb3d", scope: null, file: !94, line: 131, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !206)
// CHECK:STDOUT: !206 = !{!207}
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !205, type: !7)
// CHECK:STDOUT: !208 = !DILocation(line: 137, column: 5, scope: !205)
// CHECK:STDOUT: !209 = !DILocation(line: 137, column: 28, scope: !205)
// CHECK:STDOUT: !210 = !DILocation(line: 138, column: 5, scope: !205)
// CHECK:STDOUT: !211 = !DILocation(line: 139, column: 5, scope: !205)
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !198, line: 59, type: !199, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !213)
// CHECK:STDOUT: !213 = !{!214}
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !212, type: !7)
// CHECK:STDOUT: !215 = !DILocation(line: 59, column: 28, scope: !212)
// CHECK:STDOUT: !216 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.5bd8d5767580997a", scope: null, file: !217, line: 23, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !218)
// CHECK:STDOUT: !217 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !218 = !{!219}
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !216, type: !7)
// CHECK:STDOUT: !220 = !DILocation(line: 23, column: 25, scope: !216)
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e8cfefb1bcff6750:core.Destroy.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.da424a1efa3a0e0f"(ptr %return, ptr %c), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.da424a1efa3a0e0f"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !18 {
// CHECK:STDOUT:   call void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6"(ptr %return, ptr %self), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !24 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6"(ptr %value), !dbg !27
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6"(ptr %return, ptr %value1), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6"(ptr %return), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6"(ptr %value) #0 !dbg !34 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !37
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !37
// CHECK:STDOUT:   ret i1 %2, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !39 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !43
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !44
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !44
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6"(ptr sret({ {}, i1 }) %return) #0 !dbg !46 {
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e8cfefb1bcff6750:core.Destroy.Core", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
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
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.da424a1efa3a0e0f", scope: null, file: !19, line: 50, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !19 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 51, column: 12, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 51, column: 5, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6", scope: null, file: !19, line: 147, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 148, column: 9, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 148, column: 8, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 149, column: 15, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 149, column: 14, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 149, column: 7, scope: !24)
// CHECK:STDOUT: !32 = !DILocation(line: 151, column: 14, scope: !24)
// CHECK:STDOUT: !33 = !DILocation(line: 151, column: 7, scope: !24)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6", scope: null, file: !19, line: 141, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 142, column: 12, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 142, column: 5, scope: !34)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6", scope: null, file: !19, line: 131, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 137, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 137, column: 28, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 138, column: 5, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 139, column: 5, scope: !39)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.49fa0e4cc21a3ef6", scope: null, file: !19, line: 125, type: !47, spFlags: DISPFlagDefinition, unit: !2)
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
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.4791ff83c858ed65"(ptr %c), !dbg !10
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.4791ff83c858ed65"(ptr %self) #0 !dbg !12 {
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.4791ff83c858ed65", scope: null, file: !13, line: 50, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !13 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 51, column: 5, scope: !12)
