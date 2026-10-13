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
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.49ebf9b244db2604:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d19d32d3b3a297a1:core.Destroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret(<{ i32, i1 }>) %return, ptr %o) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.0176148a5c50a958(ptr %o), !dbg !30
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.0176148a5c50a958(ptr %o), !dbg !32
// CHECK:STDOUT:   %.loc5_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !33
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.423003c0ed122739"(ptr %return, i32 %.loc5_12.2), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.1c5c1cd4e7e6d532(ptr %return), !dbg !37
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
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.423003c0ed122739"(ptr %_.var.loc11, i32 %a), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc12), !dbg !50
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.2fca651b624aec3b"(ptr %_.var.loc12, i32 %a), !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !53
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.910e80862b30ac54"(ptr %_.var.loc13, i32 %a), !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !56
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.f0eed05b796df3bf"(ptr %_.var.loc14, i32 %a), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !59
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.2fca651b624aec3b"(ptr %_.var.loc15, i32 %b), !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !62
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.db742163b958e819"(ptr %_.var.loc16, i32 %b), !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !65
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.f0eed05b796df3bf"(ptr %_.var.loc17, i32 %b), !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !68
// CHECK:STDOUT:   call void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.5fc0811aa20c2f57"(ptr %_.var.loc18, i32 %b), !dbg !68
// CHECK:STDOUT:   call void @"_CSelfDestruct.b7b274de082dcb96:core.Destroy.Core"(ptr %_.var.loc18), !dbg !68
// CHECK:STDOUT:   call void @"_CSelfDestruct.997b73c74c4e179f:core.Destroy.Core"(ptr %_.var.loc17), !dbg !65
// CHECK:STDOUT:   call void @"_CSelfDestruct.f93e486abe6b5cfe:core.Destroy.Core"(ptr %_.var.loc16), !dbg !62
// CHECK:STDOUT:   call void @"_CSelfDestruct.3d12b9ad3598e418:core.Destroy.Core"(ptr %_.var.loc15), !dbg !59
// CHECK:STDOUT:   call void @"_CSelfDestruct.b7b274de082dcb96:core.Destroy.Core"(ptr %_.var.loc14), !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.997b73c74c4e179f:core.Destroy.Core"(ptr %_.var.loc13), !dbg !53
// CHECK:STDOUT:   call void @"_CSelfDestruct.f93e486abe6b5cfe:core.Destroy.Core"(ptr %_.var.loc12), !dbg !50
// CHECK:STDOUT:   call void @"_CSelfDestruct.3d12b9ad3598e418:core.Destroy.Core"(ptr %_.var.loc11), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ed3b5dcc9f2a0f16:core.Destroy.Core"(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.ed3b5dcc9f2a0f16:core.Destroy.Core"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.ed3b5dcc9f2a0f16:core.Destroy.Core"(ptr %self), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4f8c0853b2a8342d:core.Destroy.Core"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4f8c0853b2a8342d:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4f8c0853b2a8342d:core.Destroy.Core"(ptr %self), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fd06fe99ef8d5695:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2c621ceeef95ce46:core.Destroy.Core"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !99
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc18_3.1.has_value), !dbg !99
// CHECK:STDOUT:   %.loc18_3.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !99
// CHECK:STDOUT:   call void @"_CSelfDestruct.fd06fe99ef8d5695:core.Destroy.Core"(ptr %.loc18_3.2.value), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2c621ceeef95ce46:core.Destroy.Core"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2c621ceeef95ce46:core.Destroy.Core"(ptr %self), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f3a741ef694e0907:core.Destroy.Core"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !107
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc18_3.1.has_value), !dbg !107
// CHECK:STDOUT:   %.loc18_3.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !107
// CHECK:STDOUT:   call void @"_CSelfDestruct.fd06fe99ef8d5695:core.Destroy.Core"(ptr %.loc18_3.2.value), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f3a741ef694e0907:core.Destroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f3a741ef694e0907:core.Destroy.Core"(ptr %self), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.997b73c74c4e179f:core.Destroy.Core"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b7b274de082dcb96:core.Destroy.Core"(ptr %self) #0 !dbg !117 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b7b274de082dcb96:core.Destroy.Core"(ptr %self) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b7b274de082dcb96:core.Destroy.Core"(ptr %self), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1fff740ae901e27a:core.Destroy.Core"(ptr %self) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1fff740ae901e27a:core.Destroy.Core"(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1fff740ae901e27a:core.Destroy.Core"(ptr %self), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.675c2c8c144a6956:core.Destroy.Core"(ptr %self) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c876c318c747a9fe:core.Destroy.Core"(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !139
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc16_3.1.has_value), !dbg !139
// CHECK:STDOUT:   %.loc16_3.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !139
// CHECK:STDOUT:   call void @"_CSelfDestruct.675c2c8c144a6956:core.Destroy.Core"(ptr %.loc16_3.2.value), !dbg !139
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c876c318c747a9fe:core.Destroy.Core"(ptr %self) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c876c318c747a9fe:core.Destroy.Core"(ptr %self), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c38593eb0b3838a1:core.Destroy.Core"(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !147
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc16_3.1.has_value), !dbg !147
// CHECK:STDOUT:   %.loc16_3.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !147
// CHECK:STDOUT:   call void @"_CSelfDestruct.675c2c8c144a6956:core.Destroy.Core"(ptr %.loc16_3.2.value), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c38593eb0b3838a1:core.Destroy.Core"(ptr %self) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c38593eb0b3838a1:core.Destroy.Core"(ptr %self), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3d12b9ad3598e418:core.Destroy.Core"(ptr %self) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f93e486abe6b5cfe:core.Destroy.Core"(ptr %self) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f93e486abe6b5cfe:core.Destroy.Core"(ptr %self) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f93e486abe6b5cfe:core.Destroy.Core"(ptr %self), !dbg !163
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.0176148a5c50a958(ptr %self) #0 !dbg !165 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.d727653b20be66f9"(ptr %self), !dbg !171
// CHECK:STDOUT:   ret i1 %1, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.0176148a5c50a958(ptr %self) #0 !dbg !174 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.d727653b20be66f9"(ptr %self), !dbg !180
// CHECK:STDOUT:   ret ptr %1, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.423003c0ed122739"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !183 {
// CHECK:STDOUT:   call void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.1c5c1cd4e7e6d532"(ptr %return, i32 %self), !dbg !188
// CHECK:STDOUT:   ret void, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.1c5c1cd4e7e6d532(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !191 {
// CHECK:STDOUT:   call void @"_CNone.74becf39533834ee:OptionalStorage.Core.df66d5def8ac52f2"(ptr %return), !dbg !194
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.2fca651b624aec3b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !197 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.423003c0ed122739"(ptr %return, i32 %self), !dbg !202
// CHECK:STDOUT:   ret void, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.910e80862b30ac54"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !205 {
// CHECK:STDOUT:   call void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.57b234c83081ed44"(ptr %return, i32 %self), !dbg !210
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.f0eed05b796df3bf"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !213 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.910e80862b30ac54"(ptr %return, i32 %self), !dbg !218
// CHECK:STDOUT:   ret void, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.2fca651b624aec3b"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !221 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.423003c0ed122739"(ptr %return, i32 %self), !dbg !227
// CHECK:STDOUT:   ret void, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.db742163b958e819"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !230 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.2fca651b624aec3b"(ptr %return, i32 %self), !dbg !236
// CHECK:STDOUT:   ret void, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.f0eed05b796df3bf"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !239 {
// CHECK:STDOUT:   call void @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.910e80862b30ac54"(ptr %return, i32 %self), !dbg !245
// CHECK:STDOUT:   ret void, !dbg !246
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.5fc0811aa20c2f57"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !248 {
// CHECK:STDOUT:   call void @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.f0eed05b796df3bf"(ptr %return, i32 %self), !dbg !254
// CHECK:STDOUT:   ret void, !dbg !255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.d727653b20be66f9"(ptr %value) #0 !dbg !257 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !263
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !264
// CHECK:STDOUT:   ret i1 %2, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.d727653b20be66f9"(ptr %value) #0 !dbg !267 {
// CHECK:STDOUT:   ret ptr %value, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.1c5c1cd4e7e6d532"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !275 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.1c5c1cd4e7e6d532(ptr %return, i32 %self), !dbg !281
// CHECK:STDOUT:   ret void, !dbg !282
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.74becf39533834ee:OptionalStorage.Core.df66d5def8ac52f2"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !284 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !291
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !291
// CHECK:STDOUT:   ret void, !dbg !292
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.57b234c83081ed44"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !293 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !298
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !298
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.6fb4abd7eb9b95d9"(i32 %self), !dbg !298
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !298
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !298
// CHECK:STDOUT:   call void @_CSome.Optional.Core.02920cdb7a7b6313(ptr %return, i32 %2), !dbg !299
// CHECK:STDOUT:   call void @"_CSelfDestruct.ed3b5dcc9f2a0f16:core.Destroy.Core"(ptr %temp), !dbg !298
// CHECK:STDOUT:   ret void, !dbg !300
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.1c5c1cd4e7e6d532(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !302 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.df66d5def8ac52f2"(ptr %return, i32 %value), !dbg !307
// CHECK:STDOUT:   ret void, !dbg !308
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.6fb4abd7eb9b95d9"(i32 %self) #0 !dbg !310 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.5c3c9fee3ed8d376:ImplicitAs.5f11b08eda604ac7.Core.d6cc459014ac5a00"(i32 %self), !dbg !315
// CHECK:STDOUT:   ret i32 %1, !dbg !318
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.02920cdb7a7b6313(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !320 {
// CHECK:STDOUT:   call void @"_CSome.74becf39533834ee:OptionalStorage.Core.fdf1c393625b4edd"(ptr %return, i32 %value), !dbg !325
// CHECK:STDOUT:   ret void, !dbg !326
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.df66d5def8ac52f2"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !328 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !337
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !337
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !339
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !339
// CHECK:STDOUT:   ret void, !dbg !340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.5c3c9fee3ed8d376:ImplicitAs.5f11b08eda604ac7.Core.d6cc459014ac5a00"(i32 %self) #0 !dbg !343 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !348
// CHECK:STDOUT:   ret i32 %1, !dbg !350
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.74becf39533834ee:OptionalStorage.Core.fdf1c393625b4edd"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !352 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !361
// CHECK:STDOUT:   %1 = call i32 @"_COp.a7c16f707f533a23:Copy.Core.2aef1d52a6e1f957"(i32 %self), !dbg !362
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !361
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !363
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !363
// CHECK:STDOUT:   ret void, !dbg !364
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !366 {
// CHECK:STDOUT:   ret i32 %self, !dbg !372
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.a7c16f707f533a23:Copy.Core.2aef1d52a6e1f957"(i32 %self) #0 !dbg !375 {
// CHECK:STDOUT:   ret i32 %self, !dbg !382
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.997b73c74c4e179f:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.b7b274de082dcb96:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.3d12b9ad3598e418:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.f93e486abe6b5cfe:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.423003c0ed122739", { 0, 1, 3, 2 }
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
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.49ebf9b244db2604:core.Destroy.Core", scope: null, file: !16, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 164, column: 19, scope: !17)
// CHECK:STDOUT: !20 = !{!18}
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d19d32d3b3a297a1:core.Destroy.Core", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !24)
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
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ed3b5dcc9f2a0f16:core.Destroy.Core", scope: null, file: !4, line: 13, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !75 = !DILocation(line: 13, column: 10, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.ed3b5dcc9f2a0f16:core.Destroy.Core", scope: null, file: !4, line: 13, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !7)
// CHECK:STDOUT: !79 = !DILocation(line: 13, column: 10, scope: !77)
// CHECK:STDOUT: !80 = !{!78}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4f8c0853b2a8342d:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !7)
// CHECK:STDOUT: !83 = !DILocation(line: 18, column: 3, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4f8c0853b2a8342d:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !7)
// CHECK:STDOUT: !87 = !DILocation(line: 18, column: 3, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fd06fe99ef8d5695:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !7)
// CHECK:STDOUT: !91 = !DILocation(line: 18, column: 3, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 18, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2c621ceeef95ce46:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !99 = !DILocation(line: 18, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2c621ceeef95ce46:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !7)
// CHECK:STDOUT: !103 = !DILocation(line: 18, column: 3, scope: !101)
// CHECK:STDOUT: !104 = !{!102}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f3a741ef694e0907:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 18, column: 3, scope: !105)
// CHECK:STDOUT: !108 = !{!106}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f3a741ef694e0907:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !111 = !DILocation(line: 18, column: 3, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.997b73c74c4e179f:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !116)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !7)
// CHECK:STDOUT: !115 = !DILocation(line: 18, column: 3, scope: !113)
// CHECK:STDOUT: !116 = !{!114}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b7b274de082dcb96:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !120)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 18, column: 3, scope: !117)
// CHECK:STDOUT: !120 = !{!118}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b7b274de082dcb96:core.Destroy.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !124)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !7)
// CHECK:STDOUT: !123 = !DILocation(line: 18, column: 3, scope: !121)
// CHECK:STDOUT: !124 = !{!122}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1fff740ae901e27a:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !128)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !7)
// CHECK:STDOUT: !127 = !DILocation(line: 16, column: 3, scope: !125)
// CHECK:STDOUT: !128 = !{!126}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1fff740ae901e27a:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !7)
// CHECK:STDOUT: !131 = !DILocation(line: 16, column: 3, scope: !129)
// CHECK:STDOUT: !132 = !{!130}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.675c2c8c144a6956:core.Destroy.Core", scope: null, file: !4, line: 16, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !136)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !6)
// CHECK:STDOUT: !135 = !DILocation(line: 16, column: 3, scope: !133)
// CHECK:STDOUT: !136 = !{!134}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c876c318c747a9fe:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !7)
// CHECK:STDOUT: !139 = !DILocation(line: 16, column: 3, scope: !137)
// CHECK:STDOUT: !140 = !{!138}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c876c318c747a9fe:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !144)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !7)
// CHECK:STDOUT: !143 = !DILocation(line: 16, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !{!142}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c38593eb0b3838a1:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !148)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !7)
// CHECK:STDOUT: !147 = !DILocation(line: 16, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !{!146}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c38593eb0b3838a1:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !7)
// CHECK:STDOUT: !151 = !DILocation(line: 16, column: 3, scope: !149)
// CHECK:STDOUT: !152 = !{!150}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3d12b9ad3598e418:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !156)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !7)
// CHECK:STDOUT: !155 = !DILocation(line: 16, column: 3, scope: !153)
// CHECK:STDOUT: !156 = !{!154}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f93e486abe6b5cfe:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !160)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !7)
// CHECK:STDOUT: !159 = !DILocation(line: 16, column: 3, scope: !157)
// CHECK:STDOUT: !160 = !{!158}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f93e486abe6b5cfe:core.Destroy.Core", scope: null, file: !4, line: 16, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !164)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !7)
// CHECK:STDOUT: !163 = !DILocation(line: 16, column: 3, scope: !161)
// CHECK:STDOUT: !164 = !{!162}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.0176148a5c50a958", scope: null, file: !16, line: 36, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !173)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !7)
// CHECK:STDOUT: !171 = !DILocation(line: 37, column: 12, scope: !165)
// CHECK:STDOUT: !172 = !DILocation(line: 37, column: 5, scope: !165)
// CHECK:STDOUT: !173 = !{!166}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.0176148a5c50a958", scope: null, file: !16, line: 39, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !182)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !7)
// CHECK:STDOUT: !180 = !DILocation(line: 40, column: 12, scope: !174)
// CHECK:STDOUT: !181 = !DILocation(line: 40, column: 5, scope: !174)
// CHECK:STDOUT: !182 = !{!175}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.423003c0ed122739", scope: null, file: !16, line: 105, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !190)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !6)
// CHECK:STDOUT: !188 = !DILocation(line: 106, column: 12, scope: !183)
// CHECK:STDOUT: !189 = !DILocation(line: 106, column: 5, scope: !183)
// CHECK:STDOUT: !190 = !{!184}
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.1c5c1cd4e7e6d532", scope: null, file: !16, line: 30, type: !39, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !194 = !DILocation(line: 31, column: 12, scope: !191)
// CHECK:STDOUT: !195 = !DILocation(line: 31, column: 5, scope: !191)
// CHECK:STDOUT: !196 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.2fca651b624aec3b", scope: null, file: !196, line: 43, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !204)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !6)
// CHECK:STDOUT: !202 = !DILocation(line: 43, column: 40, scope: !197)
// CHECK:STDOUT: !203 = !DILocation(line: 43, column: 33, scope: !197)
// CHECK:STDOUT: !204 = !{!198}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.796a44cfad36b7ff.Core.910e80862b30ac54", scope: null, file: !16, line: 105, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !212)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !6)
// CHECK:STDOUT: !210 = !DILocation(line: 106, column: 12, scope: !205)
// CHECK:STDOUT: !211 = !DILocation(line: 106, column: 5, scope: !205)
// CHECK:STDOUT: !212 = !{!206}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.f0eed05b796df3bf", scope: null, file: !196, line: 43, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !220)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !6)
// CHECK:STDOUT: !218 = !DILocation(line: 43, column: 40, scope: !213)
// CHECK:STDOUT: !219 = !DILocation(line: 43, column: 33, scope: !213)
// CHECK:STDOUT: !220 = !{!214}
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.2fca651b624aec3b", scope: null, file: !196, line: 47, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !229)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !6)
// CHECK:STDOUT: !227 = !DILocation(line: 47, column: 34, scope: !221)
// CHECK:STDOUT: !228 = !DILocation(line: 47, column: 27, scope: !221)
// CHECK:STDOUT: !229 = !{!222}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.db742163b958e819", scope: null, file: !196, line: 47, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !238)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !6)
// CHECK:STDOUT: !236 = !DILocation(line: 47, column: 34, scope: !230)
// CHECK:STDOUT: !237 = !DILocation(line: 47, column: 27, scope: !230)
// CHECK:STDOUT: !238 = !{!231}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.f0eed05b796df3bf", scope: null, file: !196, line: 47, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !247)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !6)
// CHECK:STDOUT: !245 = !DILocation(line: 47, column: 34, scope: !239)
// CHECK:STDOUT: !246 = !DILocation(line: 47, column: 27, scope: !239)
// CHECK:STDOUT: !247 = !{!240}
// CHECK:STDOUT: !248 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c18e58a62bf591fb:ImplicitAs.eb057aa32837c84e.Core.5fc0811aa20c2f57", scope: null, file: !196, line: 47, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !256)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 1, scope: !248, type: !6)
// CHECK:STDOUT: !254 = !DILocation(line: 47, column: 34, scope: !248)
// CHECK:STDOUT: !255 = !DILocation(line: 47, column: 27, scope: !248)
// CHECK:STDOUT: !256 = !{!249}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.d727653b20be66f9", scope: null, file: !16, line: 172, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !266)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !7)
// CHECK:STDOUT: !263 = !DILocation(line: 173, column: 16, scope: !257)
// CHECK:STDOUT: !264 = !DILocation(line: 173, column: 12, scope: !257)
// CHECK:STDOUT: !265 = !DILocation(line: 173, column: 5, scope: !257)
// CHECK:STDOUT: !266 = !{!258}
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.d727653b20be66f9", scope: null, file: !16, line: 175, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !274)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !7)
// CHECK:STDOUT: !273 = !DILocation(line: 176, column: 5, scope: !267)
// CHECK:STDOUT: !274 = !{!268}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.1c5c1cd4e7e6d532", scope: null, file: !16, line: 80, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !283)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !6)
// CHECK:STDOUT: !281 = !DILocation(line: 81, column: 12, scope: !275)
// CHECK:STDOUT: !282 = !DILocation(line: 81, column: 5, scope: !275)
// CHECK:STDOUT: !283 = !{!276}
// CHECK:STDOUT: !284 = distinct !DISubprogram(name: "None", linkageName: "_CNone.74becf39533834ee:OptionalStorage.Core.df66d5def8ac52f2", scope: null, file: !16, line: 125, type: !39, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !291 = !DILocation(line: 128, column: 5, scope: !284)
// CHECK:STDOUT: !292 = !DILocation(line: 129, column: 5, scope: !284)
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.d62727437df45535:OptionalAs.1dac4564233fecd1.Core.57b234c83081ed44", scope: null, file: !16, line: 87, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !301)
// CHECK:STDOUT: !294 = !DILocalVariable(arg: 1, scope: !293, type: !6)
// CHECK:STDOUT: !298 = !DILocation(line: 88, column: 29, scope: !293)
// CHECK:STDOUT: !299 = !DILocation(line: 88, column: 12, scope: !293)
// CHECK:STDOUT: !300 = !DILocation(line: 88, column: 5, scope: !293)
// CHECK:STDOUT: !301 = !{!294}
// CHECK:STDOUT: !302 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.1c5c1cd4e7e6d532", scope: null, file: !16, line: 33, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !309)
// CHECK:STDOUT: !303 = !DILocalVariable(arg: 1, scope: !302, type: !6)
// CHECK:STDOUT: !307 = !DILocation(line: 34, column: 12, scope: !302)
// CHECK:STDOUT: !308 = !DILocation(line: 34, column: 5, scope: !302)
// CHECK:STDOUT: !309 = !{!303}
// CHECK:STDOUT: !310 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ef077c075a7e2432:ImplicitAs.ad22d1bbc0605210.Core.6fb4abd7eb9b95d9", scope: null, file: !196, line: 43, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !319)
// CHECK:STDOUT: !311 = !DILocalVariable(arg: 1, scope: !310, type: !6)
// CHECK:STDOUT: !315 = !DILocation(line: 43, column: 40, scope: !310)
// CHECK:STDOUT: !316 = !{!6, !6}
// CHECK:STDOUT: !317 = !DISubroutineType(types: !316)
// CHECK:STDOUT: !318 = !DILocation(line: 43, column: 33, scope: !310)
// CHECK:STDOUT: !319 = !{!311}
// CHECK:STDOUT: !320 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.02920cdb7a7b6313", scope: null, file: !16, line: 33, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !327)
// CHECK:STDOUT: !321 = !DILocalVariable(arg: 1, scope: !320, type: !6)
// CHECK:STDOUT: !325 = !DILocation(line: 34, column: 12, scope: !320)
// CHECK:STDOUT: !326 = !DILocation(line: 34, column: 5, scope: !320)
// CHECK:STDOUT: !327 = !{!321}
// CHECK:STDOUT: !328 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.df66d5def8ac52f2", scope: null, file: !16, line: 131, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !341)
// CHECK:STDOUT: !329 = !DILocalVariable(arg: 1, scope: !328, type: !6)
// CHECK:STDOUT: !337 = !DILocation(line: 137, column: 5, scope: !328)
// CHECK:STDOUT: !339 = !DILocation(line: 138, column: 5, scope: !328)
// CHECK:STDOUT: !340 = !DILocation(line: 139, column: 5, scope: !328)
// CHECK:STDOUT: !341 = !{!329}
// CHECK:STDOUT: !342 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !343 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.5c3c9fee3ed8d376:ImplicitAs.5f11b08eda604ac7.Core.d6cc459014ac5a00", scope: null, file: !342, line: 64, type: !317, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !351)
// CHECK:STDOUT: !344 = !DILocalVariable(arg: 1, scope: !343, type: !6)
// CHECK:STDOUT: !348 = !DILocation(line: 66, column: 17, scope: !343)
// CHECK:STDOUT: !350 = !DILocation(line: 66, column: 5, scope: !343)
// CHECK:STDOUT: !351 = !{!344}
// CHECK:STDOUT: !352 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.74becf39533834ee:OptionalStorage.Core.fdf1c393625b4edd", scope: null, file: !16, line: 131, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !365)
// CHECK:STDOUT: !353 = !DILocalVariable(arg: 1, scope: !352, type: !6)
// CHECK:STDOUT: !361 = !DILocation(line: 137, column: 5, scope: !352)
// CHECK:STDOUT: !362 = !DILocation(line: 137, column: 28, scope: !352)
// CHECK:STDOUT: !363 = !DILocation(line: 138, column: 5, scope: !352)
// CHECK:STDOUT: !364 = !DILocation(line: 139, column: 5, scope: !352)
// CHECK:STDOUT: !365 = !{!353}
// CHECK:STDOUT: !366 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !342, line: 59, type: !317, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !373)
// CHECK:STDOUT: !367 = !DILocalVariable(arg: 1, scope: !366, type: !6)
// CHECK:STDOUT: !372 = !DILocation(line: 59, column: 28, scope: !366)
// CHECK:STDOUT: !373 = !{!367}
// CHECK:STDOUT: !374 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !375 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a7c16f707f533a23:Copy.Core.2aef1d52a6e1f957", scope: null, file: !374, line: 23, type: !36, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !383)
// CHECK:STDOUT: !376 = !DILocalVariable(arg: 1, scope: !375, type: !6)
// CHECK:STDOUT: !382 = !DILocation(line: 23, column: 25, scope: !375)
// CHECK:STDOUT: !383 = !{!376}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.Copyable.Main:Copy.Core"(ptr sret({}), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !386 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !388
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb3dd30fa3a4aa62:core.Destroy.Core"(ptr %self) #0 !dbg !390 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !392
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr sret({ {}, i1 }) %return, ptr %c) #0 !dbg !394 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31"(ptr %return, ptr %c), !dbg !399
// CHECK:STDOUT:   ret void, !dbg !400
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !402 {
// CHECK:STDOUT:   call void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return, ptr %self), !dbg !408
// CHECK:STDOUT:   ret void, !dbg !409
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return, ptr %value) #0 !dbg !411 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %value), !dbg !417
// CHECK:STDOUT:   br i1 %1, label %2, label %3, !dbg !418
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 0, !dbg !419
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return, ptr %value1), !dbg !420
// CHECK:STDOUT:   ret void, !dbg !421
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %return), !dbg !422
// CHECK:STDOUT:   ret void, !dbg !423
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr %value) #0 !dbg !425 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %value, i32 0, i32 1, !dbg !430
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !430
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !430
// CHECK:STDOUT:   ret i1 %2, !dbg !431
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return, ptr %self) #0 !dbg !433 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 0, !dbg !442
// CHECK:STDOUT:   call void @"_COp.Copyable.Main:Copy.Core"(ptr %value, ptr %self), !dbg !443
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !444
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !444
// CHECK:STDOUT:   ret void, !dbg !445
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c"(ptr sret({ {}, i1 }) %return) #0 !dbg !447 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { {}, i1 }, ptr %return, i32 0, i32 1, !dbg !454
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !454
// CHECK:STDOUT:   ret void, !dbg !455
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!385}
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
// CHECK:STDOUT: !384 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !385 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !384, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !386 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !384, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !385, retainedNodes: !389)
// CHECK:STDOUT: !387 = !DILocalVariable(arg: 1, scope: !386, type: !7)
// CHECK:STDOUT: !388 = !DILocation(line: 9, column: 9, scope: !386)
// CHECK:STDOUT: !389 = !{!387}
// CHECK:STDOUT: !390 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb3dd30fa3a4aa62:core.Destroy.Core", scope: null, file: !384, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !385, retainedNodes: !393)
// CHECK:STDOUT: !391 = !DILocalVariable(arg: 1, scope: !390, type: !7)
// CHECK:STDOUT: !392 = !DILocation(line: 9, column: 9, scope: !390)
// CHECK:STDOUT: !393 = !{!391}
// CHECK:STDOUT: !394 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !384, line: 9, type: !13, spFlags: DISPFlagDefinition, unit: !385, retainedNodes: !401)
// CHECK:STDOUT: !395 = !DILocalVariable(arg: 1, scope: !394, type: !7)
// CHECK:STDOUT: !399 = !DILocation(line: 11, column: 10, scope: !394)
// CHECK:STDOUT: !400 = !DILocation(line: 11, column: 3, scope: !394)
// CHECK:STDOUT: !401 = !{!395}
// CHECK:STDOUT: !402 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.a2b7303f9ef09f31", scope: null, file: !16, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !385, retainedNodes: !410)
// CHECK:STDOUT: !403 = !DILocalVariable(arg: 1, scope: !402, type: !7)
// CHECK:STDOUT: !408 = !DILocation(line: 51, column: 12, scope: !402)
// CHECK:STDOUT: !409 = !DILocation(line: 51, column: 5, scope: !402)
// CHECK:STDOUT: !410 = !{!403}
// CHECK:STDOUT: !411 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 147, type: !13, spFlags: DISPFlagDefinition, unit: !385, retainedNodes: !424)
// CHECK:STDOUT: !412 = !DILocalVariable(arg: 1, scope: !411, type: !7)
// CHECK:STDOUT: !417 = !DILocation(line: 148, column: 9, scope: !411)
// CHECK:STDOUT: !418 = !DILocation(line: 148, column: 8, scope: !411)
// CHECK:STDOUT: !419 = !DILocation(line: 149, column: 15, scope: !411)
// CHECK:STDOUT: !420 = !DILocation(line: 149, column: 14, scope: !411)
// CHECK:STDOUT: !421 = !DILocation(line: 149, column: 7, scope: !411)
// CHECK:STDOUT: !422 = !DILocation(line: 151, column: 14, scope: !411)
// CHECK:STDOUT: !423 = !DILocation(line: 151, column: 7, scope: !411)
// CHECK:STDOUT: !424 = !{!412}
// CHECK:STDOUT: !425 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 141, type: !13, spFlags: DISPFlagDefinition, unit: !385, retainedNodes: !432)
// CHECK:STDOUT: !426 = !DILocalVariable(arg: 1, scope: !425, type: !7)
// CHECK:STDOUT: !430 = !DILocation(line: 142, column: 12, scope: !425)
// CHECK:STDOUT: !431 = !DILocation(line: 142, column: 5, scope: !425)
// CHECK:STDOUT: !432 = !{!426}
// CHECK:STDOUT: !433 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 131, type: !13, spFlags: DISPFlagDefinition, unit: !385, retainedNodes: !446)
// CHECK:STDOUT: !434 = !DILocalVariable(arg: 1, scope: !433, type: !7)
// CHECK:STDOUT: !442 = !DILocation(line: 137, column: 5, scope: !433)
// CHECK:STDOUT: !443 = !DILocation(line: 137, column: 28, scope: !433)
// CHECK:STDOUT: !444 = !DILocation(line: 138, column: 5, scope: !433)
// CHECK:STDOUT: !445 = !DILocation(line: 139, column: 5, scope: !433)
// CHECK:STDOUT: !446 = !{!434}
// CHECK:STDOUT: !447 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d34df0287397e90c", scope: null, file: !16, line: 125, type: !39, spFlags: DISPFlagDefinition, unit: !385)
// CHECK:STDOUT: !454 = !DILocation(line: 128, column: 5, scope: !447)
// CHECK:STDOUT: !455 = !DILocation(line: 129, column: 5, scope: !447)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_pointer.carbon'
// CHECK:STDOUT: source_filename = "copy_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %self) #0 !dbg !458 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !460
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %c) #0 !dbg !462 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.as.Copy.impl.Op.call = call ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %c), !dbg !467
// CHECK:STDOUT:   ret ptr %Optional.as.Copy.impl.Op.call, !dbg !468
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608"(ptr %self) #0 !dbg !470 {
// CHECK:STDOUT:   ret ptr %self, !dbg !477
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!457}
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
// CHECK:STDOUT: !456 = !DIFile(filename: "copy_pointer.carbon", directory: "")
// CHECK:STDOUT: !457 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !456, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !458 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b633cf020343e69:core.Destroy.Core", scope: null, file: !16, line: 164, type: !9, spFlags: DISPFlagDefinition, unit: !457, retainedNodes: !461)
// CHECK:STDOUT: !459 = !DILocalVariable(arg: 1, scope: !458, type: !7)
// CHECK:STDOUT: !460 = !DILocation(line: 164, column: 19, scope: !458)
// CHECK:STDOUT: !461 = !{!459}
// CHECK:STDOUT: !462 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !456, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !457, retainedNodes: !469)
// CHECK:STDOUT: !463 = !DILocalVariable(arg: 1, scope: !462, type: !7)
// CHECK:STDOUT: !467 = !DILocation(line: 6, column: 10, scope: !462)
// CHECK:STDOUT: !468 = !DILocation(line: 6, column: 3, scope: !462)
// CHECK:STDOUT: !469 = !{!463}
// CHECK:STDOUT: !470 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Optional.24b7347a93330e68.Core:Copy.Core.3e2e92aa38469608", scope: null, file: !16, line: 50, type: !13, spFlags: DISPFlagDefinition, unit: !457, retainedNodes: !478)
// CHECK:STDOUT: !471 = !DILocalVariable(arg: 1, scope: !470, type: !7)
// CHECK:STDOUT: !477 = !DILocation(line: 51, column: 5, scope: !470)
// CHECK:STDOUT: !478 = !{!471}
// CHECK:STDOUT:
