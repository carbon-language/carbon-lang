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
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret({ i32, i1 }) %return, ptr %o) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.cb8cde10e3c2d324(ptr %o), !dbg !10
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.cb8cde10e3c2d324(ptr %o), !dbg !12
// CHECK:STDOUT:   %.loc18_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !13
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.622e77b643a0d3a8"(ptr %return, i32 %.loc18_12.2), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.d4e523823e737747(ptr %return), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc24 = alloca { i32, i1 }, align 8, !dbg !24
// CHECK:STDOUT:   %_.var.loc25 = alloca { i32, i1 }, align 8, !dbg !25
// CHECK:STDOUT:   %_.var.loc26 = alloca { i32, i1 }, align 8, !dbg !26
// CHECK:STDOUT:   %_.var.loc27 = alloca { i32, i1 }, align 8, !dbg !27
// CHECK:STDOUT:   %_.var.loc28 = alloca { i32, i1 }, align 8, !dbg !28
// CHECK:STDOUT:   %_.var.loc29 = alloca { i32, i1 }, align 8, !dbg !29
// CHECK:STDOUT:   %_.var.loc30 = alloca { i32, i1 }, align 8, !dbg !30
// CHECK:STDOUT:   %_.var.loc31 = alloca { i32, i1 }, align 8, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc24), !dbg !24
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.622e77b643a0d3a8"(ptr %_.var.loc24, i32 %a), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc25), !dbg !25
// CHECK:STDOUT:   call void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.bd3a79eb4ffd3025"(ptr %_.var.loc25, i32 %a), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc26), !dbg !26
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.a7b795aef3d7a0e5"(ptr %_.var.loc26, i32 %a), !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc27), !dbg !27
// CHECK:STDOUT:   call void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.b37806f50d6ac6f2"(ptr %_.var.loc27, i32 %a), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc28), !dbg !28
// CHECK:STDOUT:   call void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.bd3a79eb4ffd3025"(ptr %_.var.loc28, i32 %b), !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc29), !dbg !29
// CHECK:STDOUT:   call void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.fdb5566b9a89e24d"(ptr %_.var.loc29, i32 %b), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc30), !dbg !30
// CHECK:STDOUT:   call void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.b37806f50d6ac6f2"(ptr %_.var.loc30, i32 %b), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc31), !dbg !31
// CHECK:STDOUT:   call void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.5fb98646c26c8976"(ptr %_.var.loc31, i32 %b), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.cb8cde10e3c2d324(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT:   %1 = icmp eq ptr %self, null, !dbg !37
// CHECK:STDOUT:   ret i1 %1, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.cb8cde10e3c2d324(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %self), !dbg !42
// CHECK:STDOUT:   ret ptr %1, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.622e77b643a0d3a8"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !44 {
// CHECK:STDOUT:   call void @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d4e523823e737747"(ptr %return, i32 %self), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.d4e523823e737747(ptr sret({ i32, i1 }) %return) #0 !dbg !51 {
// CHECK:STDOUT:   call void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.329e9a7481f16207"(ptr %return), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.bd3a79eb4ffd3025"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !56 {
// CHECK:STDOUT:   %temp = alloca { i32, i1 }, align 8, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !60
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.622e77b643a0d3a8"(ptr %temp, i32 %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.a7b795aef3d7a0e5"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !62 {
// CHECK:STDOUT:   call void @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.dbad48707eb766d5"(ptr %return, i32 %self), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.b37806f50d6ac6f2"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !67 {
// CHECK:STDOUT:   %temp = alloca { i32, i1 }, align 8, !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !70
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.a7b795aef3d7a0e5"(ptr %temp, i32 %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.bd3a79eb4ffd3025"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !72 {
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.622e77b643a0d3a8"(ptr %return, i32 %self), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.fdb5566b9a89e24d"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !77 {
// CHECK:STDOUT:   call void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.bd3a79eb4ffd3025"(ptr %return, i32 %self), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.b37806f50d6ac6f2"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !82 {
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.a7b795aef3d7a0e5"(ptr %return, i32 %self), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.5fb98646c26c8976"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.b37806f50d6ac6f2"(ptr %return, i32 %self), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %value) #0 !dbg !92 {
// CHECK:STDOUT:   ret ptr %value, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d4e523823e737747"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !96 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.d4e523823e737747(ptr %return, i32 %self), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.329e9a7481f16207"(ptr sret({ i32, i1 }) %return) #0 !dbg !101 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !102
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.dbad48707eb766d5"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !104 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !107
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !107
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.b930bfdac0979466"(i32 %self), !dbg !107
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !107
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !107
// CHECK:STDOUT:   call void @_CSome.Optional.Core.2a814810f68c37ba(ptr %return, i32 %2), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.d4e523823e737747(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !110 {
// CHECK:STDOUT:   call void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.329e9a7481f16207"(ptr %return, i32 %value), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.b930bfdac0979466"(i32 %self) #0 !dbg !115 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !118
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !118
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.64ccbb8e5d9a0b8e"(i32 %self), !dbg !118
// CHECK:STDOUT:   ret i32 %1, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.2a814810f68c37ba(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !120 {
// CHECK:STDOUT:   call void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.d9fa83018d7f62e1"(ptr %return, i32 %value), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.329e9a7481f16207"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !125 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !128
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !128
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !129
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !129
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.64ccbb8e5d9a0b8e"(i32 %self) #0 !dbg !131 {
// CHECK:STDOUT:   ret i32 %self, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.d9fa83018d7f62e1"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !137 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !140
// CHECK:STDOUT:   %1 = call i32 @"_COp.34e7a685d3648824:Copy.Core.64ccbb8e5d9a0b8e"(i32 %self), !dbg !141
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !140
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !142
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !142
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.34e7a685d3648824:Copy.Core.64ccbb8e5d9a0b8e"(i32 %self) #0 !dbg !144 {
// CHECK:STDOUT:   ret i32 %self, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.622e77b643a0d3a8", { 0, 1, 3, 2 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 11, 10, 9, 8, 7, 6, 5, 4 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 7, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 6, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 18, column: 13, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 18, column: 12, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 20, column: 10, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !3, line: 23, type: !18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !20, !20}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!22, !23}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !17, type: !20)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 2, scope: !17, type: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 24, column: 3, scope: !17)
// CHECK:STDOUT: !25 = !DILocation(line: 25, column: 3, scope: !17)
// CHECK:STDOUT: !26 = !DILocation(line: 26, column: 3, scope: !17)
// CHECK:STDOUT: !27 = !DILocation(line: 27, column: 3, scope: !17)
// CHECK:STDOUT: !28 = !DILocation(line: 28, column: 3, scope: !17)
// CHECK:STDOUT: !29 = !DILocation(line: 29, column: 3, scope: !17)
// CHECK:STDOUT: !30 = !DILocation(line: 30, column: 3, scope: !17)
// CHECK:STDOUT: !31 = !DILocation(line: 31, column: 3, scope: !17)
// CHECK:STDOUT: !32 = !DILocation(line: 23, column: 1, scope: !17)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.cb8cde10e3c2d324", scope: null, file: !34, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !34 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 33, column: 12, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 33, column: 5, scope: !33)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.cb8cde10e3c2d324", scope: null, file: !34, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 36, column: 12, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 36, column: 5, scope: !39)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.622e77b643a0d3a8", scope: null, file: !34, line: 82, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !46 = !{!7, !20}
// CHECK:STDOUT: !47 = !{!48}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !44, type: !20)
// CHECK:STDOUT: !49 = !DILocation(line: 83, column: 12, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 83, column: 5, scope: !44)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.d4e523823e737747", scope: null, file: !34, line: 26, type: !52, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !52 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !53 = !{!7}
// CHECK:STDOUT: !54 = !DILocation(line: 27, column: 12, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 27, column: 5, scope: !51)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.bd3a79eb4ffd3025", scope: null, file: !57, line: 28, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !57 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !56, type: !20)
// CHECK:STDOUT: !60 = !DILocation(line: 28, column: 45, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 28, column: 38, scope: !56)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.a7b795aef3d7a0e5", scope: null, file: !34, line: 82, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 83, column: 12, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 83, column: 5, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.b37806f50d6ac6f2", scope: null, file: !57, line: 28, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !20)
// CHECK:STDOUT: !70 = !DILocation(line: 28, column: 45, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 28, column: 38, scope: !67)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.bd3a79eb4ffd3025", scope: null, file: !57, line: 32, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !73)
// CHECK:STDOUT: !73 = !{!74}
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !72, type: !20)
// CHECK:STDOUT: !75 = !DILocation(line: 32, column: 45, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 32, column: 38, scope: !72)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.fdb5566b9a89e24d", scope: null, file: !57, line: 32, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !77, type: !20)
// CHECK:STDOUT: !80 = !DILocation(line: 32, column: 45, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 32, column: 38, scope: !77)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.b37806f50d6ac6f2", scope: null, file: !57, line: 32, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !83)
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !82, type: !20)
// CHECK:STDOUT: !85 = !DILocation(line: 32, column: 45, scope: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 32, column: 38, scope: !82)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.5fb98646c26c8976", scope: null, file: !57, line: 32, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !90 = !DILocation(line: 32, column: 45, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 32, column: 38, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4", scope: null, file: !34, line: 131, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 132, column: 5, scope: !92)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.d4e523823e737747", scope: null, file: !34, line: 68, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !97)
// CHECK:STDOUT: !97 = !{!98}
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !96, type: !20)
// CHECK:STDOUT: !99 = !DILocation(line: 69, column: 12, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 69, column: 5, scope: !96)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "None", linkageName: "_CNone.3e8267224c5dc9c2:OptionalStorage.Core.329e9a7481f16207", scope: null, file: !34, line: 99, type: !52, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !102 = !DILocation(line: 101, column: 5, scope: !101)
// CHECK:STDOUT: !103 = !DILocation(line: 102, column: 5, scope: !101)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.dbad48707eb766d5", scope: null, file: !34, line: 75, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !105)
// CHECK:STDOUT: !105 = !{!106}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !104, type: !20)
// CHECK:STDOUT: !107 = !DILocation(line: 76, column: 29, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 76, column: 12, scope: !104)
// CHECK:STDOUT: !109 = !DILocation(line: 76, column: 5, scope: !104)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d4e523823e737747", scope: null, file: !34, line: 29, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !111)
// CHECK:STDOUT: !111 = !{!112}
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !110, type: !20)
// CHECK:STDOUT: !113 = !DILocation(line: 30, column: 12, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 30, column: 5, scope: !110)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.b930bfdac0979466", scope: null, file: !57, line: 28, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !116)
// CHECK:STDOUT: !116 = !{!117}
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !115, type: !20)
// CHECK:STDOUT: !118 = !DILocation(line: 28, column: 45, scope: !115)
// CHECK:STDOUT: !119 = !DILocation(line: 28, column: 38, scope: !115)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.2a814810f68c37ba", scope: null, file: !34, line: 29, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !121)
// CHECK:STDOUT: !121 = !{!122}
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !120, type: !20)
// CHECK:STDOUT: !123 = !DILocation(line: 30, column: 12, scope: !120)
// CHECK:STDOUT: !124 = !DILocation(line: 30, column: 5, scope: !120)
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.3e8267224c5dc9c2:OptionalStorage.Core.329e9a7481f16207", scope: null, file: !34, line: 104, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !126)
// CHECK:STDOUT: !126 = !{!127}
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !125, type: !20)
// CHECK:STDOUT: !128 = !DILocation(line: 108, column: 5, scope: !125)
// CHECK:STDOUT: !129 = !DILocation(line: 109, column: 5, scope: !125)
// CHECK:STDOUT: !130 = !DILocation(line: 110, column: 5, scope: !125)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.64ccbb8e5d9a0b8e", scope: null, file: !57, line: 24, type: !132, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DISubroutineType(types: !133)
// CHECK:STDOUT: !133 = !{!20, !20}
// CHECK:STDOUT: !134 = !{!135}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !131, type: !20)
// CHECK:STDOUT: !136 = !DILocation(line: 24, column: 38, scope: !131)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.3e8267224c5dc9c2:OptionalStorage.Core.d9fa83018d7f62e1", scope: null, file: !34, line: 104, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !138)
// CHECK:STDOUT: !138 = !{!139}
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !137, type: !20)
// CHECK:STDOUT: !140 = !DILocation(line: 108, column: 5, scope: !137)
// CHECK:STDOUT: !141 = !DILocation(line: 108, column: 28, scope: !137)
// CHECK:STDOUT: !142 = !DILocation(line: 109, column: 5, scope: !137)
// CHECK:STDOUT: !143 = !DILocation(line: 110, column: 5, scope: !137)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Op", linkageName: "_COp.34e7a685d3648824:Copy.Core.64ccbb8e5d9a0b8e", scope: null, file: !145, line: 19, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !146)
// CHECK:STDOUT: !145 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !146 = !{!147}
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !144, type: !20)
// CHECK:STDOUT: !148 = !DILocation(line: 19, column: 33, scope: !144)
