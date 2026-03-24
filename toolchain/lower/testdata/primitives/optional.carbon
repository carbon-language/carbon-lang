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
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.03e3348579103d79(ptr %o), !dbg !10
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.03e3348579103d79(ptr %o), !dbg !12
// CHECK:STDOUT:   %.loc18_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !13
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.6b1f19368d09e579"(ptr %return, i32 %.loc18_12.2), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.30ced37eb1e63547(ptr %return), !dbg !15
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
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.6b1f19368d09e579"(ptr %_.var.loc24, i32 %a), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc25), !dbg !25
// CHECK:STDOUT:   call void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.77b263a5fd240ddf"(ptr %_.var.loc25, i32 %a), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc26), !dbg !26
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.787d589b54a95071"(ptr %_.var.loc26, i32 %a), !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc27), !dbg !27
// CHECK:STDOUT:   call void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.25fb32d66f981ccd"(ptr %_.var.loc27, i32 %a), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc28), !dbg !28
// CHECK:STDOUT:   call void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.77b263a5fd240ddf"(ptr %_.var.loc28, i32 %b), !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc29), !dbg !29
// CHECK:STDOUT:   call void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.693532dcb04bf6dc"(ptr %_.var.loc29, i32 %b), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc30), !dbg !30
// CHECK:STDOUT:   call void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.25fb32d66f981ccd"(ptr %_.var.loc30, i32 %b), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc31), !dbg !31
// CHECK:STDOUT:   call void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.1c991c9fdfba671f"(ptr %_.var.loc31, i32 %b), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.03e3348579103d79(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %self), !dbg !37
// CHECK:STDOUT:   ret i1 %1, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.03e3348579103d79(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %self), !dbg !42
// CHECK:STDOUT:   ret ptr %1, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.6b1f19368d09e579"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !44 {
// CHECK:STDOUT:   call void @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.30ced37eb1e63547"(ptr %return, i32 %self), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.30ced37eb1e63547(ptr sret({ i32, i1 }) %return) #0 !dbg !51 {
// CHECK:STDOUT:   call void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr %return), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.77b263a5fd240ddf"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !56 {
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.6b1f19368d09e579"(ptr %return, i32 %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.787d589b54a95071"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !62 {
// CHECK:STDOUT:   call void @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.5d35ad54738af8e4"(ptr %return, i32 %self), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.25fb32d66f981ccd"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !67 {
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.787d589b54a95071"(ptr %return, i32 %self), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.77b263a5fd240ddf"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !72 {
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.6b1f19368d09e579"(ptr %return, i32 %self), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.693532dcb04bf6dc"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !77 {
// CHECK:STDOUT:   call void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.77b263a5fd240ddf"(ptr %return, i32 %self), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.25fb32d66f981ccd"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !82 {
// CHECK:STDOUT:   call void @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.787d589b54a95071"(ptr %return, i32 %self), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.1c991c9fdfba671f"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.25fb32d66f981ccd"(ptr %return, i32 %self), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %value) #0 !dbg !92 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !95
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !96
// CHECK:STDOUT:   ret i1 %2, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4"(ptr %value) #0 !dbg !98 {
// CHECK:STDOUT:   ret ptr %value, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.30ced37eb1e63547"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !102 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.30ced37eb1e63547(ptr %return, i32 %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr sret({ i32, i1 }) %return) #0 !dbg !107 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !108
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.5d35ad54738af8e4"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !110 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !113
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !113
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.90a4f5ca3e06badd"(i32 %self), !dbg !113
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !113
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !113
// CHECK:STDOUT:   call void @_CSome.Optional.Core.b0246d3dd29c9210(ptr %return, i32 %2), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.30ced37eb1e63547(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !116 {
// CHECK:STDOUT:   call void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr %return, i32 %value), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.90a4f5ca3e06badd"(i32 %self) #0 !dbg !121 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.628184612c80324d"(i32 %self), !dbg !124
// CHECK:STDOUT:   ret i32 %1, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.b0246d3dd29c9210(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !126 {
// CHECK:STDOUT:   call void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.5325ffa2b57e13bd"(ptr %return, i32 %value), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !131 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !134
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !134
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !135
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.628184612c80324d"(i32 %self) #0 !dbg !137 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !143
// CHECK:STDOUT:   ret i32 %1, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.3e8267224c5dc9c2:OptionalStorage.Core.5325ffa2b57e13bd"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !145 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !148
// CHECK:STDOUT:   %1 = call i32 @"_COp.34e7a685d3648824:Copy.Core.64ccbb8e5d9a0b8e"(i32 %self), !dbg !149
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !148
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !150
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !152 {
// CHECK:STDOUT:   ret i32 %self, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.34e7a685d3648824:Copy.Core.64ccbb8e5d9a0b8e"(i32 %self) #0 !dbg !156 {
// CHECK:STDOUT:   ret i32 %self, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.6b1f19368d09e579", { 0, 1, 3, 2 }
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
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.03e3348579103d79", scope: null, file: !34, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !34 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 34, column: 12, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 34, column: 5, scope: !33)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.03e3348579103d79", scope: null, file: !34, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 37, column: 12, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 37, column: 5, scope: !39)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.6b1f19368d09e579", scope: null, file: !34, line: 96, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !46 = !{!7, !20}
// CHECK:STDOUT: !47 = !{!48}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !44, type: !20)
// CHECK:STDOUT: !49 = !DILocation(line: 97, column: 12, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 97, column: 5, scope: !44)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.30ced37eb1e63547", scope: null, file: !34, line: 27, type: !52, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !52 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !53 = !{!7}
// CHECK:STDOUT: !54 = !DILocation(line: 28, column: 12, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 28, column: 5, scope: !51)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.77b263a5fd240ddf", scope: null, file: !57, line: 40, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !57 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !56, type: !20)
// CHECK:STDOUT: !60 = !DILocation(line: 40, column: 45, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 40, column: 38, scope: !56)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.787d589b54a95071", scope: null, file: !34, line: 96, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 97, column: 12, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 97, column: 5, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.25fb32d66f981ccd", scope: null, file: !57, line: 40, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !20)
// CHECK:STDOUT: !70 = !DILocation(line: 40, column: 45, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 40, column: 38, scope: !67)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.77b263a5fd240ddf", scope: null, file: !57, line: 44, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !73)
// CHECK:STDOUT: !73 = !{!74}
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !72, type: !20)
// CHECK:STDOUT: !75 = !DILocation(line: 44, column: 45, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 44, column: 38, scope: !72)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.693532dcb04bf6dc", scope: null, file: !57, line: 44, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !77, type: !20)
// CHECK:STDOUT: !80 = !DILocation(line: 44, column: 45, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 44, column: 38, scope: !77)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.25fb32d66f981ccd", scope: null, file: !57, line: 44, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !83)
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !82, type: !20)
// CHECK:STDOUT: !85 = !DILocation(line: 44, column: 45, scope: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 44, column: 38, scope: !82)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.17f42c13d842b71d:ImplicitAs.eb057aa32837c84e.Core.1c991c9fdfba671f", scope: null, file: !57, line: 44, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !90 = !DILocation(line: 44, column: 45, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 44, column: 38, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4", scope: null, file: !34, line: 156, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !7)
// CHECK:STDOUT: !95 = !DILocation(line: 157, column: 16, scope: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 157, column: 12, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 157, column: 5, scope: !92)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88d1103f417c6d4", scope: null, file: !34, line: 159, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !99)
// CHECK:STDOUT: !99 = !{!100}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !98, type: !7)
// CHECK:STDOUT: !101 = !DILocation(line: 160, column: 5, scope: !98)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.30ced37eb1e63547", scope: null, file: !34, line: 71, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !103)
// CHECK:STDOUT: !103 = !{!104}
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !102, type: !20)
// CHECK:STDOUT: !105 = !DILocation(line: 72, column: 12, scope: !102)
// CHECK:STDOUT: !106 = !DILocation(line: 72, column: 5, scope: !102)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "None", linkageName: "_CNone.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1", scope: null, file: !34, line: 116, type: !52, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !108 = !DILocation(line: 119, column: 5, scope: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 120, column: 5, scope: !107)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.3667eb502d1ddfa9:OptionalAs.c7a50af9bdd61b43.Core.5d35ad54738af8e4", scope: null, file: !34, line: 78, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !111)
// CHECK:STDOUT: !111 = !{!112}
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !110, type: !20)
// CHECK:STDOUT: !113 = !DILocation(line: 79, column: 29, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 79, column: 12, scope: !110)
// CHECK:STDOUT: !115 = !DILocation(line: 79, column: 5, scope: !110)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.30ced37eb1e63547", scope: null, file: !34, line: 30, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !117)
// CHECK:STDOUT: !117 = !{!118}
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !116, type: !20)
// CHECK:STDOUT: !119 = !DILocation(line: 31, column: 12, scope: !116)
// CHECK:STDOUT: !120 = !DILocation(line: 31, column: 5, scope: !116)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.a44fce96e16342e7:ImplicitAs.ad22d1bbc0605210.Core.90a4f5ca3e06badd", scope: null, file: !57, line: 40, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !122)
// CHECK:STDOUT: !122 = !{!123}
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !121, type: !20)
// CHECK:STDOUT: !124 = !DILocation(line: 40, column: 45, scope: !121)
// CHECK:STDOUT: !125 = !DILocation(line: 40, column: 38, scope: !121)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.b0246d3dd29c9210", scope: null, file: !34, line: 30, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !127)
// CHECK:STDOUT: !127 = !{!128}
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !126, type: !20)
// CHECK:STDOUT: !129 = !DILocation(line: 31, column: 12, scope: !126)
// CHECK:STDOUT: !130 = !DILocation(line: 31, column: 5, scope: !126)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.3e8267224c5dc9c2:OptionalStorage.Core.bfb441135f07cbe1", scope: null, file: !34, line: 122, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !132)
// CHECK:STDOUT: !132 = !{!133}
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !131, type: !20)
// CHECK:STDOUT: !134 = !DILocation(line: 128, column: 5, scope: !131)
// CHECK:STDOUT: !135 = !DILocation(line: 129, column: 5, scope: !131)
// CHECK:STDOUT: !136 = !DILocation(line: 130, column: 5, scope: !131)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.628184612c80324d", scope: null, file: !138, line: 62, type: !139, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !141)
// CHECK:STDOUT: !138 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !139 = !DISubroutineType(types: !140)
// CHECK:STDOUT: !140 = !{!20, !20}
// CHECK:STDOUT: !141 = !{!142}
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !137, type: !20)
// CHECK:STDOUT: !143 = !DILocation(line: 64, column: 17, scope: !137)
// CHECK:STDOUT: !144 = !DILocation(line: 64, column: 5, scope: !137)
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.3e8267224c5dc9c2:OptionalStorage.Core.5325ffa2b57e13bd", scope: null, file: !34, line: 122, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !146)
// CHECK:STDOUT: !146 = !{!147}
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !145, type: !20)
// CHECK:STDOUT: !148 = !DILocation(line: 128, column: 5, scope: !145)
// CHECK:STDOUT: !149 = !DILocation(line: 128, column: 28, scope: !145)
// CHECK:STDOUT: !150 = !DILocation(line: 129, column: 5, scope: !145)
// CHECK:STDOUT: !151 = !DILocation(line: 130, column: 5, scope: !145)
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !138, line: 57, type: !139, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !153)
// CHECK:STDOUT: !153 = !{!154}
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !152, type: !20)
// CHECK:STDOUT: !155 = !DILocation(line: 57, column: 36, scope: !152)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "Op", linkageName: "_COp.34e7a685d3648824:Copy.Core.64ccbb8e5d9a0b8e", scope: null, file: !157, line: 19, type: !45, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !158)
// CHECK:STDOUT: !157 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !158 = !{!159}
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !156, type: !20)
// CHECK:STDOUT: !160 = !DILocation(line: 19, column: 33, scope: !156)
