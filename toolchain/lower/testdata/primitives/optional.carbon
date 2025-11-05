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
  // TODO: The code generated for these conversions is wrong: we initialize a
  // temporary rather than the local variable.
  var oa: Core.Optional(i32) = a;
  var coa: const Core.Optional(i32) = a;
  var oca: Core.Optional(const i32) = a;
  var coca: const Core.Optional(const i32) = a;
  var ob: Core.Optional(i32) = b;
  var cob: const Core.Optional(i32) = b;
  var ocb: Core.Optional(const i32) = b;
  var cocb: const Core.Optional(const i32) = b;
}

// CHECK:STDOUT: ; ModuleID = 'optional.carbon'
// CHECK:STDOUT: source_filename = "optional.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret({ i32, i1 }) %return, ptr %o) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_20.1.temp = alloca { i32, i1 }, align 8, !dbg !7
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.217efae529e578bc(ptr %o), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.217efae529e578bc(ptr %o), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_20.1.temp), !dbg !7
// CHECK:STDOUT:   %.loc18_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !11
// CHECK:STDOUT:   call void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.dbc2dba5c3cff15b"(ptr %.loc18_20.1.temp, i32 %.loc18_12.2), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.f16306f3d30b9711(ptr %return), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddOrRemoveConst.Main(i32 %a, i32 %b) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %oa.var = alloca { i32, i1 }, align 8, !dbg !15
// CHECK:STDOUT:   %.loc26_3.1.temp = alloca { i32, i1 }, align 8, !dbg !15
// CHECK:STDOUT:   %coa.var = alloca { i32, i1 }, align 8, !dbg !16
// CHECK:STDOUT:   %.loc27_3.1.temp = alloca { i32, i1 }, align 8, !dbg !16
// CHECK:STDOUT:   %oca.var = alloca { i32, i1 }, align 8, !dbg !17
// CHECK:STDOUT:   %.loc28_3.1.temp = alloca { i32, i1 }, align 8, !dbg !17
// CHECK:STDOUT:   %coca.var = alloca { i32, i1 }, align 8, !dbg !18
// CHECK:STDOUT:   %.loc29_3.1.temp = alloca { i32, i1 }, align 8, !dbg !18
// CHECK:STDOUT:   %ob.var = alloca { i32, i1 }, align 8, !dbg !19
// CHECK:STDOUT:   %.loc30_3.1.temp = alloca { i32, i1 }, align 8, !dbg !19
// CHECK:STDOUT:   %cob.var = alloca { i32, i1 }, align 8, !dbg !20
// CHECK:STDOUT:   %.loc31_3.1.temp = alloca { i32, i1 }, align 8, !dbg !20
// CHECK:STDOUT:   %ocb.var = alloca { i32, i1 }, align 8, !dbg !21
// CHECK:STDOUT:   %.loc32_3.1.temp = alloca { i32, i1 }, align 8, !dbg !21
// CHECK:STDOUT:   %cocb.var = alloca { i32, i1 }, align 8, !dbg !22
// CHECK:STDOUT:   %.loc33_3.1.temp = alloca { i32, i1 }, align 8, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %oa.var), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_3.1.temp), !dbg !15
// CHECK:STDOUT:   call void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.dbc2dba5c3cff15b"(ptr %.loc26_3.1.temp, i32 %a), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %coa.var), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_3.1.temp), !dbg !16
// CHECK:STDOUT:   call void @"_CConvert.ab0f2d07242360d7:ImplicitAs.Core.7bf331f068e9c2df"(ptr %.loc27_3.1.temp, i32 %a), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %oca.var), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_3.1.temp), !dbg !17
// CHECK:STDOUT:   call void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.aa7929136b20b999"(ptr %.loc28_3.1.temp, i32 %a), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %coca.var), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_3.1.temp), !dbg !18
// CHECK:STDOUT:   call void @"_CConvert.ab0f2d07242360d7:ImplicitAs.Core.988e6ea8b7110d27"(ptr %.loc29_3.1.temp, i32 %a), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ob.var), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc30_3.1.temp), !dbg !19
// CHECK:STDOUT:   call void @"_CConvert.4a889a7251a91071:ImplicitAs.Core.7bf331f068e9c2df"(ptr %.loc30_3.1.temp, i32 %b), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %cob.var), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc31_3.1.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CConvert.4a889a7251a91071:ImplicitAs.Core.af35f070baa4994b"(ptr %.loc31_3.1.temp, i32 %b), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ocb.var), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_3.1.temp), !dbg !21
// CHECK:STDOUT:   call void @"_CConvert.4a889a7251a91071:ImplicitAs.Core.988e6ea8b7110d27"(ptr %.loc32_3.1.temp, i32 %b), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %cocb.var), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_3.1.temp), !dbg !22
// CHECK:STDOUT:   call void @"_CConvert.4a889a7251a91071:ImplicitAs.Core.42efcbde6ff4083b"(ptr %.loc33_3.1.temp, i32 %b), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.217efae529e578bc(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT:   %1 = load ptr, ptr %self, align 8, !dbg !26
// CHECK:STDOUT:   %2 = icmp eq ptr %1, null, !dbg !26
// CHECK:STDOUT:   ret i1 %2, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.217efae529e578bc(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.4f0b5cc38af595d2:OptionalStorage.Core.b88d1103f417c6d4"(ptr %self), !dbg !29
// CHECK:STDOUT:   ret ptr %1, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.dbc2dba5c3cff15b"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !31 {
// CHECK:STDOUT:   call void @"_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.f16306f3d30b9711"(ptr %return, i32 %self), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.f16306f3d30b9711(ptr sret({ i32, i1 }) %return) #0 !dbg !34 {
// CHECK:STDOUT:   call void @"_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %return), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ab0f2d07242360d7:ImplicitAs.Core.7bf331f068e9c2df"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !37 {
// CHECK:STDOUT:   %temp = alloca { i32, i1 }, align 8, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !39
// CHECK:STDOUT:   call void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.dbc2dba5c3cff15b"(ptr %temp, i32 %self), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.aa7929136b20b999"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !41 {
// CHECK:STDOUT:   call void @"_CConvert.9d11af2db03b082d:OptionalAs.Core.f6268c15480add99"(ptr %return, i32 %self), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.ab0f2d07242360d7:ImplicitAs.Core.988e6ea8b7110d27"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !44 {
// CHECK:STDOUT:   %temp = alloca { i32, i1 }, align 8, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !45
// CHECK:STDOUT:   call void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.aa7929136b20b999"(ptr %temp, i32 %self), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.4a889a7251a91071:ImplicitAs.Core.7bf331f068e9c2df"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !47 {
// CHECK:STDOUT:   call void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.dbc2dba5c3cff15b"(ptr %return, i32 %self), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.4a889a7251a91071:ImplicitAs.Core.af35f070baa4994b"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !50 {
// CHECK:STDOUT:   call void @"_CConvert.ab0f2d07242360d7:ImplicitAs.Core.7bf331f068e9c2df"(ptr %return, i32 %self), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.4a889a7251a91071:ImplicitAs.Core.988e6ea8b7110d27"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !53 {
// CHECK:STDOUT:   call void @"_CConvert.f82b7327366b1770:ImplicitAs.Core.aa7929136b20b999"(ptr %return, i32 %self), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.4a889a7251a91071:ImplicitAs.Core.42efcbde6ff4083b"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !56 {
// CHECK:STDOUT:   call void @"_CConvert.ab0f2d07242360d7:ImplicitAs.Core.988e6ea8b7110d27"(ptr %return, i32 %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.4f0b5cc38af595d2:OptionalStorage.Core.b88d1103f417c6d4"(ptr %value) #0 !dbg !59 {
// CHECK:STDOUT:   %1 = load ptr, ptr %value, align 8, !dbg !60
// CHECK:STDOUT:   ret ptr %1, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.f16306f3d30b9711"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !62 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.f16306f3d30b9711(ptr %return, i32 %self), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr sret({ i32, i1 }) %return) #0 !dbg !65 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !66
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.9d11af2db03b082d:OptionalAs.Core.f6268c15480add99"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !68 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !69
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !69
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.ab0f2d07242360d7:ImplicitAs.Core.f9a27c9a1900a4ff"(i32 %self), !dbg !69
// CHECK:STDOUT:   store i32 %1, ptr %temp, align 4, !dbg !69
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4, !dbg !69
// CHECK:STDOUT:   call void @_CSome.Optional.Core.9427236976e7a9ec(ptr %return, i32 %2), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.f16306f3d30b9711(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !72 {
// CHECK:STDOUT:   call void @"_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %return, i32 %value), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.ab0f2d07242360d7:ImplicitAs.Core.f9a27c9a1900a4ff"(i32 %self) #0 !dbg !75 {
// CHECK:STDOUT:   %temp = alloca i32, align 4, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !76
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899"(i32 %self), !dbg !76
// CHECK:STDOUT:   ret i32 %1, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.9427236976e7a9ec(ptr sret({ i32, i1 }) %return, i32 %value) #0 !dbg !78 {
// CHECK:STDOUT:   call void @"_CSome.225258f1a45e9386:OptionalStorage.Core.ae62e7630b150053"(ptr %return, i32 %value), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !81 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !82
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !82
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !83
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !83
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899"(i32 %self) #0 !dbg !85 {
// CHECK:STDOUT:   ret i32 %self, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.225258f1a45e9386:OptionalStorage.Core.ae62e7630b150053"(ptr sret({ i32, i1 }) %return, i32 %self) #0 !dbg !87 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !88
// CHECK:STDOUT:   %1 = call i32 @"_COp.14f25726eade61db:Copy.Core.5450dc8e8b8e0899"(i32 %self), !dbg !89
// CHECK:STDOUT:   store i32 %1, ptr %value, align 4, !dbg !88
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !90
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.14f25726eade61db:Copy.Core.5450dc8e8b8e0899"(i32 %self) #0 !dbg !92 {
// CHECK:STDOUT:   ret i32 %self, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.f82b7327366b1770:ImplicitAs.Core.dbc2dba5c3cff15b", { 0, 1, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "optional.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 17, column: 6, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 13, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 12, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 20, column: 10, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "AddOrRemoveConst", linkageName: "_CAddOrRemoveConst.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 26, column: 3, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 27, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 28, column: 3, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 29, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 30, column: 3, scope: !14)
// CHECK:STDOUT: !20 = !DILocation(line: 31, column: 3, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 32, column: 3, scope: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 33, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 23, column: 1, scope: !14)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.217efae529e578bc", scope: null, file: !25, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !26 = !DILocation(line: 33, column: 12, scope: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 33, column: 5, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.217efae529e578bc", scope: null, file: !25, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DILocation(line: 36, column: 12, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 36, column: 5, scope: !28)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.f82b7327366b1770:ImplicitAs.Core.dbc2dba5c3cff15b", scope: null, file: !25, line: 82, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !32 = !DILocation(line: 83, column: 12, scope: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 83, column: 5, scope: !31)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.f16306f3d30b9711", scope: null, file: !25, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !35 = !DILocation(line: 27, column: 12, scope: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 27, column: 5, scope: !34)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ab0f2d07242360d7:ImplicitAs.Core.7bf331f068e9c2df", scope: null, file: !38, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !38 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !39 = !DILocation(line: 28, column: 45, scope: !37)
// CHECK:STDOUT: !40 = !DILocation(line: 28, column: 38, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.f82b7327366b1770:ImplicitAs.Core.aa7929136b20b999", scope: null, file: !25, line: 82, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !42 = !DILocation(line: 83, column: 12, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 83, column: 5, scope: !41)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ab0f2d07242360d7:ImplicitAs.Core.988e6ea8b7110d27", scope: null, file: !38, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !45 = !DILocation(line: 28, column: 45, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 28, column: 38, scope: !44)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.4a889a7251a91071:ImplicitAs.Core.7bf331f068e9c2df", scope: null, file: !38, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !48 = !DILocation(line: 32, column: 45, scope: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 32, column: 38, scope: !47)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.4a889a7251a91071:ImplicitAs.Core.af35f070baa4994b", scope: null, file: !38, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !51 = !DILocation(line: 32, column: 45, scope: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 32, column: 38, scope: !50)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.4a889a7251a91071:ImplicitAs.Core.988e6ea8b7110d27", scope: null, file: !38, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !54 = !DILocation(line: 32, column: 45, scope: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 32, column: 38, scope: !53)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.4a889a7251a91071:ImplicitAs.Core.42efcbde6ff4083b", scope: null, file: !38, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !57 = !DILocation(line: 32, column: 45, scope: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 32, column: 38, scope: !56)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.4f0b5cc38af595d2:OptionalStorage.Core.b88d1103f417c6d4", scope: null, file: !25, line: 131, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !60 = !DILocation(line: 132, column: 12, scope: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 132, column: 5, scope: !59)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.f16306f3d30b9711", scope: null, file: !25, line: 68, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !63 = !DILocation(line: 69, column: 12, scope: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 69, column: 5, scope: !62)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "None", linkageName: "_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899", scope: null, file: !25, line: 99, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !66 = !DILocation(line: 101, column: 5, scope: !65)
// CHECK:STDOUT: !67 = !DILocation(line: 102, column: 5, scope: !65)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9d11af2db03b082d:OptionalAs.Core.f6268c15480add99", scope: null, file: !25, line: 75, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !69 = !DILocation(line: 76, column: 29, scope: !68)
// CHECK:STDOUT: !70 = !DILocation(line: 76, column: 12, scope: !68)
// CHECK:STDOUT: !71 = !DILocation(line: 76, column: 5, scope: !68)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.f16306f3d30b9711", scope: null, file: !25, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !73 = !DILocation(line: 30, column: 12, scope: !72)
// CHECK:STDOUT: !74 = !DILocation(line: 30, column: 5, scope: !72)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ab0f2d07242360d7:ImplicitAs.Core.f9a27c9a1900a4ff", scope: null, file: !38, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !76 = !DILocation(line: 28, column: 45, scope: !75)
// CHECK:STDOUT: !77 = !DILocation(line: 28, column: 38, scope: !75)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.9427236976e7a9ec", scope: null, file: !25, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !79 = !DILocation(line: 30, column: 12, scope: !78)
// CHECK:STDOUT: !80 = !DILocation(line: 30, column: 5, scope: !78)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899", scope: null, file: !25, line: 104, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !82 = !DILocation(line: 108, column: 5, scope: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 109, column: 5, scope: !81)
// CHECK:STDOUT: !84 = !DILocation(line: 110, column: 5, scope: !81)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899", scope: null, file: !38, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !86 = !DILocation(line: 24, column: 38, scope: !85)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.225258f1a45e9386:OptionalStorage.Core.ae62e7630b150053", scope: null, file: !25, line: 104, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !88 = !DILocation(line: 108, column: 5, scope: !87)
// CHECK:STDOUT: !89 = !DILocation(line: 108, column: 28, scope: !87)
// CHECK:STDOUT: !90 = !DILocation(line: 109, column: 5, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 110, column: 5, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Op", linkageName: "_COp.14f25726eade61db:Copy.Core.5450dc8e8b8e0899", scope: null, file: !93, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !93 = !DIFile(filename: "{{.*}}/prelude/copy.carbon", directory: "")
// CHECK:STDOUT: !94 = !DILocation(line: 19, column: 33, scope: !92)
