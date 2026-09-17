// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/iterate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/iterate.carbon

fn G(n: i32);

fn F() {
  let a: array(i32, 6) = (1, 2, 3, 4, 5, 6);
  for (n: i32 in a) {
    G(n);
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'iterate.carbon'
// CHECK:STDOUT: source_filename = "iterate.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [6 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6]
// CHECK:STDOUT: @array.loc16_43.24 = internal constant [6 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.3.temp = alloca [6 x i32], align 4, !dbg !15
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !20
// CHECK:STDOUT:   %.loc17_19.5.temp = alloca <{ i32, i1 }>, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_43.3.temp), !dbg !15
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 0, !dbg !15
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 1, !dbg !15
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 2, !dbg !15
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 3, !dbg !15
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 4, !dbg !15
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 5, !dbg !15
// CHECK:STDOUT:   %array_type.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.4f783373a4c5a0db"(ptr @array.loc16_43.24), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !20
// CHECK:STDOUT:   store i32 %array_type.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CNext.e163a239f9fa927c:Iterate.Core.4f783373a4c5a0db"(ptr %.loc17_19.5.temp, ptr @array.loc16_43.24, ptr %var), !dbg !20
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.d56e393be6e32080(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.d56e393be6e32080(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %var), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.04193412fded2aa2(ptr @array), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.32ec81fcfb7ba5dc:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c3be8338012ae399:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c800cdf671d92dcc:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f5c18a58c263f72b:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1366a4d5330239bd:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.4f783373a4c5a0db"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT:   ret i32 0, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.e163a239f9fa927c:Iterate.Core.4f783373a4c5a0db"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !63 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !71
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !71
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !72
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !75
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !78
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !78
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !76
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !76
// CHECK:STDOUT:   call void @_CSome.Optional.Core.d56e393be6e32080(ptr %return, i32 %6), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.d56e393be6e32080(ptr %return), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.d56e393be6e32080(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %self), !dbg !95
// CHECK:STDOUT:   ret i1 %1, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.d56e393be6e32080(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %self), !dbg !104
// CHECK:STDOUT:   ret i32 %1, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f5c18a58c263f72b:core.Destroy.Core"(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.04193412fded2aa2(ptr %self) #0 !dbg !122 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1366a4d5330239bd:core.Destroy.Core"(ptr %self), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !130 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.d56e393be6e32080(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !138 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %return, i32 %value), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.d56e393be6e32080(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !146 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %return), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %value) #0 !dbg !151 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !156
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !156
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !156
// CHECK:STDOUT:   ret i1 %2, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %value) #0 !dbg !159 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !164
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !164
// CHECK:STDOUT:   ret i32 %1, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !167 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !170
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !171
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !171
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !171
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !175 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !184
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !184
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !186
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !186
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !189 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !196
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !196
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !198 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !203
// CHECK:STDOUT:   ret i32 %1, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !207 {
// CHECK:STDOUT:   ret i32 %self, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.31a8c47192dc2066, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "iterate.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !12 = !{null, !11}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !4, line: 15, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 26, scope: !14)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 7, scope: !14)
// CHECK:STDOUT: !21 = !{!6, !11}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!11, !11, !11}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!11, !11}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 18, column: 5, scope: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 3, scope: !14)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 1, scope: !14)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !11)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 7, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.32ec81fcfb7ba5dc:core.Destroy.Core", scope: null, file: !4, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !6)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 7, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3be8338012ae399:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !11)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 7, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c800cdf671d92dcc:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !11)
// CHECK:STDOUT: !45 = !DILocation(line: 17, column: 7, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f5c18a58c263f72b:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !11)
// CHECK:STDOUT: !49 = !DILocation(line: 17, column: 7, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1366a4d5330239bd:core.Destroy.Core", scope: null, file: !4, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !11)
// CHECK:STDOUT: !53 = !DILocation(line: 16, column: 26, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.e163a239f9fa927c:Iterate.Core.4f783373a4c5a0db", scope: null, file: !55, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !11)
// CHECK:STDOUT: !61 = !DILocation(line: 23, column: 38, scope: !56)
// CHECK:STDOUT: !62 = !{!57}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.e163a239f9fa927c:Iterate.Core.4f783373a4c5a0db", scope: null, file: !55, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !11)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 2, scope: !63, type: !11)
// CHECK:STDOUT: !71 = !DILocation(line: 25, column: 9, scope: !63)
// CHECK:STDOUT: !72 = !DILocation(line: 25, column: 8, scope: !63)
// CHECK:STDOUT: !75 = !DILocation(line: 26, column: 7, scope: !63)
// CHECK:STDOUT: !76 = !DILocation(line: 27, column: 31, scope: !63)
// CHECK:STDOUT: !78 = !DILocation(line: 27, column: 36, scope: !63)
// CHECK:STDOUT: !79 = !DILocation(line: 27, column: 14, scope: !63)
// CHECK:STDOUT: !80 = !{!11, !6}
// CHECK:STDOUT: !81 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !82 = !DILocation(line: 27, column: 7, scope: !63)
// CHECK:STDOUT: !83 = !DILocation(line: 29, column: 14, scope: !63)
// CHECK:STDOUT: !84 = !{!11}
// CHECK:STDOUT: !85 = !DISubroutineType(types: !84)
// CHECK:STDOUT: !86 = !DILocation(line: 29, column: 7, scope: !63)
// CHECK:STDOUT: !87 = !{!64, !65}
// CHECK:STDOUT: !88 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.d56e393be6e32080", scope: null, file: !88, line: 36, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !97)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !11)
// CHECK:STDOUT: !95 = !DILocation(line: 37, column: 12, scope: !89)
// CHECK:STDOUT: !96 = !DILocation(line: 37, column: 5, scope: !89)
// CHECK:STDOUT: !97 = !{!90}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.d56e393be6e32080", scope: null, file: !88, line: 39, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !106)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !11)
// CHECK:STDOUT: !104 = !DILocation(line: 40, column: 12, scope: !98)
// CHECK:STDOUT: !105 = !DILocation(line: 40, column: 5, scope: !98)
// CHECK:STDOUT: !106 = !{!99}
// CHECK:STDOUT: !107 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.31a8c47192dc2066", scope: null, file: !107, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !114)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !11)
// CHECK:STDOUT: !112 = !DILocation(line: 28, column: 5, scope: !108)
// CHECK:STDOUT: !113 = !DILocation(line: 26, column: 3, scope: !108)
// CHECK:STDOUT: !114 = !{!109}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !107, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !121)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !6)
// CHECK:STDOUT: !119 = !DILocation(line: 28, column: 5, scope: !115)
// CHECK:STDOUT: !120 = !DILocation(line: 26, column: 3, scope: !115)
// CHECK:STDOUT: !121 = !{!116}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.04193412fded2aa2", scope: null, file: !107, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !128)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !11)
// CHECK:STDOUT: !126 = !DILocation(line: 28, column: 5, scope: !122)
// CHECK:STDOUT: !127 = !DILocation(line: 26, column: 3, scope: !122)
// CHECK:STDOUT: !128 = !{!123}
// CHECK:STDOUT: !129 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !129, line: 396, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !137)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !6)
// CHECK:STDOUT: !133 = !DILocation(line: 398, column: 5, scope: !130)
// CHECK:STDOUT: !134 = !{null, !6, !6}
// CHECK:STDOUT: !135 = !DISubroutineType(types: !134)
// CHECK:STDOUT: !136 = !DILocation(line: 396, column: 3, scope: !130)
// CHECK:STDOUT: !137 = !{!131}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d56e393be6e32080", scope: null, file: !88, line: 33, type: !81, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !145)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !6)
// CHECK:STDOUT: !143 = !DILocation(line: 34, column: 12, scope: !138)
// CHECK:STDOUT: !144 = !DILocation(line: 34, column: 5, scope: !138)
// CHECK:STDOUT: !145 = !{!139}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.d56e393be6e32080", scope: null, file: !88, line: 30, type: !85, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !149 = !DILocation(line: 31, column: 12, scope: !146)
// CHECK:STDOUT: !150 = !DILocation(line: 31, column: 5, scope: !146)
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !88, line: 141, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !158)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !11)
// CHECK:STDOUT: !156 = !DILocation(line: 142, column: 12, scope: !151)
// CHECK:STDOUT: !157 = !DILocation(line: 142, column: 5, scope: !151)
// CHECK:STDOUT: !158 = !{!152}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !88, line: 144, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !166)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !11)
// CHECK:STDOUT: !164 = !DILocation(line: 145, column: 12, scope: !159)
// CHECK:STDOUT: !165 = !DILocation(line: 145, column: 5, scope: !159)
// CHECK:STDOUT: !166 = !{!160}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !129, line: 331, type: !135, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !174)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !6)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 2, scope: !167, type: !6)
// CHECK:STDOUT: !170 = !DILocation(line: 0, scope: !167)
// CHECK:STDOUT: !171 = !DILocation(line: 331, column: 3, scope: !167)
// CHECK:STDOUT: !172 = !{!6, !6}
// CHECK:STDOUT: !173 = !DISubroutineType(types: !172)
// CHECK:STDOUT: !174 = !{!168, !169}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !88, line: 131, type: !81, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !188)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !6)
// CHECK:STDOUT: !184 = !DILocation(line: 137, column: 5, scope: !175)
// CHECK:STDOUT: !186 = !DILocation(line: 138, column: 5, scope: !175)
// CHECK:STDOUT: !187 = !DILocation(line: 139, column: 5, scope: !175)
// CHECK:STDOUT: !188 = !{!176}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !88, line: 125, type: !85, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !196 = !DILocation(line: 128, column: 5, scope: !189)
// CHECK:STDOUT: !197 = !DILocation(line: 129, column: 5, scope: !189)
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !129, line: 64, type: !173, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !206)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !6)
// CHECK:STDOUT: !203 = !DILocation(line: 66, column: 17, scope: !198)
// CHECK:STDOUT: !205 = !DILocation(line: 66, column: 5, scope: !198)
// CHECK:STDOUT: !206 = !{!199}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !129, line: 59, type: !173, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !214)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !6)
// CHECK:STDOUT: !213 = !DILocation(line: 59, column: 28, scope: !207)
// CHECK:STDOUT: !214 = !{!208}
// CHECK:STDOUT:
