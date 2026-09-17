// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/for/for.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/for/for.carbon

fn F();
fn G();
fn H();

fn For() {
  F();
  for (_: i32 in Core.Range(100)) {
    G();
  }
  H();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'for.carbon'
// CHECK:STDOUT: source_filename = "for.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFor.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_32.1.temp = alloca { i32, i32 }, align 4, !dbg !19
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !20
// CHECK:STDOUT:   %.loc19_33.1.temp = alloca <{ i32, i1 }>, align 4, !dbg !20
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_32.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_CRange.Core(ptr %.loc19_32.1.temp, i32 100), !dbg !19
// CHECK:STDOUT:   %IntRange.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc19_32.1.temp), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !20
// CHECK:STDOUT:   store i32 %IntRange.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_33.1.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %.loc19_33.1.temp, ptr %.loc19_32.1.temp, ptr %var), !dbg !20
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.d56e393be6e32080(ptr %.loc19_33.1.temp), !dbg !20
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.d56e393be6e32080(ptr %.loc19_33.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !27
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %.loc19_33.1.temp), !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %.loc19_33.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %var), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.5a42872629477478(ptr %.loc19_32.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CRange.Core(ptr sret({ i32, i32 }), i32)
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
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e933d5ff8212e171:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.c9184c281ed4335d:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   %start = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !67
// CHECK:STDOUT:   %1 = load i32, ptr %start, align 4, !dbg !67
// CHECK:STDOUT:   ret i32 %1, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !70 {
// CHECK:STDOUT:   %1 = alloca i32, align 4, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !77
// CHECK:STDOUT:   %2 = load i32, ptr %cursor, align 4, !dbg !79
// CHECK:STDOUT:   store i32 %2, ptr %1, align 4, !dbg !77
// CHECK:STDOUT:   %end = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !82
// CHECK:STDOUT:   %3 = load i32, ptr %end, align 4, !dbg !82
// CHECK:STDOUT:   %4 = load i32, ptr %1, align 4, !dbg !81
// CHECK:STDOUT:   %5 = icmp slt i32 %4, %3, !dbg !81
// CHECK:STDOUT:   br i1 %5, label %6, label %8, !dbg !83
// CHECK:STDOUT:
// CHECK:STDOUT: 6:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !86
// CHECK:STDOUT:   %7 = load i32, ptr %1, align 4, !dbg !87
// CHECK:STDOUT:   call void @_CSome.Optional.Core.d56e393be6e32080(ptr %return, i32 %7), !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %1), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.d56e393be6e32080(ptr %return), !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %1), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.d56e393be6e32080(ptr %self) #0 !dbg !96 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %self), !dbg !102
// CHECK:STDOUT:   ret i1 %1, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.d56e393be6e32080(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %self), !dbg !111
// CHECK:STDOUT:   ret i32 %1, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.31a8c47192dc2066(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f5c18a58c263f72b:core.Destroy.Core"(ptr %self), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !122 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.5a42872629477478(ptr %self) #0 !dbg !129 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.c9184c281ed4335d:core.Destroy.Core"(ptr %self), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CInclusiveRange.Core(ptr sret({ i32, i32 }), i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !137 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !140
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.d56e393be6e32080(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !145 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %return, i32 %value), !dbg !150
// CHECK:STDOUT:   ret void, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.d56e393be6e32080(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !153 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %return), !dbg !156
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %value) #0 !dbg !158 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !163
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !163
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !163
// CHECK:STDOUT:   ret i1 %2, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr %value) #0 !dbg !166 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !171
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !171
// CHECK:STDOUT:   ret i32 %1, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !174 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !177
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !178
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !178
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !178
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !182 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !191
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !191
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !193
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !193
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !196 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !203
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !203
// CHECK:STDOUT:   ret void, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !205 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !210
// CHECK:STDOUT:   ret i32 %1, !dbg !212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !214 {
// CHECK:STDOUT:   ret i32 %self, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.31a8c47192dc2066, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 2, 1, 0 }
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
// CHECK:STDOUT: !4 = !DIFile(filename: "for.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{!9, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !9}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "For", linkageName: "_CFor.Main", scope: null, file: !4, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 3, scope: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 19, column: 18, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 19, column: 7, scope: !16)
// CHECK:STDOUT: !21 = !{!6, !9}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!9, !9, !9}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!9, !9}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 20, column: 5, scope: !16)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 3, scope: !16)
// CHECK:STDOUT: !29 = !DILocation(line: 22, column: 3, scope: !16)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 1, scope: !16)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !4, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !9)
// CHECK:STDOUT: !33 = !DILocation(line: 19, column: 7, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.32ec81fcfb7ba5dc:core.Destroy.Core", scope: null, file: !4, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !6)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 7, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c3be8338012ae399:core.Destroy.Core", scope: null, file: !4, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !9)
// CHECK:STDOUT: !41 = !DILocation(line: 19, column: 7, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c800cdf671d92dcc:core.Destroy.Core", scope: null, file: !4, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !9)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 7, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f5c18a58c263f72b:core.Destroy.Core", scope: null, file: !4, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !9)
// CHECK:STDOUT: !49 = !DILocation(line: 19, column: 7, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e933d5ff8212e171:core.Destroy.Core", scope: null, file: !4, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !9)
// CHECK:STDOUT: !53 = !DILocation(line: 19, column: 18, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.c9184c281ed4335d:core.Destroy.Core", scope: null, file: !4, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !9)
// CHECK:STDOUT: !57 = !DILocation(line: 19, column: 18, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/range.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !61, line: 19, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !69)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !9)
// CHECK:STDOUT: !67 = !DILocation(line: 19, column: 43, scope: !62)
// CHECK:STDOUT: !68 = !DILocation(line: 19, column: 36, scope: !62)
// CHECK:STDOUT: !69 = !{!63}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.IntRange.9452f4c51951679b.Core:Iterate.Core.be1e879c1ad406d8", scope: null, file: !61, line: 20, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !94)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !9)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 2, scope: !70, type: !9)
// CHECK:STDOUT: !77 = !DILocation(line: 21, column: 7, scope: !70)
// CHECK:STDOUT: !79 = !DILocation(line: 21, column: 27, scope: !70)
// CHECK:STDOUT: !81 = !DILocation(line: 22, column: 11, scope: !70)
// CHECK:STDOUT: !82 = !DILocation(line: 22, column: 19, scope: !70)
// CHECK:STDOUT: !83 = !DILocation(line: 22, column: 10, scope: !70)
// CHECK:STDOUT: !86 = !DILocation(line: 23, column: 9, scope: !70)
// CHECK:STDOUT: !87 = !DILocation(line: 24, column: 38, scope: !70)
// CHECK:STDOUT: !88 = !DILocation(line: 24, column: 16, scope: !70)
// CHECK:STDOUT: !89 = !DILocation(line: 24, column: 9, scope: !70)
// CHECK:STDOUT: !90 = !DILocation(line: 26, column: 16, scope: !70)
// CHECK:STDOUT: !91 = !{!9}
// CHECK:STDOUT: !92 = !DISubroutineType(types: !91)
// CHECK:STDOUT: !93 = !DILocation(line: 26, column: 9, scope: !70)
// CHECK:STDOUT: !94 = !{!71, !72}
// CHECK:STDOUT: !95 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.d56e393be6e32080", scope: null, file: !95, line: 36, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !9)
// CHECK:STDOUT: !102 = !DILocation(line: 37, column: 12, scope: !96)
// CHECK:STDOUT: !103 = !DILocation(line: 37, column: 5, scope: !96)
// CHECK:STDOUT: !104 = !{!97}
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.d56e393be6e32080", scope: null, file: !95, line: 39, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !113)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !9)
// CHECK:STDOUT: !111 = !DILocation(line: 40, column: 12, scope: !105)
// CHECK:STDOUT: !112 = !DILocation(line: 40, column: 5, scope: !105)
// CHECK:STDOUT: !113 = !{!106}
// CHECK:STDOUT: !114 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.31a8c47192dc2066", scope: null, file: !114, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !121)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !9)
// CHECK:STDOUT: !119 = !DILocation(line: 28, column: 5, scope: !115)
// CHECK:STDOUT: !120 = !DILocation(line: 26, column: 3, scope: !115)
// CHECK:STDOUT: !121 = !{!116}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !114, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !128)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !6)
// CHECK:STDOUT: !126 = !DILocation(line: 28, column: 5, scope: !122)
// CHECK:STDOUT: !127 = !DILocation(line: 26, column: 3, scope: !122)
// CHECK:STDOUT: !128 = !{!123}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.5a42872629477478", scope: null, file: !114, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !135)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !9)
// CHECK:STDOUT: !133 = !DILocation(line: 28, column: 5, scope: !129)
// CHECK:STDOUT: !134 = !DILocation(line: 26, column: 3, scope: !129)
// CHECK:STDOUT: !135 = !{!130}
// CHECK:STDOUT: !136 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !136, line: 396, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !144)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !6)
// CHECK:STDOUT: !140 = !DILocation(line: 398, column: 5, scope: !137)
// CHECK:STDOUT: !141 = !{null, !6, !6}
// CHECK:STDOUT: !142 = !DISubroutineType(types: !141)
// CHECK:STDOUT: !143 = !DILocation(line: 396, column: 3, scope: !137)
// CHECK:STDOUT: !144 = !{!138}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.d56e393be6e32080", scope: null, file: !95, line: 33, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !152)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !6)
// CHECK:STDOUT: !150 = !DILocation(line: 34, column: 12, scope: !145)
// CHECK:STDOUT: !151 = !DILocation(line: 34, column: 5, scope: !145)
// CHECK:STDOUT: !152 = !{!146}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.d56e393be6e32080", scope: null, file: !95, line: 30, type: !92, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !156 = !DILocation(line: 31, column: 12, scope: !153)
// CHECK:STDOUT: !157 = !DILocation(line: 31, column: 5, scope: !153)
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !95, line: 141, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !165)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !9)
// CHECK:STDOUT: !163 = !DILocation(line: 142, column: 12, scope: !158)
// CHECK:STDOUT: !164 = !DILocation(line: 142, column: 5, scope: !158)
// CHECK:STDOUT: !165 = !{!159}
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !95, line: 144, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !173)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !166, type: !9)
// CHECK:STDOUT: !171 = !DILocation(line: 145, column: 12, scope: !166)
// CHECK:STDOUT: !172 = !DILocation(line: 145, column: 5, scope: !166)
// CHECK:STDOUT: !173 = !{!167}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !136, line: 331, type: !142, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !181)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !6)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 2, scope: !174, type: !6)
// CHECK:STDOUT: !177 = !DILocation(line: 0, scope: !174)
// CHECK:STDOUT: !178 = !DILocation(line: 331, column: 3, scope: !174)
// CHECK:STDOUT: !179 = !{!6, !6}
// CHECK:STDOUT: !180 = !DISubroutineType(types: !179)
// CHECK:STDOUT: !181 = !{!175, !176}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !95, line: 131, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !195)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !6)
// CHECK:STDOUT: !191 = !DILocation(line: 137, column: 5, scope: !182)
// CHECK:STDOUT: !193 = !DILocation(line: 138, column: 5, scope: !182)
// CHECK:STDOUT: !194 = !DILocation(line: 139, column: 5, scope: !182)
// CHECK:STDOUT: !195 = !{!183}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.5216072513bfc692", scope: null, file: !95, line: 125, type: !92, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !203 = !DILocation(line: 128, column: 5, scope: !196)
// CHECK:STDOUT: !204 = !DILocation(line: 129, column: 5, scope: !196)
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !136, line: 64, type: !180, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !213)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !6)
// CHECK:STDOUT: !210 = !DILocation(line: 66, column: 17, scope: !205)
// CHECK:STDOUT: !212 = !DILocation(line: 66, column: 5, scope: !205)
// CHECK:STDOUT: !213 = !{!206}
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !136, line: 59, type: !180, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !221)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !6)
// CHECK:STDOUT: !220 = !DILocation(line: 59, column: 28, scope: !214)
// CHECK:STDOUT: !221 = !{!215}
// CHECK:STDOUT:
