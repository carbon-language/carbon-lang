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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.3.temp = alloca [6 x i32], align 4, !dbg !7
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %.loc17_19.5.temp = alloca <{ i32, i1 }>, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_43.3.temp), !dbg !7
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 0, !dbg !7
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 1, !dbg !7
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 2, !dbg !7
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 3, !dbg !7
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 4, !dbg !7
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 5, !dbg !7
// CHECK:STDOUT:   %array_type.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.9932f4d85b117c73"(ptr @array.loc16_43.24), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %array_type.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.e163a239f9fa927c:Iterate.Core.9932f4d85b117c73"(ptr %.loc17_19.5.temp, ptr @array.loc16_43.24, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.ba7a4025e0666ad5(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.ba7a4025e0666ad5(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.04193412fded2aa2(ptr @array), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSubobjectDestroy.6f4dee545ed23f91:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.18fc07ffd08329ec:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.94d33dc7db8d438e:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ff0d3d801bd6a6ee:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ba5b124063947f4f:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.243175292981777b:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fd19f3506b5fdce5:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.9932f4d85b117c73"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT:   ret i32 0, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.e163a239f9fa927c:Iterate.Core.9932f4d85b117c73"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !53 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !59
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !59
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !60
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !61
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !62
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !62
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !63
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !63
// CHECK:STDOUT:   call void @_CSome.Optional.Core.ba7a4025e0666ad5(ptr %return, i32 %6), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.ba7a4025e0666ad5(ptr %return), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.ba7a4025e0666ad5(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %self), !dbg !74
// CHECK:STDOUT:   ret i1 %1, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.ba7a4025e0666ad5(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %self), !dbg !79
// CHECK:STDOUT:   ret i32 %1, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.243175292981777b:core.Destroy.Core"(ptr %self), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.04193412fded2aa2(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fd19f3506b5fdce5:core.Destroy.Core"(ptr %self), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.ba7a4025e0666ad5(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !103 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %return, i32 %value), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.ba7a4025e0666ad5(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !110 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %return), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %value) #0 !dbg !115 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !118
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !118
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !118
// CHECK:STDOUT:   ret i1 %2, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr %value) #0 !dbg !120 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !123
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !123
// CHECK:STDOUT:   ret i32 %1, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !125 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !131
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !132
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !132
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !132
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !133 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !136
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !137
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !137
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !139 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !140
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !140
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !142 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !147
// CHECK:STDOUT:   ret i32 %1, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !149 {
// CHECK:STDOUT:   ret i32 %self, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.e76e36aefa0c6287, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "iterate.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 26, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 7, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.18fc07ffd08329ec:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 17, column: 7, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.94d33dc7db8d438e:core.Destroy.Core", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !15)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 7, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ff0d3d801bd6a6ee:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 7, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ba5b124063947f4f:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !22)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 7, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.243175292981777b:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !22)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 7, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fd19f3506b5fdce5:core.Destroy.Core", scope: null, file: !1, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !22)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 26, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.e163a239f9fa927c:Iterate.Core.9932f4d85b117c73", scope: null, file: !47, line: 23, type: !48, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !47 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !48 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !49 = !{!15, !22}
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !46, type: !22)
// CHECK:STDOUT: !52 = !DILocation(line: 23, column: 38, scope: !46)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.e163a239f9fa927c:Iterate.Core.9932f4d85b117c73", scope: null, file: !47, line: 24, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !55 = !{!22, !22, !22}
// CHECK:STDOUT: !56 = !{!57, !58}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !53, type: !22)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !53, type: !22)
// CHECK:STDOUT: !59 = !DILocation(line: 25, column: 9, scope: !53)
// CHECK:STDOUT: !60 = !DILocation(line: 25, column: 8, scope: !53)
// CHECK:STDOUT: !61 = !DILocation(line: 26, column: 7, scope: !53)
// CHECK:STDOUT: !62 = !DILocation(line: 27, column: 36, scope: !53)
// CHECK:STDOUT: !63 = !DILocation(line: 27, column: 31, scope: !53)
// CHECK:STDOUT: !64 = !DILocation(line: 27, column: 14, scope: !53)
// CHECK:STDOUT: !65 = !DILocation(line: 27, column: 7, scope: !53)
// CHECK:STDOUT: !66 = !DILocation(line: 29, column: 14, scope: !53)
// CHECK:STDOUT: !67 = !DILocation(line: 29, column: 7, scope: !53)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.ba7a4025e0666ad5", scope: null, file: !69, line: 36, type: !70, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !72)
// CHECK:STDOUT: !69 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !70 = !DISubroutineType(types: !71)
// CHECK:STDOUT: !71 = !{!22, !22}
// CHECK:STDOUT: !72 = !{!73}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !68, type: !22)
// CHECK:STDOUT: !74 = !DILocation(line: 37, column: 12, scope: !68)
// CHECK:STDOUT: !75 = !DILocation(line: 37, column: 5, scope: !68)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.ba7a4025e0666ad5", scope: null, file: !69, line: 39, type: !48, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !77)
// CHECK:STDOUT: !77 = !{!78}
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !76, type: !22)
// CHECK:STDOUT: !79 = !DILocation(line: 40, column: 12, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 40, column: 5, scope: !76)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.e76e36aefa0c6287", scope: null, file: !82, line: 26, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !83)
// CHECK:STDOUT: !82 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !81, type: !22)
// CHECK:STDOUT: !85 = !DILocation(line: 28, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 26, column: 3, scope: !81)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !82, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !15)
// CHECK:STDOUT: !90 = !DILocation(line: 28, column: 5, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 26, column: 3, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.04193412fded2aa2", scope: null, file: !82, line: 26, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !22)
// CHECK:STDOUT: !95 = !DILocation(line: 28, column: 5, scope: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 26, column: 3, scope: !92)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !98, line: 396, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !99)
// CHECK:STDOUT: !98 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !99 = !{!100}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !97, type: !15)
// CHECK:STDOUT: !101 = !DILocation(line: 398, column: 5, scope: !97)
// CHECK:STDOUT: !102 = !DILocation(line: 396, column: 3, scope: !97)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.ba7a4025e0666ad5", scope: null, file: !69, line: 33, type: !104, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DISubroutineType(types: !105)
// CHECK:STDOUT: !105 = !{!22, !15}
// CHECK:STDOUT: !106 = !{!107}
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !103, type: !15)
// CHECK:STDOUT: !108 = !DILocation(line: 34, column: 12, scope: !103)
// CHECK:STDOUT: !109 = !DILocation(line: 34, column: 5, scope: !103)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.ba7a4025e0666ad5", scope: null, file: !69, line: 30, type: !111, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !111 = !DISubroutineType(types: !112)
// CHECK:STDOUT: !112 = !{!22}
// CHECK:STDOUT: !113 = !DILocation(line: 31, column: 12, scope: !110)
// CHECK:STDOUT: !114 = !DILocation(line: 31, column: 5, scope: !110)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec", scope: null, file: !69, line: 141, type: !70, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !116)
// CHECK:STDOUT: !116 = !{!117}
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !115, type: !22)
// CHECK:STDOUT: !118 = !DILocation(line: 142, column: 12, scope: !115)
// CHECK:STDOUT: !119 = !DILocation(line: 142, column: 5, scope: !115)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec", scope: null, file: !69, line: 144, type: !48, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !121)
// CHECK:STDOUT: !121 = !{!122}
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !120, type: !22)
// CHECK:STDOUT: !123 = !DILocation(line: 145, column: 12, scope: !120)
// CHECK:STDOUT: !124 = !DILocation(line: 145, column: 5, scope: !120)
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !98, line: 331, type: !126, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !128)
// CHECK:STDOUT: !126 = !DISubroutineType(types: !127)
// CHECK:STDOUT: !127 = !{null, !15, !15}
// CHECK:STDOUT: !128 = !{!129, !130}
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !125, type: !15)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 2, scope: !125, type: !15)
// CHECK:STDOUT: !131 = !DILocation(line: 0, scope: !125)
// CHECK:STDOUT: !132 = !DILocation(line: 331, column: 3, scope: !125)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec", scope: null, file: !69, line: 131, type: !104, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !134)
// CHECK:STDOUT: !134 = !{!135}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !133, type: !15)
// CHECK:STDOUT: !136 = !DILocation(line: 137, column: 5, scope: !133)
// CHECK:STDOUT: !137 = !DILocation(line: 138, column: 5, scope: !133)
// CHECK:STDOUT: !138 = !DILocation(line: 139, column: 5, scope: !133)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.46b5dffdde151aec", scope: null, file: !69, line: 125, type: !111, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !140 = !DILocation(line: 128, column: 5, scope: !139)
// CHECK:STDOUT: !141 = !DILocation(line: 129, column: 5, scope: !139)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !98, line: 64, type: !143, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !145)
// CHECK:STDOUT: !143 = !DISubroutineType(types: !144)
// CHECK:STDOUT: !144 = !{!15, !15}
// CHECK:STDOUT: !145 = !{!146}
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !142, type: !15)
// CHECK:STDOUT: !147 = !DILocation(line: 66, column: 17, scope: !142)
// CHECK:STDOUT: !148 = !DILocation(line: 66, column: 5, scope: !142)
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !98, line: 59, type: !143, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !150)
// CHECK:STDOUT: !150 = !{!151}
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !149, type: !15)
// CHECK:STDOUT: !152 = !DILocation(line: 59, column: 28, scope: !149)
// CHECK:STDOUT:
