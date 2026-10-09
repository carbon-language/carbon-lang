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
// CHECK:STDOUT:   %Array.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.Array.e6a7e802f5bda6f1.Core:Iterate.Core.cb65b0323a88fe77"(ptr @array.loc16_43.24), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !20
// CHECK:STDOUT:   store i32 %Array.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CNext.Array.e6a7e802f5bda6f1.Core:Iterate.Core.cb65b0323a88fe77"(ptr %.loc17_19.5.temp, ptr @array.loc16_43.24, ptr %var), !dbg !20
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.8758b884e7707351(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.8758b884e7707351(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !28
// CHECK:STDOUT:   call void @"_CSelfDestruct.dfe4cc77d44e03b0:core.Destroy.Core"(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   br label %for.next, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_CSelfDestruct.dfe4cc77d44e03b0:core.Destroy.Core"(ptr %.loc17_19.5.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %var), !dbg !20
// CHECK:STDOUT:   call void @"_CSelfDestruct.9d2d3c8e3be07b36:core.Destroy.Core"(ptr @array), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.47314e2d166df899:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.47314e2d166df899:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.47314e2d166df899:core.Destroy.Core"(ptr %self), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0a222400365c29a7:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.88c7b0ef15c9f643:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_19.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !49
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc17_19.1.has_value), !dbg !49
// CHECK:STDOUT:   %.loc17_19.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   call void @"_CSelfDestruct.0a222400365c29a7:core.Destroy.Core"(ptr %.loc17_19.2.value), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.88c7b0ef15c9f643:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.88c7b0ef15c9f643:core.Destroy.Core"(ptr %self), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d75ae2cfe181bf5c:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_19.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !57
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc17_19.1.has_value), !dbg !57
// CHECK:STDOUT:   %.loc17_19.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !57
// CHECK:STDOUT:   call void @"_CSelfDestruct.0a222400365c29a7:core.Destroy.Core"(ptr %.loc17_19.2.value), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d75ae2cfe181bf5c:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d75ae2cfe181bf5c:core.Destroy.Core"(ptr %self), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dfe4cc77d44e03b0:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.9d2d3c8e3be07b36:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_43.1.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 5, !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc16_43.1.array.index), !dbg !69
// CHECK:STDOUT:   %.loc16_43.2.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 4, !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc16_43.2.array.index), !dbg !69
// CHECK:STDOUT:   %.loc16_43.3.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 3, !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc16_43.3.array.index), !dbg !69
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 2, !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc16_43.4.array.index), !dbg !69
// CHECK:STDOUT:   %.loc16_43.5.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 1, !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc16_43.5.array.index), !dbg !69
// CHECK:STDOUT:   %.loc16_43.6.array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i64 0, !dbg !69
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc16_43.6.array.index), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.9d2d3c8e3be07b36:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.9d2d3c8e3be07b36:core.Destroy.Core"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.Array.e6a7e802f5bda6f1.Core:Iterate.Core.cb65b0323a88fe77"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT:   ret i32 0, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.Array.e6a7e802f5bda6f1.Core:Iterate.Core.cb65b0323a88fe77"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !83 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !90
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !90
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !91
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !93
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !95
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !95
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !94
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !94
// CHECK:STDOUT:   call void @_CSome.Optional.Core.8758b884e7707351(ptr %return, i32 %6), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.8758b884e7707351(ptr %return), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.8758b884e7707351(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75"(ptr %self), !dbg !112
// CHECK:STDOUT:   ret i1 %1, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.8758b884e7707351(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75"(ptr %self), !dbg !121
// CHECK:STDOUT:   ret i32 %1, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !125 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.912f2df1be3b96b1.Core:Int.9452f4c51951679b.Core:AddAssignWith.8dbbdb092e12f130.Core.2fed6acaf7ad0605"(ptr %self, i32 1), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.8758b884e7707351(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !133 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75"(ptr %return, i32 %value), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.8758b884e7707351(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !141 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75"(ptr %return), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75"(ptr %value) #0 !dbg !146 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !151
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !151
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !151
// CHECK:STDOUT:   ret i1 %2, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75"(ptr %value) #0 !dbg !154 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !159
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !159
// CHECK:STDOUT:   ret i32 %1, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.912f2df1be3b96b1.Core:Int.9452f4c51951679b.Core:AddAssignWith.8dbbdb092e12f130.Core.2fed6acaf7ad0605"(ptr %self, i32 %other) #2 !dbg !162 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.31e2532c327e6d0d:ImplicitAs.3d18dd75cd3bccde.Core.d8021919251f415d"(i32 %other), !dbg !165
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !166
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !166
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !166
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !170 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !179
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !179
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !181
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !181
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !184 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !191
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !191
// CHECK:STDOUT:   ret void, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.31e2532c327e6d0d:ImplicitAs.3d18dd75cd3bccde.Core.d8021919251f415d"(i32 %self) #0 !dbg !193 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !198
// CHECK:STDOUT:   ret i32 %1, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !202 {
// CHECK:STDOUT:   ret i32 %self, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.dfe4cc77d44e03b0:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !23 = !{!11, !11, !6}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!11, !11}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 18, column: 5, scope: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 3, scope: !14)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 1, scope: !14)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.47314e2d166df899:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !11)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 7, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.47314e2d166df899:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !11)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 7, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0a222400365c29a7:core.Destroy.Core", scope: null, file: !4, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !6)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 7, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !11)
// CHECK:STDOUT: !45 = !DILocation(line: 17, column: 7, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.88c7b0ef15c9f643:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !11)
// CHECK:STDOUT: !49 = !DILocation(line: 17, column: 7, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.88c7b0ef15c9f643:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !11)
// CHECK:STDOUT: !53 = !DILocation(line: 17, column: 7, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d75ae2cfe181bf5c:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !11)
// CHECK:STDOUT: !57 = !DILocation(line: 17, column: 7, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d75ae2cfe181bf5c:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !11)
// CHECK:STDOUT: !61 = !DILocation(line: 17, column: 7, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dfe4cc77d44e03b0:core.Destroy.Core", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !11)
// CHECK:STDOUT: !65 = !DILocation(line: 17, column: 7, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.9d2d3c8e3be07b36:core.Destroy.Core", scope: null, file: !4, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !11)
// CHECK:STDOUT: !69 = !DILocation(line: 16, column: 26, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.9d2d3c8e3be07b36:core.Destroy.Core", scope: null, file: !4, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !11)
// CHECK:STDOUT: !73 = !DILocation(line: 16, column: 26, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.Array.e6a7e802f5bda6f1.Core:Iterate.Core.cb65b0323a88fe77", scope: null, file: !75, line: 23, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !82)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !11)
// CHECK:STDOUT: !81 = !DILocation(line: 23, column: 38, scope: !76)
// CHECK:STDOUT: !82 = !{!77}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.Array.e6a7e802f5bda6f1.Core:Iterate.Core.cb65b0323a88fe77", scope: null, file: !75, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !104)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !11)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 2, scope: !83, type: !6)
// CHECK:STDOUT: !90 = !DILocation(line: 25, column: 9, scope: !83)
// CHECK:STDOUT: !91 = !DILocation(line: 25, column: 8, scope: !83)
// CHECK:STDOUT: !93 = !DILocation(line: 26, column: 7, scope: !83)
// CHECK:STDOUT: !94 = !DILocation(line: 27, column: 31, scope: !83)
// CHECK:STDOUT: !95 = !DILocation(line: 27, column: 36, scope: !83)
// CHECK:STDOUT: !96 = !DILocation(line: 27, column: 14, scope: !83)
// CHECK:STDOUT: !97 = !{!11, !6}
// CHECK:STDOUT: !98 = !DISubroutineType(types: !97)
// CHECK:STDOUT: !99 = !DILocation(line: 27, column: 7, scope: !83)
// CHECK:STDOUT: !100 = !DILocation(line: 29, column: 14, scope: !83)
// CHECK:STDOUT: !101 = !{!11}
// CHECK:STDOUT: !102 = !DISubroutineType(types: !101)
// CHECK:STDOUT: !103 = !DILocation(line: 29, column: 7, scope: !83)
// CHECK:STDOUT: !104 = !{!84, !85}
// CHECK:STDOUT: !105 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.8758b884e7707351", scope: null, file: !105, line: 36, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !114)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !11)
// CHECK:STDOUT: !112 = !DILocation(line: 37, column: 12, scope: !106)
// CHECK:STDOUT: !113 = !DILocation(line: 37, column: 5, scope: !106)
// CHECK:STDOUT: !114 = !{!107}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.8758b884e7707351", scope: null, file: !105, line: 39, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !123)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !11)
// CHECK:STDOUT: !121 = !DILocation(line: 40, column: 12, scope: !115)
// CHECK:STDOUT: !122 = !DILocation(line: 40, column: 5, scope: !115)
// CHECK:STDOUT: !123 = !{!116}
// CHECK:STDOUT: !124 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !124, line: 396, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !132)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !6)
// CHECK:STDOUT: !128 = !DILocation(line: 398, column: 5, scope: !125)
// CHECK:STDOUT: !129 = !{null, !6, !6}
// CHECK:STDOUT: !130 = !DISubroutineType(types: !129)
// CHECK:STDOUT: !131 = !DILocation(line: 396, column: 3, scope: !125)
// CHECK:STDOUT: !132 = !{!126}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.8758b884e7707351", scope: null, file: !105, line: 33, type: !98, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !140)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !6)
// CHECK:STDOUT: !138 = !DILocation(line: 34, column: 12, scope: !133)
// CHECK:STDOUT: !139 = !DILocation(line: 34, column: 5, scope: !133)
// CHECK:STDOUT: !140 = !{!134}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.8758b884e7707351", scope: null, file: !105, line: 30, type: !102, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !144 = !DILocation(line: 31, column: 12, scope: !141)
// CHECK:STDOUT: !145 = !DILocation(line: 31, column: 5, scope: !141)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75", scope: null, file: !105, line: 141, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !153)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !11)
// CHECK:STDOUT: !151 = !DILocation(line: 142, column: 12, scope: !146)
// CHECK:STDOUT: !152 = !DILocation(line: 142, column: 5, scope: !146)
// CHECK:STDOUT: !153 = !{!147}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75", scope: null, file: !105, line: 144, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !161)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !11)
// CHECK:STDOUT: !159 = !DILocation(line: 145, column: 12, scope: !154)
// CHECK:STDOUT: !160 = !DILocation(line: 145, column: 5, scope: !154)
// CHECK:STDOUT: !161 = !{!155}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.912f2df1be3b96b1.Core:Int.9452f4c51951679b.Core:AddAssignWith.8dbbdb092e12f130.Core.2fed6acaf7ad0605", scope: null, file: !124, line: 331, type: !130, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !6)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 2, scope: !162, type: !6)
// CHECK:STDOUT: !165 = !DILocation(line: 0, scope: !162)
// CHECK:STDOUT: !166 = !DILocation(line: 331, column: 3, scope: !162)
// CHECK:STDOUT: !167 = !{!6, !6}
// CHECK:STDOUT: !168 = !DISubroutineType(types: !167)
// CHECK:STDOUT: !169 = !{!163, !164}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75", scope: null, file: !105, line: 131, type: !98, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !183)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !6)
// CHECK:STDOUT: !179 = !DILocation(line: 137, column: 5, scope: !170)
// CHECK:STDOUT: !181 = !DILocation(line: 138, column: 5, scope: !170)
// CHECK:STDOUT: !182 = !DILocation(line: 139, column: 5, scope: !170)
// CHECK:STDOUT: !183 = !{!171}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.64609c610134ef75", scope: null, file: !105, line: 125, type: !102, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !191 = !DILocation(line: 128, column: 5, scope: !184)
// CHECK:STDOUT: !192 = !DILocation(line: 129, column: 5, scope: !184)
// CHECK:STDOUT: !193 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.31e2532c327e6d0d:ImplicitAs.3d18dd75cd3bccde.Core.d8021919251f415d", scope: null, file: !124, line: 64, type: !168, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !201)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !193, type: !6)
// CHECK:STDOUT: !198 = !DILocation(line: 66, column: 17, scope: !193)
// CHECK:STDOUT: !200 = !DILocation(line: 66, column: 5, scope: !193)
// CHECK:STDOUT: !201 = !{!194}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !124, line: 59, type: !168, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !209)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !202, type: !6)
// CHECK:STDOUT: !208 = !DILocation(line: 59, column: 28, scope: !202)
// CHECK:STDOUT: !209 = !{!203}
// CHECK:STDOUT:
