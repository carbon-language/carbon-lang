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
// CHECK:STDOUT:   %.loc17_19.1.temp = alloca <{ i32, i1 }>, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_43.3.temp), !dbg !7
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 0, !dbg !7
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 1, !dbg !7
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 2, !dbg !7
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 3, !dbg !7
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 4, !dbg !7
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 5, !dbg !7
// CHECK:STDOUT:   %array_type.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.b8111087f462a219:Iterate.Core.50a5645d0fe1902e"(ptr @array.loc16_43.24), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %array_type.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.b8111087f462a219:Iterate.Core.50a5645d0fe1902e"(ptr %.loc17_19.1.temp, ptr @array.loc16_43.24, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.9d6f831f9a08a854(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.9d6f831f9a08a854(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !9
// CHECK:STDOUT:   br label %for.next, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.fb3a1b9ee3968ef1:core.Destroy.Core"(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.8ee2ca6e48da1a18:core.Destroy.Core"(ptr @array), !dbg !7
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
// CHECK:STDOUT: define weak_odr void @"_COp.fb3a1b9ee3968ef1:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ee2ca6e48da1a18:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.b8111087f462a219:Iterate.Core.50a5645d0fe1902e"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT:   ret i32 0, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.b8111087f462a219:Iterate.Core.50a5645d0fe1902e"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !37 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !43
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !43
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !44
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !45
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !46
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !46
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !47
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !47
// CHECK:STDOUT:   call void @_CSome.Optional.Core.9d6f831f9a08a854(ptr %return, i32 %6), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.9d6f831f9a08a854(ptr %return), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.9d6f831f9a08a854(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %self), !dbg !58
// CHECK:STDOUT:   ret i1 %1, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.9d6f831f9a08a854(ptr %self) #0 !dbg !60 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %self), !dbg !63
// CHECK:STDOUT:   ret i32 %1, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.5e64a0d5d6c6f700.Core:Int.9452f4c51951679b.Core:AddAssignWith.4fe8b3905f8c6cd6.Core.d8796c5c9758c898"(ptr %self, i32 1), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.9d6f831f9a08a854(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !71 {
// CHECK:STDOUT:   call void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %return, i32 %value), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.9d6f831f9a08a854(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !78 {
// CHECK:STDOUT:   call void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %return), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %value) #0 !dbg !83 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !86
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !86
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !86
// CHECK:STDOUT:   ret i1 %2, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %value) #0 !dbg !88 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !91
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !91
// CHECK:STDOUT:   ret i32 %1, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.5e64a0d5d6c6f700.Core:Int.9452f4c51951679b.Core:AddAssignWith.4fe8b3905f8c6cd6.Core.d8796c5c9758c898"(ptr %self, i32 %other) #2 !dbg !93 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.c87c855149fb259e:ImplicitAs.e4ded66cc6bddc4a.Core.1bcae782b458290d"(i32 %other), !dbg !99
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !100
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !100
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !100
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !101 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !104
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !104
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !105
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !107 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !108
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.c87c855149fb259e:ImplicitAs.e4ded66cc6bddc4a.Core.1bcae782b458290d"(i32 %self) #0 !dbg !110 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !115
// CHECK:STDOUT:   ret i32 %1, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !117 {
// CHECK:STDOUT:   ret i32 %self, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "iterate.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 26, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 7, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 7, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fb3a1b9ee3968ef1:core.Destroy.Core", scope: null, file: !3, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 17, column: 7, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ee2ca6e48da1a18:core.Destroy.Core", scope: null, file: !3, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !22)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 26, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.b8111087f462a219:Iterate.Core.50a5645d0fe1902e", scope: null, file: !31, line: 23, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !34)
// CHECK:STDOUT: !31 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !32 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !33 = !{!15, !22}
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !30, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 23, column: 46, scope: !30)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.b8111087f462a219:Iterate.Core.50a5645d0fe1902e", scope: null, file: !31, line: 24, type: !38, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !39 = !{!22, !22, !22}
// CHECK:STDOUT: !40 = !{!41, !42}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !37, type: !22)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !37, type: !22)
// CHECK:STDOUT: !43 = !DILocation(line: 25, column: 9, scope: !37)
// CHECK:STDOUT: !44 = !DILocation(line: 25, column: 8, scope: !37)
// CHECK:STDOUT: !45 = !DILocation(line: 26, column: 7, scope: !37)
// CHECK:STDOUT: !46 = !DILocation(line: 27, column: 36, scope: !37)
// CHECK:STDOUT: !47 = !DILocation(line: 27, column: 31, scope: !37)
// CHECK:STDOUT: !48 = !DILocation(line: 27, column: 14, scope: !37)
// CHECK:STDOUT: !49 = !DILocation(line: 27, column: 7, scope: !37)
// CHECK:STDOUT: !50 = !DILocation(line: 29, column: 14, scope: !37)
// CHECK:STDOUT: !51 = !DILocation(line: 29, column: 7, scope: !37)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.9d6f831f9a08a854", scope: null, file: !53, line: 36, type: !54, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !56)
// CHECK:STDOUT: !53 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !54 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !55 = !{!22, !22}
// CHECK:STDOUT: !56 = !{!57}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !52, type: !22)
// CHECK:STDOUT: !58 = !DILocation(line: 37, column: 12, scope: !52)
// CHECK:STDOUT: !59 = !DILocation(line: 37, column: 5, scope: !52)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.9d6f831f9a08a854", scope: null, file: !53, line: 39, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !61)
// CHECK:STDOUT: !61 = !{!62}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !60, type: !22)
// CHECK:STDOUT: !63 = !DILocation(line: 40, column: 12, scope: !60)
// CHECK:STDOUT: !64 = !DILocation(line: 40, column: 5, scope: !60)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !66, line: 363, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !67)
// CHECK:STDOUT: !66 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !65, type: !15)
// CHECK:STDOUT: !69 = !DILocation(line: 365, column: 5, scope: !65)
// CHECK:STDOUT: !70 = !DILocation(line: 363, column: 3, scope: !65)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.9d6f831f9a08a854", scope: null, file: !53, line: 33, type: !72, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DISubroutineType(types: !73)
// CHECK:STDOUT: !73 = !{!22, !15}
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !71, type: !15)
// CHECK:STDOUT: !76 = !DILocation(line: 34, column: 12, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 34, column: 5, scope: !71)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.9d6f831f9a08a854", scope: null, file: !53, line: 30, type: !79, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !79 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !80 = !{!22}
// CHECK:STDOUT: !81 = !DILocation(line: 31, column: 12, scope: !78)
// CHECK:STDOUT: !82 = !DILocation(line: 31, column: 5, scope: !78)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085", scope: null, file: !53, line: 141, type: !54, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !84)
// CHECK:STDOUT: !84 = !{!85}
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !83, type: !22)
// CHECK:STDOUT: !86 = !DILocation(line: 142, column: 12, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 142, column: 5, scope: !83)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085", scope: null, file: !53, line: 144, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !22)
// CHECK:STDOUT: !91 = !DILocation(line: 145, column: 12, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 145, column: 5, scope: !88)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.5e64a0d5d6c6f700.Core:Int.9452f4c51951679b.Core:AddAssignWith.4fe8b3905f8c6cd6.Core.d8796c5c9758c898", scope: null, file: !66, line: 299, type: !94, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DISubroutineType(types: !95)
// CHECK:STDOUT: !95 = !{null, !15, !15}
// CHECK:STDOUT: !96 = !{!97, !98}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !93, type: !15)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 2, scope: !93, type: !15)
// CHECK:STDOUT: !99 = !DILocation(line: 4294967295, scope: !93)
// CHECK:STDOUT: !100 = !DILocation(line: 299, column: 3, scope: !93)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085", scope: null, file: !53, line: 131, type: !72, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !102)
// CHECK:STDOUT: !102 = !{!103}
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !101, type: !15)
// CHECK:STDOUT: !104 = !DILocation(line: 137, column: 5, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 138, column: 5, scope: !101)
// CHECK:STDOUT: !106 = !DILocation(line: 139, column: 5, scope: !101)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "None", linkageName: "_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085", scope: null, file: !53, line: 125, type: !79, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !108 = !DILocation(line: 128, column: 5, scope: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 129, column: 5, scope: !107)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c87c855149fb259e:ImplicitAs.e4ded66cc6bddc4a.Core.1bcae782b458290d", scope: null, file: !66, line: 62, type: !111, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !113)
// CHECK:STDOUT: !111 = !DISubroutineType(types: !112)
// CHECK:STDOUT: !112 = !{!15, !15}
// CHECK:STDOUT: !113 = !{!114}
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !110, type: !15)
// CHECK:STDOUT: !115 = !DILocation(line: 64, column: 17, scope: !110)
// CHECK:STDOUT: !116 = !DILocation(line: 64, column: 5, scope: !110)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !66, line: 57, type: !111, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !118)
// CHECK:STDOUT: !118 = !{!119}
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !117, type: !15)
// CHECK:STDOUT: !120 = !DILocation(line: 57, column: 36, scope: !117)
