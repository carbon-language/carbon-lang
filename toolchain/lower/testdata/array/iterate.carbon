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
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.0c0d9e3bce9efa01(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.0c0d9e3bce9efa01(ptr %.loc17_19.1.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !9
// CHECK:STDOUT:   br label %for.next, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.928f6c9bcff4e104:core.Destroy.Core"(ptr %.loc17_19.1.temp), !dbg !8
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
// CHECK:STDOUT: define weak_odr void @"_COp.1bc1533dda6a5e5d:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.928f6c9bcff4e104:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ee2ca6e48da1a18:core.Destroy.Core"(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.b8111087f462a219:Iterate.Core.50a5645d0fe1902e"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT:   ret i32 0, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.b8111087f462a219:Iterate.Core.50a5645d0fe1902e"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !42 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !48
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !48
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !49
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !50
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !51
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !51
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !52
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !52
// CHECK:STDOUT:   call void @_CSome.Optional.Core.0c0d9e3bce9efa01(ptr %return, i32 %6), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.0c0d9e3bce9efa01(ptr %return), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.0c0d9e3bce9efa01(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %self), !dbg !62
// CHECK:STDOUT:   ret i1 %1, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.0c0d9e3bce9efa01(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %self), !dbg !67
// CHECK:STDOUT:   ret i32 %1, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.5e64a0d5d6c6f700.Core:Int.9452f4c51951679b.Core:AddAssignWith.4fe8b3905f8c6cd6.Core.d8796c5c9758c898"(ptr %self, i32 1), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.0c0d9e3bce9efa01(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !75 {
// CHECK:STDOUT:   call void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %return, i32 %value), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.0c0d9e3bce9efa01(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !82 {
// CHECK:STDOUT:   call void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %return), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %value) #0 !dbg !87 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !90
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !90
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !90
// CHECK:STDOUT:   ret i1 %2, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr %value) #0 !dbg !92 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !95
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !95
// CHECK:STDOUT:   ret i32 %1, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.5e64a0d5d6c6f700.Core:Int.9452f4c51951679b.Core:AddAssignWith.4fe8b3905f8c6cd6.Core.d8796c5c9758c898"(ptr %self, i32 %other) #2 !dbg !97 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.c87c855149fb259e:ImplicitAs.e4ded66cc6bddc4a.Core.1bcae782b458290d"(i32 %other), !dbg !103
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !104
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !104
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !104
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !105 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !108
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !109
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !109
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !111 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !112
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.c87c855149fb259e:ImplicitAs.e4ded66cc6bddc4a.Core.1bcae782b458290d"(i32 %self) #0 !dbg !114 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !119
// CHECK:STDOUT:   ret i32 %1, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !121 {
// CHECK:STDOUT:   ret i32 %self, !dbg !124
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
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1bc1533dda6a5e5d:core.Destroy.Core", scope: null, file: !20, line: 124, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !24)
// CHECK:STDOUT: !20 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !19, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 124, column: 19, scope: !19)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.928f6c9bcff4e104:core.Destroy.Core", scope: null, file: !3, line: 17, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 7, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ee2ca6e48da1a18:core.Destroy.Core", scope: null, file: !3, line: 16, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !32)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !31, type: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 26, scope: !31)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.b8111087f462a219:Iterate.Core.50a5645d0fe1902e", scope: null, file: !36, line: 23, type: !37, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !36 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !37 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !38 = !{!15, !23}
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !35, type: !23)
// CHECK:STDOUT: !41 = !DILocation(line: 23, column: 46, scope: !35)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.b8111087f462a219:Iterate.Core.50a5645d0fe1902e", scope: null, file: !36, line: 24, type: !43, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !44 = !{!23, !23, !23}
// CHECK:STDOUT: !45 = !{!46, !47}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !42, type: !23)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 2, scope: !42, type: !23)
// CHECK:STDOUT: !48 = !DILocation(line: 25, column: 9, scope: !42)
// CHECK:STDOUT: !49 = !DILocation(line: 25, column: 8, scope: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 26, column: 7, scope: !42)
// CHECK:STDOUT: !51 = !DILocation(line: 27, column: 36, scope: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 27, column: 31, scope: !42)
// CHECK:STDOUT: !53 = !DILocation(line: 27, column: 14, scope: !42)
// CHECK:STDOUT: !54 = !DILocation(line: 27, column: 7, scope: !42)
// CHECK:STDOUT: !55 = !DILocation(line: 29, column: 14, scope: !42)
// CHECK:STDOUT: !56 = !DILocation(line: 29, column: 7, scope: !42)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.0c0d9e3bce9efa01", scope: null, file: !20, line: 36, type: !58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !59 = !{!23, !23}
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !57, type: !23)
// CHECK:STDOUT: !62 = !DILocation(line: 37, column: 12, scope: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 37, column: 5, scope: !57)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.0c0d9e3bce9efa01", scope: null, file: !20, line: 39, type: !37, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !65)
// CHECK:STDOUT: !65 = !{!66}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !64, type: !23)
// CHECK:STDOUT: !67 = !DILocation(line: 40, column: 12, scope: !64)
// CHECK:STDOUT: !68 = !DILocation(line: 40, column: 5, scope: !64)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !70, line: 363, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !71)
// CHECK:STDOUT: !70 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !71 = !{!72}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !69, type: !15)
// CHECK:STDOUT: !73 = !DILocation(line: 365, column: 5, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 363, column: 3, scope: !69)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.0c0d9e3bce9efa01", scope: null, file: !20, line: 33, type: !76, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DISubroutineType(types: !77)
// CHECK:STDOUT: !77 = !{!23, !15}
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !75, type: !15)
// CHECK:STDOUT: !80 = !DILocation(line: 34, column: 12, scope: !75)
// CHECK:STDOUT: !81 = !DILocation(line: 34, column: 5, scope: !75)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.0c0d9e3bce9efa01", scope: null, file: !20, line: 30, type: !83, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !83 = !DISubroutineType(types: !84)
// CHECK:STDOUT: !84 = !{!23}
// CHECK:STDOUT: !85 = !DILocation(line: 31, column: 12, scope: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 31, column: 5, scope: !82)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085", scope: null, file: !20, line: 141, type: !58, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !23)
// CHECK:STDOUT: !90 = !DILocation(line: 142, column: 12, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 142, column: 5, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085", scope: null, file: !20, line: 144, type: !37, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !23)
// CHECK:STDOUT: !95 = !DILocation(line: 145, column: 12, scope: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 145, column: 5, scope: !92)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.5e64a0d5d6c6f700.Core:Int.9452f4c51951679b.Core:AddAssignWith.4fe8b3905f8c6cd6.Core.d8796c5c9758c898", scope: null, file: !70, line: 299, type: !98, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DISubroutineType(types: !99)
// CHECK:STDOUT: !99 = !{null, !15, !15}
// CHECK:STDOUT: !100 = !{!101, !102}
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !97, type: !15)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 2, scope: !97, type: !15)
// CHECK:STDOUT: !103 = !DILocation(line: 4294967295, scope: !97)
// CHECK:STDOUT: !104 = !DILocation(line: 299, column: 3, scope: !97)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085", scope: null, file: !20, line: 131, type: !76, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !106)
// CHECK:STDOUT: !106 = !{!107}
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !105, type: !15)
// CHECK:STDOUT: !108 = !DILocation(line: 137, column: 5, scope: !105)
// CHECK:STDOUT: !109 = !DILocation(line: 138, column: 5, scope: !105)
// CHECK:STDOUT: !110 = !DILocation(line: 139, column: 5, scope: !105)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "None", linkageName: "_CNone.ca6ae00af8d27ba2:OptionalStorage.Core.6d535d38260d6085", scope: null, file: !20, line: 125, type: !83, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !112 = !DILocation(line: 128, column: 5, scope: !111)
// CHECK:STDOUT: !113 = !DILocation(line: 129, column: 5, scope: !111)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.c87c855149fb259e:ImplicitAs.e4ded66cc6bddc4a.Core.1bcae782b458290d", scope: null, file: !70, line: 62, type: !115, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !117)
// CHECK:STDOUT: !115 = !DISubroutineType(types: !116)
// CHECK:STDOUT: !116 = !{!15, !15}
// CHECK:STDOUT: !117 = !{!118}
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !114, type: !15)
// CHECK:STDOUT: !119 = !DILocation(line: 64, column: 17, scope: !114)
// CHECK:STDOUT: !120 = !DILocation(line: 64, column: 5, scope: !114)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !70, line: 57, type: !115, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !122)
// CHECK:STDOUT: !122 = !{!123}
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !121, type: !15)
// CHECK:STDOUT: !124 = !DILocation(line: 57, column: 36, scope: !121)
