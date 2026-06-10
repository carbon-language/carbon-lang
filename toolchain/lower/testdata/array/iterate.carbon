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
// CHECK:STDOUT:   %.loc17_19.3.temp = alloca <{ i32, i1 }>, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc16_43.3.temp), !dbg !7
// CHECK:STDOUT:   %.loc16_43.4.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 0, !dbg !7
// CHECK:STDOUT:   %.loc16_43.7.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 1, !dbg !7
// CHECK:STDOUT:   %.loc16_43.10.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 2, !dbg !7
// CHECK:STDOUT:   %.loc16_43.13.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 3, !dbg !7
// CHECK:STDOUT:   %.loc16_43.16.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 4, !dbg !7
// CHECK:STDOUT:   %.loc16_43.19.array.index = getelementptr inbounds [6 x i32], ptr %.loc16_43.3.temp, i32 0, i64 5, !dbg !7
// CHECK:STDOUT:   %array_type.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.620427d555024fe2"(ptr @array.loc16_43.24), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %array_type.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.3.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.e163a239f9fa927c:Iterate.Core.620427d555024fe2"(ptr %.loc17_19.3.temp, ptr @array.loc16_43.24, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.18012f5191ea64dc(ptr %.loc17_19.3.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.18012f5191ea64dc(ptr %.loc17_19.3.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !9
// CHECK:STDOUT:   br label %for.next, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.ab782c7eec490fc7:core.Destroy.Core"(ptr %.loc17_19.3.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.3a5df7cec745952b:core.Destroy.Core"(ptr @array), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94ba1c30d421702e:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0fc9634857315db2:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.69700c1fc1b2744c:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f49c06ba556be406:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ab782c7eec490fc7:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3a5df7cec745952b:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.620427d555024fe2"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT:   ret i32 0, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.e163a239f9fa927c:Iterate.Core.620427d555024fe2"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !53 {
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
// CHECK:STDOUT:   call void @_CSome.Optional.Core.18012f5191ea64dc(ptr %return, i32 %6), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.18012f5191ea64dc(ptr %return), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.18012f5191ea64dc(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %self), !dbg !74
// CHECK:STDOUT:   ret i1 %1, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.18012f5191ea64dc(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %self), !dbg !79
// CHECK:STDOUT:   ret i32 %1, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661"(ptr %self, i32 1), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.18012f5191ea64dc(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %return, i32 %value), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.18012f5191ea64dc(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !94 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %return), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %value) #0 !dbg !99 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !102
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !102
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !102
// CHECK:STDOUT:   ret i1 %2, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %value) #0 !dbg !104 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !107
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !107
// CHECK:STDOUT:   ret i32 %1, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661"(ptr %self, i32 %other) #2 !dbg !109 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47"(i32 %other), !dbg !115
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !116
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !116
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !116
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !117 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !120
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !120
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !121
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !121
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !123 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !124
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47"(i32 %self) #0 !dbg !126 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !131
// CHECK:STDOUT:   ret i32 %1, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !133 {
// CHECK:STDOUT:   ret i32 %self, !dbg !136
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !3, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 7, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94ba1c30d421702e:core.Destroy.Core", scope: null, file: !3, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 17, column: 7, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0fc9634857315db2:core.Destroy.Core", scope: null, file: !3, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !15)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 7, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp.69700c1fc1b2744c:core.Destroy.Core", scope: null, file: !3, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 7, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f49c06ba556be406:core.Destroy.Core", scope: null, file: !3, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !22)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 7, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ab782c7eec490fc7:core.Destroy.Core", scope: null, file: !3, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !22)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 7, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3a5df7cec745952b:core.Destroy.Core", scope: null, file: !3, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !22)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 26, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.e163a239f9fa927c:Iterate.Core.620427d555024fe2", scope: null, file: !47, line: 23, type: !48, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !47 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !48 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !49 = !{!15, !22}
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !46, type: !22)
// CHECK:STDOUT: !52 = !DILocation(line: 23, column: 38, scope: !46)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.e163a239f9fa927c:Iterate.Core.620427d555024fe2", scope: null, file: !47, line: 24, type: !54, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !56)
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
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.18012f5191ea64dc", scope: null, file: !69, line: 36, type: !70, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !72)
// CHECK:STDOUT: !69 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !70 = !DISubroutineType(types: !71)
// CHECK:STDOUT: !71 = !{!22, !22}
// CHECK:STDOUT: !72 = !{!73}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !68, type: !22)
// CHECK:STDOUT: !74 = !DILocation(line: 37, column: 12, scope: !68)
// CHECK:STDOUT: !75 = !DILocation(line: 37, column: 5, scope: !68)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.18012f5191ea64dc", scope: null, file: !69, line: 39, type: !48, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !77)
// CHECK:STDOUT: !77 = !{!78}
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !76, type: !22)
// CHECK:STDOUT: !79 = !DILocation(line: 40, column: 12, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 40, column: 5, scope: !76)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !82, line: 396, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !83)
// CHECK:STDOUT: !82 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !81, type: !15)
// CHECK:STDOUT: !85 = !DILocation(line: 398, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 396, column: 3, scope: !81)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.18012f5191ea64dc", scope: null, file: !69, line: 33, type: !88, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DISubroutineType(types: !89)
// CHECK:STDOUT: !89 = !{!22, !15}
// CHECK:STDOUT: !90 = !{!91}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !87, type: !15)
// CHECK:STDOUT: !92 = !DILocation(line: 34, column: 12, scope: !87)
// CHECK:STDOUT: !93 = !DILocation(line: 34, column: 5, scope: !87)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.18012f5191ea64dc", scope: null, file: !69, line: 30, type: !95, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !95 = !DISubroutineType(types: !96)
// CHECK:STDOUT: !96 = !{!22}
// CHECK:STDOUT: !97 = !DILocation(line: 31, column: 12, scope: !94)
// CHECK:STDOUT: !98 = !DILocation(line: 31, column: 5, scope: !94)
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !69, line: 141, type: !70, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !100)
// CHECK:STDOUT: !100 = !{!101}
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !99, type: !22)
// CHECK:STDOUT: !102 = !DILocation(line: 142, column: 12, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 142, column: 5, scope: !99)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !69, line: 144, type: !48, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !105)
// CHECK:STDOUT: !105 = !{!106}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !104, type: !22)
// CHECK:STDOUT: !107 = !DILocation(line: 145, column: 12, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 145, column: 5, scope: !104)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.6f9e8c1abf8bb3ea.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.44914b6ff6901661", scope: null, file: !82, line: 331, type: !110, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DISubroutineType(types: !111)
// CHECK:STDOUT: !111 = !{null, !15, !15}
// CHECK:STDOUT: !112 = !{!113, !114}
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !109, type: !15)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 2, scope: !109, type: !15)
// CHECK:STDOUT: !115 = !DILocation(line: 4294967295, scope: !109)
// CHECK:STDOUT: !116 = !DILocation(line: 331, column: 3, scope: !109)
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !69, line: 131, type: !88, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !118)
// CHECK:STDOUT: !118 = !{!119}
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !117, type: !15)
// CHECK:STDOUT: !120 = !DILocation(line: 137, column: 5, scope: !117)
// CHECK:STDOUT: !121 = !DILocation(line: 138, column: 5, scope: !117)
// CHECK:STDOUT: !122 = !DILocation(line: 139, column: 5, scope: !117)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !69, line: 125, type: !95, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !124 = !DILocation(line: 128, column: 5, scope: !123)
// CHECK:STDOUT: !125 = !DILocation(line: 129, column: 5, scope: !123)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.40372ba0ea03ef47", scope: null, file: !82, line: 64, type: !127, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !129)
// CHECK:STDOUT: !127 = !DISubroutineType(types: !128)
// CHECK:STDOUT: !128 = !{!15, !15}
// CHECK:STDOUT: !129 = !{!130}
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !126, type: !15)
// CHECK:STDOUT: !131 = !DILocation(line: 66, column: 17, scope: !126)
// CHECK:STDOUT: !132 = !DILocation(line: 66, column: 5, scope: !126)
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !82, line: 59, type: !127, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !134)
// CHECK:STDOUT: !134 = !{!135}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !133, type: !15)
// CHECK:STDOUT: !136 = !DILocation(line: 59, column: 28, scope: !133)
