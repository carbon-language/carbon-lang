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
// CHECK:STDOUT:   %array_type.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.9c4042a9412793cf"(ptr @array.loc16_43.24), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !8
// CHECK:STDOUT:   store i32 %array_type.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @"_CNext.e163a239f9fa927c:Iterate.Core.9c4042a9412793cf"(ptr %.loc17_19.5.temp, ptr @array.loc16_43.24, ptr %var), !dbg !8
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.a5214d4ceffb522b(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.a5214d4ceffb522b(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(i32 %Optional.Get.call), !dbg !9
// CHECK:STDOUT:   call void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   br label %for.next, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %.loc17_19.5.temp), !dbg !8
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.1366a4d5330239bd:core.Destroy.Core"(ptr @array), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1d6a7672bf2f14c6:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0239a84bcf0315a7:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ad05ff27ec7895c0:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ae16cb2db38d93af:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.891c4207e57d0f40:core.Destroy.Core"(ptr %self) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1366a4d5330239bd:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.e163a239f9fa927c:Iterate.Core.9c4042a9412793cf"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT:   ret i32 0, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.e163a239f9fa927c:Iterate.Core.9c4042a9412793cf"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !49 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !55
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 6, !dbg !55
// CHECK:STDOUT:   br i1 %2, label %3, label %7, !dbg !56
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !57
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !58
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !58
// CHECK:STDOUT:   %array.index = getelementptr inbounds [6 x i32], ptr %self, i32 0, i32 %5, !dbg !59
// CHECK:STDOUT:   %6 = load i32, ptr %array.index, align 4, !dbg !59
// CHECK:STDOUT:   call void @_CSome.Optional.Core.a5214d4ceffb522b(ptr %return, i32 %6), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.a5214d4ceffb522b(ptr %return), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.a5214d4ceffb522b(ptr %self) #0 !dbg !64 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret i1 %1, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.a5214d4ceffb522b(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %self), !dbg !75
// CHECK:STDOUT:   ret i32 %1, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 1), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.a5214d4ceffb522b(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !83 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %return, i32 %value), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.a5214d4ceffb522b(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !90 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %return), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %value) #0 !dbg !95 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !98
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !98
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !98
// CHECK:STDOUT:   ret i1 %2, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr %value) #0 !dbg !100 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !103
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !103
// CHECK:STDOUT:   ret i32 %1, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895"(ptr %self, i32 %other) #2 !dbg !105 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %other), !dbg !111
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !112
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !112
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !112
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !113 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !116
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !117
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !119 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !120
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044"(i32 %self) #0 !dbg !122 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !127
// CHECK:STDOUT:   ret i32 %1, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !129 {
// CHECK:STDOUT:   ret i32 %self, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.891c4207e57d0f40:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1d6a7672bf2f14c6:core.Destroy.Core", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 7, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0239a84bcf0315a7:core.Destroy.Core", scope: null, file: !1, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 17, column: 7, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ad05ff27ec7895c0:core.Destroy.Core", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !15)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 7, scope: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ae16cb2db38d93af:core.Destroy.Core", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !15)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 7, scope: !30)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Op", linkageName: "_COp.891c4207e57d0f40:core.Destroy.Core", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !15)
// CHECK:STDOUT: !37 = !DILocation(line: 17, column: 7, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1366a4d5330239bd:core.Destroy.Core", scope: null, file: !1, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !15)
// CHECK:STDOUT: !41 = !DILocation(line: 16, column: 26, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.e163a239f9fa927c:Iterate.Core.9c4042a9412793cf", scope: null, file: !43, line: 23, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !43 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !44 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !45 = !{!22, !15}
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !42, type: !15)
// CHECK:STDOUT: !48 = !DILocation(line: 23, column: 38, scope: !42)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.e163a239f9fa927c:Iterate.Core.9c4042a9412793cf", scope: null, file: !43, line: 24, type: !50, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !51 = !{!15, !15, !15}
// CHECK:STDOUT: !52 = !{!53, !54}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !49, type: !15)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 2, scope: !49, type: !15)
// CHECK:STDOUT: !55 = !DILocation(line: 25, column: 9, scope: !49)
// CHECK:STDOUT: !56 = !DILocation(line: 25, column: 8, scope: !49)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 7, scope: !49)
// CHECK:STDOUT: !58 = !DILocation(line: 27, column: 36, scope: !49)
// CHECK:STDOUT: !59 = !DILocation(line: 27, column: 31, scope: !49)
// CHECK:STDOUT: !60 = !DILocation(line: 27, column: 14, scope: !49)
// CHECK:STDOUT: !61 = !DILocation(line: 27, column: 7, scope: !49)
// CHECK:STDOUT: !62 = !DILocation(line: 29, column: 14, scope: !49)
// CHECK:STDOUT: !63 = !DILocation(line: 29, column: 7, scope: !49)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.a5214d4ceffb522b", scope: null, file: !65, line: 36, type: !66, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !68)
// CHECK:STDOUT: !65 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !66 = !DISubroutineType(types: !67)
// CHECK:STDOUT: !67 = !{!15, !15}
// CHECK:STDOUT: !68 = !{!69}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !64, type: !15)
// CHECK:STDOUT: !70 = !DILocation(line: 37, column: 12, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 37, column: 5, scope: !64)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.a5214d4ceffb522b", scope: null, file: !65, line: 39, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !73)
// CHECK:STDOUT: !73 = !{!74}
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !72, type: !15)
// CHECK:STDOUT: !75 = !DILocation(line: 40, column: 12, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 40, column: 5, scope: !72)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !78, line: 396, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !79)
// CHECK:STDOUT: !78 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !79 = !{!80}
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !77, type: !22)
// CHECK:STDOUT: !81 = !DILocation(line: 398, column: 5, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 396, column: 3, scope: !77)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.a5214d4ceffb522b", scope: null, file: !65, line: 33, type: !84, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DISubroutineType(types: !85)
// CHECK:STDOUT: !85 = !{!15, !22}
// CHECK:STDOUT: !86 = !{!87}
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !83, type: !22)
// CHECK:STDOUT: !88 = !DILocation(line: 34, column: 12, scope: !83)
// CHECK:STDOUT: !89 = !DILocation(line: 34, column: 5, scope: !83)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.a5214d4ceffb522b", scope: null, file: !65, line: 30, type: !91, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !91 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !92 = !{!15}
// CHECK:STDOUT: !93 = !DILocation(line: 31, column: 12, scope: !90)
// CHECK:STDOUT: !94 = !DILocation(line: 31, column: 5, scope: !90)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !65, line: 141, type: !66, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !96)
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !95, type: !15)
// CHECK:STDOUT: !98 = !DILocation(line: 142, column: 12, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 142, column: 5, scope: !95)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !65, line: 144, type: !44, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !15)
// CHECK:STDOUT: !103 = !DILocation(line: 145, column: 12, scope: !100)
// CHECK:STDOUT: !104 = !DILocation(line: 145, column: 5, scope: !100)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.39e2e78a3c2e036d.Core:Int.9452f4c51951679b.Core:AddAssignWith.bcc2a64c00f3554f.Core.83481f1872d67895", scope: null, file: !78, line: 331, type: !106, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DISubroutineType(types: !107)
// CHECK:STDOUT: !107 = !{null, !22, !22}
// CHECK:STDOUT: !108 = !{!109, !110}
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !105, type: !22)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !105, type: !22)
// CHECK:STDOUT: !111 = !DILocation(line: 0, scope: !105)
// CHECK:STDOUT: !112 = !DILocation(line: 331, column: 3, scope: !105)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !65, line: 131, type: !84, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !114)
// CHECK:STDOUT: !114 = !{!115}
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !113, type: !22)
// CHECK:STDOUT: !116 = !DILocation(line: 137, column: 5, scope: !113)
// CHECK:STDOUT: !117 = !DILocation(line: 138, column: 5, scope: !113)
// CHECK:STDOUT: !118 = !DILocation(line: 139, column: 5, scope: !113)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.553a1f59d1dd8381", scope: null, file: !65, line: 125, type: !91, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !120 = !DILocation(line: 128, column: 5, scope: !119)
// CHECK:STDOUT: !121 = !DILocation(line: 129, column: 5, scope: !119)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.bb469f8924cce044", scope: null, file: !78, line: 64, type: !123, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !125)
// CHECK:STDOUT: !123 = !DISubroutineType(types: !124)
// CHECK:STDOUT: !124 = !{!22, !22}
// CHECK:STDOUT: !125 = !{!126}
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !122, type: !22)
// CHECK:STDOUT: !127 = !DILocation(line: 66, column: 17, scope: !122)
// CHECK:STDOUT: !128 = !DILocation(line: 66, column: 5, scope: !122)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !78, line: 59, type: !123, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !130)
// CHECK:STDOUT: !130 = !{!131}
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !129, type: !22)
// CHECK:STDOUT: !132 = !DILocation(line: 59, column: 28, scope: !129)
// CHECK:STDOUT:
