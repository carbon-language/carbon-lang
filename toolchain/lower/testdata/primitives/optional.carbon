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

// CHECK:STDOUT: ; ModuleID = 'optional.carbon'
// CHECK:STDOUT: source_filename = "optional.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CConvert.Main(ptr sret({ i32, i1 }) %return, ptr %o) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_20.1.temp = alloca { i32, i1 }, align 8, !dbg !7
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.217efae529e578bc(ptr %o), !dbg !8
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %if.then, label %if.else, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   %Optional.Get.call = call ptr @_CGet.Optional.Core.217efae529e578bc(ptr %o), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_20.1.temp), !dbg !7
// CHECK:STDOUT:   %.loc18_12.2 = load i32, ptr %Optional.Get.call, align 4, !dbg !11
// CHECK:STDOUT:   call void @"_CConvert.8d16edc1dfe20a7c:ImplicitAs.Core.f16306f3d30b9711"(ptr %.loc18_20.1.temp, i32 %.loc18_12.2), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CNone.Optional.Core.f16306f3d30b9711(ptr %return), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.217efae529e578bc(ptr %self) !dbg !14 {
// CHECK:STDOUT:   %1 = load ptr, ptr %self, align 8, !dbg !16
// CHECK:STDOUT:   %2 = icmp eq ptr %1, null, !dbg !16
// CHECK:STDOUT:   ret i1 %2, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.217efae529e578bc(ptr %self) !dbg !18 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.4f0b5cc38af595d2:OptionalStorage.Core.b88d1103f417c6d4"(ptr %self), !dbg !19
// CHECK:STDOUT:   ret ptr %1, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.8d16edc1dfe20a7c:ImplicitAs.Core.f16306f3d30b9711"(ptr sret({ i32, i1 }) %return, i32 %self) !dbg !21 {
// CHECK:STDOUT:   call void @_CSome.Optional.Core.f16306f3d30b9711(ptr %return, i32 %self), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.f16306f3d30b9711(ptr sret({ i32, i1 }) %return) !dbg !24 {
// CHECK:STDOUT:   call void @"_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %return), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.4f0b5cc38af595d2:OptionalStorage.Core.b88d1103f417c6d4"(ptr %value) !dbg !27 {
// CHECK:STDOUT:   %1 = load ptr, ptr %value, align 8, !dbg !28
// CHECK:STDOUT:   ret ptr %1, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.f16306f3d30b9711(ptr sret({ i32, i1 }) %return, i32 %value) !dbg !30 {
// CHECK:STDOUT:   call void @"_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr %return, i32 %value), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr sret({ i32, i1 }) %return) !dbg !33 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr void @"_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899"(ptr sret({ i32, i1 }) %return, i32 %self) !dbg !36 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !37
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw { i32, i1 }, ptr %return, i32 0, i32 1, !dbg !38
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.217efae529e578bc", scope: null, file: !15, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !16 = !DILocation(line: 32, column: 12, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 32, column: 5, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.217efae529e578bc", scope: null, file: !15, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 35, column: 12, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 35, column: 5, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8d16edc1dfe20a7c:ImplicitAs.Core.f16306f3d30b9711", scope: null, file: !15, line: 39, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 40, column: 14, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 40, column: 7, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.f16306f3d30b9711", scope: null, file: !15, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 26, column: 12, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 26, column: 5, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.4f0b5cc38af595d2:OptionalStorage.Core.b88d1103f417c6d4", scope: null, file: !15, line: 92, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 93, column: 12, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 93, column: 5, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.f16306f3d30b9711", scope: null, file: !15, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 29, column: 12, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 29, column: 5, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "None", linkageName: "_CNone.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899", scope: null, file: !15, line: 60, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !34 = !DILocation(line: 62, column: 5, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 63, column: 5, scope: !33)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.225258f1a45e9386:OptionalStorage.Core.5450dc8e8b8e0899", scope: null, file: !15, line: 65, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 69, column: 5, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 70, column: 5, scope: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 71, column: 5, scope: !36)
