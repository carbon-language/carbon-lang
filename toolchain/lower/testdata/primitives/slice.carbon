// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is an integration test for the Core.Slice in the prelude, so use the
// real prelude.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/slice.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/slice.carbon

// --- make.carbon
library "[[@TEST_NAME]]";

fn MakeSlice(p: i32*, n: u64) -> Core.Slice(i32) {
  return Core.Slice(i32).Make(p, n);
}

fn GetSize(s: Core.Slice(i32)) -> u64 {
  return s.Size();
}

// --- copy.carbon
library "[[@TEST_NAME]]";

fn CopySlice(s: Core.Slice(f64)) -> Core.Slice(f64) {
  return s;
}

// --- unformed.carbon
library "[[@TEST_NAME]]";

fn Unformed(p: i32*, n: u64) -> Core.Slice(i32) {
  var s: Core.Slice(i32);
  s = Core.Slice(i32).Make(p, n);
  return s;
}

// --- iterate.carbon
library "[[@TEST_NAME]]";

fn IterateSlice(s: Core.Slice(i32)) -> i32 {
  var sum: i32 = 0;
  for (x: i32 in s) {
    sum += x;
  }
  return sum;
}


// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'make.carbon'
// CHECK:STDOUT: source_filename = "make.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeSlice.Main(ptr sret({ ptr, i64 }) %return, ptr %p, i64 %n) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Slice.Core.84588f41d61dafba(ptr %return, ptr %p, i64 %n), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetSize.Main(ptr %s) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Slice.Size.call = call i64 @_CSize.Slice.Core.84588f41d61dafba(ptr %s), !dbg !19
// CHECK:STDOUT:   ret i64 %Slice.Size.call, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Slice.Core.84588f41d61dafba(ptr sret({ ptr, i64 }) %return, ptr %ptr, i64 %size) #0 !dbg !21 {
// CHECK:STDOUT:   %ptr1 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 0, !dbg !26
// CHECK:STDOUT:   store ptr %ptr, ptr %ptr1, align 8, !dbg !26
// CHECK:STDOUT:   %size2 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 1, !dbg !26
// CHECK:STDOUT:   store i64 %size, ptr %size2, align 8, !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @_CSize.Slice.Core.84588f41d61dafba(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !31
// CHECK:STDOUT:   %1 = load i64, ptr %size, align 8, !dbg !31
// CHECK:STDOUT:   ret i64 %1, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "make.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "MakeSlice", linkageName: "_CMakeSlice.Main", scope: null, file: !1, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !8}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !9 = !{!10, !11}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocalVariable(arg: 2, scope: !4, type: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 4, column: 10, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 4, column: 3, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "GetSize", linkageName: "_CGetSize.Main", scope: null, file: !1, line: 7, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{!8, !7}
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !14, type: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 8, column: 10, scope: !14)
// CHECK:STDOUT: !20 = !DILocation(line: 8, column: 3, scope: !14)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Slice.Core.84588f41d61dafba", scope: null, file: !22, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !22 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !23 = !{!24, !25}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !21, type: !7)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !21, type: !8)
// CHECK:STDOUT: !26 = !DILocation(line: 18, column: 12, scope: !21)
// CHECK:STDOUT: !27 = !DILocation(line: 18, column: 5, scope: !21)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Size", linkageName: "_CSize.Slice.Core.84588f41d61dafba", scope: null, file: !22, line: 21, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !31 = !DILocation(line: 21, column: 33, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 21, column: 26, scope: !28)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopySlice.Main(ptr sret({ ptr, i64 }) %return, ptr %s) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Slice.eb057aa32837c84e.Core:Copy.Core.073fbce68be6103e"(ptr %return, ptr %s), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Slice.eb057aa32837c84e.Core:Copy.Core.073fbce68be6103e"(ptr sret({ ptr, i64 }) %return, ptr %self) #0 !dbg !12 {
// CHECK:STDOUT:   %ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   %1 = load ptr, ptr %ptr, align 8, !dbg !16
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !17
// CHECK:STDOUT:   %2 = load i64, ptr %size, align 8, !dbg !17
// CHECK:STDOUT:   %ptr1 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   store ptr %1, ptr %ptr1, align 8, !dbg !18
// CHECK:STDOUT:   %size2 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 1, !dbg !18
// CHECK:STDOUT:   store i64 %2, ptr %size2, align 8, !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "CopySlice", linkageName: "_CCopySlice.Main", scope: null, file: !1, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 4, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 3, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Slice.eb057aa32837c84e.Core:Copy.Core.073fbce68be6103e", scope: null, file: !13, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !13 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 29, column: 42, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 29, column: 60, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 29, column: 34, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 29, column: 27, scope: !12)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'unformed.carbon'
// CHECK:STDOUT: source_filename = "unformed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUnformed.Main(ptr sret({ ptr, i64 }) %return, ptr %p, i64 %n) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca { ptr, i64 }, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !12
// CHECK:STDOUT:   call void @_CMake.Slice.Core.84588f41d61dafba(ptr %s.var, ptr %p, i64 %n), !dbg !13
// CHECK:STDOUT:   call void @"_COp.Slice.eb057aa32837c84e.Core:Copy.Core.84588f41d61dafba"(ptr %return, ptr %s.var), !dbg !14
// CHECK:STDOUT:   call void @"_COp.0d91eb108737543a:core.Destroy.Core"(ptr %s.var), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.df57f49075d1fe6b:core.Destroy.Core"(ptr %self) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.448279b82a948352:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0d91eb108737543a:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Slice.Core.84588f41d61dafba(ptr sret({ ptr, i64 }) %return, ptr %ptr, i64 %size) #0 !dbg !32 {
// CHECK:STDOUT:   %ptr1 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   store ptr %ptr, ptr %ptr1, align 8, !dbg !37
// CHECK:STDOUT:   %size2 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   store i64 %size, ptr %size2, align 8, !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Slice.eb057aa32837c84e.Core:Copy.Core.84588f41d61dafba"(ptr sret({ ptr, i64 }) %return, ptr %self) #0 !dbg !39 {
// CHECK:STDOUT:   %ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !44
// CHECK:STDOUT:   %1 = load ptr, ptr %ptr, align 8, !dbg !44
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !45
// CHECK:STDOUT:   %2 = load i64, ptr %size, align 8, !dbg !45
// CHECK:STDOUT:   %ptr1 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   store ptr %1, ptr %ptr1, align 8, !dbg !46
// CHECK:STDOUT:   %size2 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   store i64 %2, ptr %size2, align 8, !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "unformed.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Unformed", linkageName: "_CUnformed.Main", scope: null, file: !1, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !8}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !9 = !{!10, !11}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocalVariable(arg: 2, scope: !4, type: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 4, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 5, column: 7, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 6, column: 10, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Op", linkageName: "_COp.df57f49075d1fe6b:core.Destroy.Core", scope: null, file: !1, line: 4, type: !17, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null, !8}
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !16, type: !8)
// CHECK:STDOUT: !21 = !DILocation(line: 4, column: 3, scope: !16)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.448279b82a948352:core.Destroy.Core", scope: null, file: !1, line: 4, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !7}
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !22, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 4, column: 3, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0d91eb108737543a:core.Destroy.Core", scope: null, file: !1, line: 4, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !31 = !DILocation(line: 4, column: 3, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Slice.Core.84588f41d61dafba", scope: null, file: !33, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !33 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !34 = !{!35, !36}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 2, scope: !32, type: !8)
// CHECK:STDOUT: !37 = !DILocation(line: 18, column: 12, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 18, column: 5, scope: !32)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Slice.eb057aa32837c84e.Core:Copy.Core.84588f41d61dafba", scope: null, file: !33, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!7, !7}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !44 = !DILocation(line: 29, column: 42, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 60, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 29, column: 34, scope: !39)
// CHECK:STDOUT: !47 = !DILocation(line: 29, column: 27, scope: !39)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'iterate.carbon'
// CHECK:STDOUT: source_filename = "iterate.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CIterateSlice.Main(ptr %s) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %sum.var = alloca i32, align 4, !dbg !11
// CHECK:STDOUT:   %var = alloca i64, align 8, !dbg !12
// CHECK:STDOUT:   %.loc5_19.5.temp = alloca <{ i32, i1 }>, align 4, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %sum.var), !dbg !11
// CHECK:STDOUT:   store i32 0, ptr %sum.var, align 4, !dbg !11
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.call = call i64 @"_CNewCursor.Slice.929611b8ac21773a.Core:Iterate.Core.58016a73bff04416"(ptr %s), !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !12
// CHECK:STDOUT:   store i64 %Slice.as.Iterate.impl.NewCursor.call, ptr %var, align 8, !dbg !12
// CHECK:STDOUT:   br label %for.next, !dbg !12
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_19.5.temp), !dbg !12
// CHECK:STDOUT:   call void @"_CNext.Slice.929611b8ac21773a.Core:Iterate.Core.58016a73bff04416"(ptr %.loc5_19.5.temp, ptr %s, ptr %var), !dbg !12
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.8698ccdc411f2307(ptr %.loc5_19.5.temp), !dbg !12
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !12
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.8698ccdc411f2307(ptr %.loc5_19.5.temp), !dbg !12
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call = load i32, ptr %sum.var, align 4, !dbg !13
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call1 = add i32 %Int.as.AddAssignWith.impl.Op.call, %Optional.Get.call, !dbg !13
// CHECK:STDOUT:   store i32 %Int.as.AddAssignWith.impl.Op.call1, ptr %sum.var, align 4, !dbg !13
// CHECK:STDOUT:   call void @"_COp.9619ac5c2f73b9b5:core.Destroy.Core"(ptr %.loc5_19.5.temp), !dbg !12
// CHECK:STDOUT:   br label %for.next, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_COp.9619ac5c2f73b9b5:core.Destroy.Core"(ptr %.loc5_19.5.temp), !dbg !12
// CHECK:STDOUT:   call void @"_COp.df57f49075d1fe6b:core.Destroy.Core"(ptr %var), !dbg !12
// CHECK:STDOUT:   %.loc8 = load i32, ptr %sum.var, align 4, !dbg !15
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %sum.var), !dbg !11
// CHECK:STDOUT:   ret i32 %.loc8, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.1155b488c23b5086:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.830bbdcf8982aa42:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94ba1c30d421702e:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0fc9634857315db2:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.69700c1fc1b2744c:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f49c06ba556be406:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9619ac5c2f73b9b5:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.df57f49075d1fe6b:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CNewCursor.Slice.929611b8ac21773a.Core:Iterate.Core.58016a73bff04416"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT:   ret i64 0, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.Slice.929611b8ac21773a.Core:Iterate.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !59 {
// CHECK:STDOUT:   %1 = call i64 @_CSize.Slice.Core.84588f41d61dafba(ptr %self), !dbg !65
// CHECK:STDOUT:   %2 = load i64, ptr %cursor, align 8, !dbg !66
// CHECK:STDOUT:   %3 = icmp ult i64 %2, %1, !dbg !66
// CHECK:STDOUT:   br i1 %3, label %4, label %9, !dbg !67
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2"(ptr %cursor), !dbg !68
// CHECK:STDOUT:   %5 = call ptr @_CPtr.Slice.Core.84588f41d61dafba(ptr %self), !dbg !69
// CHECK:STDOUT:   %6 = load i64, ptr %cursor, align 8, !dbg !70
// CHECK:STDOUT:   %7 = sub i64 %6, 1, !dbg !70
// CHECK:STDOUT:   %pointer.offset = getelementptr inbounds i32, ptr %5, i64 %7, !dbg !71
// CHECK:STDOUT:   %8 = load i32, ptr %pointer.offset, align 4, !dbg !72
// CHECK:STDOUT:   call void @_CSome.Optional.Core.8698ccdc411f2307(ptr %return, i32 %8), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: 9:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.8698ccdc411f2307(ptr %return), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.8698ccdc411f2307(ptr %self) #0 !dbg !77 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %self), !dbg !83
// CHECK:STDOUT:   ret i1 %1, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.8698ccdc411f2307(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %self), !dbg !88
// CHECK:STDOUT:   ret i32 %1, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @_CSize.Slice.Core.84588f41d61dafba(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !94
// CHECK:STDOUT:   %1 = load i64, ptr %size, align 8, !dbg !94
// CHECK:STDOUT:   ret i64 %1, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2"(ptr %self) #0 !dbg !96 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.eacd8d9649055d24.Core:UInt.9452f4c51951679b.Core:AddAssignWith.ed21fb5387d0c258.Core.4461b628c671be10"(ptr %self, i64 1), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CPtr.Slice.Core.84588f41d61dafba(ptr %self) #0 !dbg !102 {
// CHECK:STDOUT:   %ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !105
// CHECK:STDOUT:   %1 = load ptr, ptr %ptr, align 8, !dbg !105
// CHECK:STDOUT:   ret ptr %1, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.8698ccdc411f2307(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !107 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %return, i32 %value), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.8698ccdc411f2307(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !114 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %return), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %value) #0 !dbg !119 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !122
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !122
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !122
// CHECK:STDOUT:   ret i1 %2, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %value) #0 !dbg !124 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !127
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !127
// CHECK:STDOUT:   ret i32 %1, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.eacd8d9649055d24.Core:UInt.9452f4c51951679b.Core:AddAssignWith.ed21fb5387d0c258.Core.4461b628c671be10"(ptr %self, i64 %other) #2 !dbg !129 {
// CHECK:STDOUT:   %1 = call i64 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.4474f25e3ce5c261"(i64 %other), !dbg !135
// CHECK:STDOUT:   %2 = load i64, ptr %self, align 8, !dbg !136
// CHECK:STDOUT:   %3 = add i64 %2, %1, !dbg !136
// CHECK:STDOUT:   store i64 %3, ptr %self, align 8, !dbg !136
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !137 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !140
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !140
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !141
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !141
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !143 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !144
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.4474f25e3ce5c261"(i64 %self) #0 !dbg !146 {
// CHECK:STDOUT:   %1 = call i64 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.1ead31d2f8c1fd5f"(i64 %self), !dbg !151
// CHECK:STDOUT:   ret i64 %1, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.1ead31d2f8c1fd5f"(i64 %from) #0 !dbg !153 {
// CHECK:STDOUT:   %1 = call i64 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2"(i64 %from), !dbg !156
// CHECK:STDOUT:   ret i64 %1, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2"(i64 %self) #0 !dbg !158 {
// CHECK:STDOUT:   ret i64 %self, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.9619ac5c2f73b9b5:core.Destroy.Core", { 1, 0 }
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
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "IterateSlice", linkageName: "_CIterateSlice.Main", scope: null, file: !1, line: 3, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 7, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 6, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 5, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 10, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !1, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !7}
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 5, column: 7, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94ba1c30d421702e:core.Destroy.Core", scope: null, file: !1, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !25 = !{null, !8}
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !23, type: !8)
// CHECK:STDOUT: !28 = !DILocation(line: 5, column: 7, scope: !23)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0fc9634857315db2:core.Destroy.Core", scope: null, file: !1, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !7)
// CHECK:STDOUT: !32 = !DILocation(line: 5, column: 7, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.69700c1fc1b2744c:core.Destroy.Core", scope: null, file: !1, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !8)
// CHECK:STDOUT: !36 = !DILocation(line: 5, column: 7, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f49c06ba556be406:core.Destroy.Core", scope: null, file: !1, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !8)
// CHECK:STDOUT: !40 = !DILocation(line: 5, column: 7, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9619ac5c2f73b9b5:core.Destroy.Core", scope: null, file: !1, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !8)
// CHECK:STDOUT: !44 = !DILocation(line: 5, column: 7, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.df57f49075d1fe6b:core.Destroy.Core", scope: null, file: !1, line: 5, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !49)
// CHECK:STDOUT: !46 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !47 = !{null, !48}
// CHECK:STDOUT: !48 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !45, type: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 5, column: 7, scope: !45)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.Slice.929611b8ac21773a.Core:Iterate.Core.58016a73bff04416", scope: null, file: !53, line: 89, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !56)
// CHECK:STDOUT: !53 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !54 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !55 = !{!48, !8}
// CHECK:STDOUT: !56 = !{!57}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !52, type: !8)
// CHECK:STDOUT: !58 = !DILocation(line: 89, column: 38, scope: !52)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.Slice.929611b8ac21773a.Core:Iterate.Core.58016a73bff04416", scope: null, file: !53, line: 90, type: !60, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !61 = !{!8, !8, !8}
// CHECK:STDOUT: !62 = !{!63, !64}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !59, type: !8)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !59, type: !8)
// CHECK:STDOUT: !65 = !DILocation(line: 91, column: 19, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 91, column: 9, scope: !59)
// CHECK:STDOUT: !67 = !DILocation(line: 91, column: 8, scope: !59)
// CHECK:STDOUT: !68 = !DILocation(line: 92, column: 7, scope: !59)
// CHECK:STDOUT: !69 = !DILocation(line: 93, column: 46, scope: !59)
// CHECK:STDOUT: !70 = !DILocation(line: 93, column: 58, scope: !59)
// CHECK:STDOUT: !71 = !DILocation(line: 93, column: 32, scope: !59)
// CHECK:STDOUT: !72 = !DILocation(line: 93, column: 31, scope: !59)
// CHECK:STDOUT: !73 = !DILocation(line: 93, column: 14, scope: !59)
// CHECK:STDOUT: !74 = !DILocation(line: 93, column: 7, scope: !59)
// CHECK:STDOUT: !75 = !DILocation(line: 95, column: 14, scope: !59)
// CHECK:STDOUT: !76 = !DILocation(line: 95, column: 7, scope: !59)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.8698ccdc411f2307", scope: null, file: !78, line: 36, type: !79, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !81)
// CHECK:STDOUT: !78 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !79 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !80 = !{!8, !8}
// CHECK:STDOUT: !81 = !{!82}
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !77, type: !8)
// CHECK:STDOUT: !83 = !DILocation(line: 37, column: 12, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 37, column: 5, scope: !77)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.8698ccdc411f2307", scope: null, file: !78, line: 39, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !86)
// CHECK:STDOUT: !86 = !{!87}
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !85, type: !8)
// CHECK:STDOUT: !88 = !DILocation(line: 40, column: 12, scope: !85)
// CHECK:STDOUT: !89 = !DILocation(line: 40, column: 5, scope: !85)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Size", linkageName: "_CSize.Slice.Core.84588f41d61dafba", scope: null, file: !91, line: 21, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !92)
// CHECK:STDOUT: !91 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !90, type: !8)
// CHECK:STDOUT: !94 = !DILocation(line: 21, column: 33, scope: !90)
// CHECK:STDOUT: !95 = !DILocation(line: 21, column: 26, scope: !90)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Op", linkageName: "_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2", scope: null, file: !97, line: 458, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !98)
// CHECK:STDOUT: !97 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !98 = !{!99}
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !96, type: !48)
// CHECK:STDOUT: !100 = !DILocation(line: 460, column: 5, scope: !96)
// CHECK:STDOUT: !101 = !DILocation(line: 458, column: 3, scope: !96)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Ptr", linkageName: "_CPtr.Slice.Core.84588f41d61dafba", scope: null, file: !91, line: 22, type: !79, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !103)
// CHECK:STDOUT: !103 = !{!104}
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !102, type: !8)
// CHECK:STDOUT: !105 = !DILocation(line: 22, column: 31, scope: !102)
// CHECK:STDOUT: !106 = !DILocation(line: 22, column: 24, scope: !102)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.8698ccdc411f2307", scope: null, file: !78, line: 33, type: !108, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !109 = !{!8, !7}
// CHECK:STDOUT: !110 = !{!111}
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !107, type: !7)
// CHECK:STDOUT: !112 = !DILocation(line: 34, column: 12, scope: !107)
// CHECK:STDOUT: !113 = !DILocation(line: 34, column: 5, scope: !107)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.8698ccdc411f2307", scope: null, file: !78, line: 30, type: !115, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !115 = !DISubroutineType(types: !116)
// CHECK:STDOUT: !116 = !{!8}
// CHECK:STDOUT: !117 = !DILocation(line: 31, column: 12, scope: !114)
// CHECK:STDOUT: !118 = !DILocation(line: 31, column: 5, scope: !114)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !78, line: 141, type: !79, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !120)
// CHECK:STDOUT: !120 = !{!121}
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !119, type: !8)
// CHECK:STDOUT: !122 = !DILocation(line: 142, column: 12, scope: !119)
// CHECK:STDOUT: !123 = !DILocation(line: 142, column: 5, scope: !119)
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !78, line: 144, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !125)
// CHECK:STDOUT: !125 = !{!126}
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !124, type: !8)
// CHECK:STDOUT: !127 = !DILocation(line: 145, column: 12, scope: !124)
// CHECK:STDOUT: !128 = !DILocation(line: 145, column: 5, scope: !124)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.eacd8d9649055d24.Core:UInt.9452f4c51951679b.Core:AddAssignWith.ed21fb5387d0c258.Core.4461b628c671be10", scope: null, file: !97, line: 393, type: !130, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !132)
// CHECK:STDOUT: !130 = !DISubroutineType(types: !131)
// CHECK:STDOUT: !131 = !{null, !48, !48}
// CHECK:STDOUT: !132 = !{!133, !134}
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !129, type: !48)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 2, scope: !129, type: !48)
// CHECK:STDOUT: !135 = !DILocation(line: 0, scope: !129)
// CHECK:STDOUT: !136 = !DILocation(line: 393, column: 3, scope: !129)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !78, line: 131, type: !108, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !138)
// CHECK:STDOUT: !138 = !{!139}
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !137, type: !7)
// CHECK:STDOUT: !140 = !DILocation(line: 137, column: 5, scope: !137)
// CHECK:STDOUT: !141 = !DILocation(line: 138, column: 5, scope: !137)
// CHECK:STDOUT: !142 = !DILocation(line: 139, column: 5, scope: !137)
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !78, line: 125, type: !115, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !144 = !DILocation(line: 128, column: 5, scope: !143)
// CHECK:STDOUT: !145 = !DILocation(line: 129, column: 5, scope: !143)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.4474f25e3ce5c261", scope: null, file: !97, line: 88, type: !147, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !149)
// CHECK:STDOUT: !147 = !DISubroutineType(types: !148)
// CHECK:STDOUT: !148 = !{!48, !48}
// CHECK:STDOUT: !149 = !{!150}
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !146, type: !48)
// CHECK:STDOUT: !151 = !DILocation(line: 89, column: 12, scope: !146)
// CHECK:STDOUT: !152 = !DILocation(line: 89, column: 5, scope: !146)
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.1ead31d2f8c1fd5f", scope: null, file: !97, line: 72, type: !147, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !154)
// CHECK:STDOUT: !154 = !{!155}
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !153, type: !48)
// CHECK:STDOUT: !156 = !DILocation(line: 74, column: 17, scope: !153)
// CHECK:STDOUT: !157 = !DILocation(line: 74, column: 5, scope: !153)
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2", scope: null, file: !97, line: 60, type: !147, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !159)
// CHECK:STDOUT: !159 = !{!160}
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !158, type: !48)
// CHECK:STDOUT: !161 = !DILocation(line: 60, column: 29, scope: !158)
// CHECK:STDOUT:
