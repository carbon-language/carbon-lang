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
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Slice.Core.84588f41d61dafba", scope: null, file: !22, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !22 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !23 = !{!24, !25}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !21, type: !7)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !21, type: !8)
// CHECK:STDOUT: !26 = !DILocation(line: 21, column: 12, scope: !21)
// CHECK:STDOUT: !27 = !DILocation(line: 21, column: 5, scope: !21)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Size", linkageName: "_CSize.Slice.Core.84588f41d61dafba", scope: null, file: !22, line: 24, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !31 = !DILocation(line: 24, column: 33, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 24, column: 26, scope: !28)
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Slice.eb057aa32837c84e.Core:Copy.Core.073fbce68be6103e", scope: null, file: !13, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
// CHECK:STDOUT: !13 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 32, column: 42, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 32, column: 60, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 32, column: 34, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 32, column: 27, scope: !12)
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
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Slice.Core.84588f41d61dafba", scope: null, file: !33, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !33 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !34 = !{!35, !36}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 2, scope: !32, type: !8)
// CHECK:STDOUT: !37 = !DILocation(line: 21, column: 12, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 21, column: 5, scope: !32)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Slice.eb057aa32837c84e.Core:Copy.Core.84588f41d61dafba", scope: null, file: !33, line: 32, type: !40, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!7, !7}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !44 = !DILocation(line: 32, column: 42, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 32, column: 60, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 32, column: 34, scope: !39)
// CHECK:STDOUT: !47 = !DILocation(line: 32, column: 27, scope: !39)
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
// CHECK:STDOUT:   %1 = alloca i64, align 8, !dbg !65
// CHECK:STDOUT:   %2 = call i64 @_CSize.Slice.Core.84588f41d61dafba(ptr %self), !dbg !66
// CHECK:STDOUT:   %3 = load i64, ptr %cursor, align 8, !dbg !67
// CHECK:STDOUT:   %4 = icmp ult i64 %3, %2, !dbg !67
// CHECK:STDOUT:   br i1 %4, label %5, label %10, !dbg !68
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %0
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !65
// CHECK:STDOUT:   %6 = load i64, ptr %cursor, align 8, !dbg !69
// CHECK:STDOUT:   store i64 %6, ptr %1, align 8, !dbg !65
// CHECK:STDOUT:   call void @"_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2"(ptr %cursor), !dbg !70
// CHECK:STDOUT:   %7 = call ptr @_CPtr.Slice.Core.84588f41d61dafba(ptr %self), !dbg !71
// CHECK:STDOUT:   %8 = load i64, ptr %1, align 8, !dbg !72
// CHECK:STDOUT:   %pointer.offset = getelementptr inbounds i32, ptr %7, i64 %8, !dbg !73
// CHECK:STDOUT:   %9 = load i32, ptr %pointer.offset, align 4, !dbg !74
// CHECK:STDOUT:   call void @_CSome.Optional.Core.8698ccdc411f2307(ptr %return, i32 %9), !dbg !75
// CHECK:STDOUT:   call void @"_COp.df57f49075d1fe6b:core.Destroy.Core"(ptr %1), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT:
// CHECK:STDOUT: 10:                                               ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.8698ccdc411f2307(ptr %return), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.8698ccdc411f2307(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %self), !dbg !85
// CHECK:STDOUT:   ret i1 %1, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.8698ccdc411f2307(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %self), !dbg !90
// CHECK:STDOUT:   ret i32 %1, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @_CSize.Slice.Core.84588f41d61dafba(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !96
// CHECK:STDOUT:   %1 = load i64, ptr %size, align 8, !dbg !96
// CHECK:STDOUT:   ret i64 %1, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2"(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.eacd8d9649055d24.Core:UInt.9452f4c51951679b.Core:AddAssignWith.ed21fb5387d0c258.Core.4461b628c671be10"(ptr %self, i64 1), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CPtr.Slice.Core.84588f41d61dafba(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT:   %ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !107
// CHECK:STDOUT:   %1 = load ptr, ptr %ptr, align 8, !dbg !107
// CHECK:STDOUT:   ret ptr %1, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.8698ccdc411f2307(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !109 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %return, i32 %value), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.8698ccdc411f2307(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !116 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %return), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %value) #0 !dbg !121 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !124
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !124
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !124
// CHECK:STDOUT:   ret i1 %2, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr %value) #0 !dbg !126 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !129
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !129
// CHECK:STDOUT:   ret i32 %1, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.eacd8d9649055d24.Core:UInt.9452f4c51951679b.Core:AddAssignWith.ed21fb5387d0c258.Core.4461b628c671be10"(ptr %self, i64 %other) #2 !dbg !131 {
// CHECK:STDOUT:   %1 = call i64 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.4474f25e3ce5c261"(i64 %other), !dbg !137
// CHECK:STDOUT:   %2 = load i64, ptr %self, align 8, !dbg !138
// CHECK:STDOUT:   %3 = add i64 %2, %1, !dbg !138
// CHECK:STDOUT:   store i64 %3, ptr %self, align 8, !dbg !138
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !139 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !142
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !142
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !143
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !145 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !146
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !146
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.4474f25e3ce5c261"(i64 %self) #0 !dbg !148 {
// CHECK:STDOUT:   %1 = call i64 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.1ead31d2f8c1fd5f"(i64 %self), !dbg !153
// CHECK:STDOUT:   ret i64 %1, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.1ead31d2f8c1fd5f"(i64 %from) #0 !dbg !155 {
// CHECK:STDOUT:   %1 = call i64 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2"(i64 %from), !dbg !158
// CHECK:STDOUT:   ret i64 %1, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2"(i64 %self) #0 !dbg !160 {
// CHECK:STDOUT:   ret i64 %self, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.9619ac5c2f73b9b5:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
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
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.Slice.929611b8ac21773a.Core:Iterate.Core.58016a73bff04416", scope: null, file: !53, line: 42, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !56)
// CHECK:STDOUT: !53 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !54 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !55 = !{!48, !8}
// CHECK:STDOUT: !56 = !{!57}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !52, type: !8)
// CHECK:STDOUT: !58 = !DILocation(line: 42, column: 38, scope: !52)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.Slice.929611b8ac21773a.Core:Iterate.Core.58016a73bff04416", scope: null, file: !53, line: 43, type: !60, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !61 = !{!8, !8, !8}
// CHECK:STDOUT: !62 = !{!63, !64}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !59, type: !8)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !59, type: !8)
// CHECK:STDOUT: !65 = !DILocation(line: 45, column: 7, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 44, column: 19, scope: !59)
// CHECK:STDOUT: !67 = !DILocation(line: 44, column: 9, scope: !59)
// CHECK:STDOUT: !68 = !DILocation(line: 44, column: 8, scope: !59)
// CHECK:STDOUT: !69 = !DILocation(line: 45, column: 25, scope: !59)
// CHECK:STDOUT: !70 = !DILocation(line: 46, column: 7, scope: !59)
// CHECK:STDOUT: !71 = !DILocation(line: 47, column: 46, scope: !59)
// CHECK:STDOUT: !72 = !DILocation(line: 47, column: 58, scope: !59)
// CHECK:STDOUT: !73 = !DILocation(line: 47, column: 32, scope: !59)
// CHECK:STDOUT: !74 = !DILocation(line: 47, column: 31, scope: !59)
// CHECK:STDOUT: !75 = !DILocation(line: 47, column: 14, scope: !59)
// CHECK:STDOUT: !76 = !DILocation(line: 47, column: 7, scope: !59)
// CHECK:STDOUT: !77 = !DILocation(line: 49, column: 14, scope: !59)
// CHECK:STDOUT: !78 = !DILocation(line: 49, column: 7, scope: !59)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.8698ccdc411f2307", scope: null, file: !80, line: 36, type: !81, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !83)
// CHECK:STDOUT: !80 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !81 = !DISubroutineType(types: !82)
// CHECK:STDOUT: !82 = !{!8, !8}
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !79, type: !8)
// CHECK:STDOUT: !85 = !DILocation(line: 37, column: 12, scope: !79)
// CHECK:STDOUT: !86 = !DILocation(line: 37, column: 5, scope: !79)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.8698ccdc411f2307", scope: null, file: !80, line: 39, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !8)
// CHECK:STDOUT: !90 = !DILocation(line: 40, column: 12, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 40, column: 5, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Size", linkageName: "_CSize.Slice.Core.84588f41d61dafba", scope: null, file: !93, line: 24, type: !54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !94)
// CHECK:STDOUT: !93 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !94 = !{!95}
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !92, type: !8)
// CHECK:STDOUT: !96 = !DILocation(line: 24, column: 33, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 24, column: 26, scope: !92)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Op", linkageName: "_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2", scope: null, file: !99, line: 458, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !100)
// CHECK:STDOUT: !99 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !100 = !{!101}
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !98, type: !48)
// CHECK:STDOUT: !102 = !DILocation(line: 460, column: 5, scope: !98)
// CHECK:STDOUT: !103 = !DILocation(line: 458, column: 3, scope: !98)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Ptr", linkageName: "_CPtr.Slice.Core.84588f41d61dafba", scope: null, file: !93, line: 25, type: !81, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !105)
// CHECK:STDOUT: !105 = !{!106}
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !104, type: !8)
// CHECK:STDOUT: !107 = !DILocation(line: 25, column: 31, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 25, column: 24, scope: !104)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.8698ccdc411f2307", scope: null, file: !80, line: 33, type: !110, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DISubroutineType(types: !111)
// CHECK:STDOUT: !111 = !{!8, !7}
// CHECK:STDOUT: !112 = !{!113}
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !114 = !DILocation(line: 34, column: 12, scope: !109)
// CHECK:STDOUT: !115 = !DILocation(line: 34, column: 5, scope: !109)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.8698ccdc411f2307", scope: null, file: !80, line: 30, type: !117, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !117 = !DISubroutineType(types: !118)
// CHECK:STDOUT: !118 = !{!8}
// CHECK:STDOUT: !119 = !DILocation(line: 31, column: 12, scope: !116)
// CHECK:STDOUT: !120 = !DILocation(line: 31, column: 5, scope: !116)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !80, line: 141, type: !81, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !122)
// CHECK:STDOUT: !122 = !{!123}
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !121, type: !8)
// CHECK:STDOUT: !124 = !DILocation(line: 142, column: 12, scope: !121)
// CHECK:STDOUT: !125 = !DILocation(line: 142, column: 5, scope: !121)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !80, line: 144, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !127)
// CHECK:STDOUT: !127 = !{!128}
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !126, type: !8)
// CHECK:STDOUT: !129 = !DILocation(line: 145, column: 12, scope: !126)
// CHECK:STDOUT: !130 = !DILocation(line: 145, column: 5, scope: !126)
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.eacd8d9649055d24.Core:UInt.9452f4c51951679b.Core:AddAssignWith.ed21fb5387d0c258.Core.4461b628c671be10", scope: null, file: !99, line: 393, type: !132, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DISubroutineType(types: !133)
// CHECK:STDOUT: !133 = !{null, !48, !48}
// CHECK:STDOUT: !134 = !{!135, !136}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !131, type: !48)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 2, scope: !131, type: !48)
// CHECK:STDOUT: !137 = !DILocation(line: 0, scope: !131)
// CHECK:STDOUT: !138 = !DILocation(line: 393, column: 3, scope: !131)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !80, line: 131, type: !110, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !140)
// CHECK:STDOUT: !140 = !{!141}
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !139, type: !7)
// CHECK:STDOUT: !142 = !DILocation(line: 137, column: 5, scope: !139)
// CHECK:STDOUT: !143 = !DILocation(line: 138, column: 5, scope: !139)
// CHECK:STDOUT: !144 = !DILocation(line: 139, column: 5, scope: !139)
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.58016a73bff04416", scope: null, file: !80, line: 125, type: !117, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !146 = !DILocation(line: 128, column: 5, scope: !145)
// CHECK:STDOUT: !147 = !DILocation(line: 129, column: 5, scope: !145)
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.4474f25e3ce5c261", scope: null, file: !99, line: 88, type: !149, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !151)
// CHECK:STDOUT: !149 = !DISubroutineType(types: !150)
// CHECK:STDOUT: !150 = !{!48, !48}
// CHECK:STDOUT: !151 = !{!152}
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !148, type: !48)
// CHECK:STDOUT: !153 = !DILocation(line: 89, column: 12, scope: !148)
// CHECK:STDOUT: !154 = !DILocation(line: 89, column: 5, scope: !148)
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.1ead31d2f8c1fd5f", scope: null, file: !99, line: 72, type: !149, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !156)
// CHECK:STDOUT: !156 = !{!157}
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !155, type: !48)
// CHECK:STDOUT: !158 = !DILocation(line: 74, column: 17, scope: !155)
// CHECK:STDOUT: !159 = !DILocation(line: 74, column: 5, scope: !155)
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2", scope: null, file: !99, line: 60, type: !149, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !161)
// CHECK:STDOUT: !161 = !{!162}
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !160, type: !48)
// CHECK:STDOUT: !163 = !DILocation(line: 60, column: 29, scope: !160)
// CHECK:STDOUT:
