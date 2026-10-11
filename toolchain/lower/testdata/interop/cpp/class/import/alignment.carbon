// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/alignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/alignment.carbon

// --- alignment.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
struct NaturallyAligned {
  int n;
};

struct alignas(16) ExtraAlignment {
  int n;
};

struct __attribute__((packed)) Packed {
  int n;
};

template<typename T>
void Take(T&);
''';

fn NaturallyAligned(n: i32) {
  var a: Cpp.NaturallyAligned;
  a.n = n;
  Cpp.Take(ref a);
}

fn ExtraAlignment(n: i32) {
  var a: Cpp.ExtraAlignment;
  // TODO: We only specify a 4-byte alignment here, although we know that `a.n`
  // has more alignment.
  a.n = n;
  Cpp.Take(ref a);
}

fn Packed(n: i32) {
  var a: Cpp.Packed;
  // TODO: We incorrectly specify a 4-byte alignment here. We may need to give
  // underaligned fields a different type in order to prevent miscompiling code
  // like this.
  a.n = n;
  Cpp.Take(ref a);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'alignment.carbon'
// CHECK:STDOUT: source_filename = "alignment.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CNaturallyAligned.Main(i32 %n) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 4, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !47
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.d4cda66b3640b619"(ptr %a.var), !dbg !47
// CHECK:STDOUT:   %.loc25.n = getelementptr inbounds nuw [4 x i8], ptr %a.var, i32 0, i32 0, !dbg !49
// CHECK:STDOUT:   store i32 %n, ptr %.loc25.n, align 4, !dbg !49
// CHECK:STDOUT:   call void @_Z4TakeI16NaturallyAlignedEvRT_(ptr %a.var), !dbg !53
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.75a2b523a742a601(ptr %a.var), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN16NaturallyAlignedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !141
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !141
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.c846b89dd79c7366.Core:NaturallyAligned.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN16NaturallyAlignedC1Ev.carbon_thunk.(ptr %return), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI16NaturallyAlignedEvRT_(ptr noundef nonnull align 4 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.42caf9316ea28965.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CExtraAlignment.Main(i32 %n) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [16 x i8], align 16, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !66
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645"(ptr %a.var), !dbg !66
// CHECK:STDOUT:   %.loc33.n = getelementptr inbounds nuw [16 x i8], ptr %a.var, i32 0, i32 0, !dbg !68
// CHECK:STDOUT:   store i32 %n, ptr %.loc33.n, align 4, !dbg !68
// CHECK:STDOUT:   call void @_Z4TakeI14ExtraAlignmentEvRT_(ptr %a.var), !dbg !72
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.10abfe78d01d932a(ptr %a.var), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN14ExtraAlignmentC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !143
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !143
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp"(ptr sret([16 x i8]) %return) #2 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN14ExtraAlignmentC1Ev.carbon_thunk.(ptr %return), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI14ExtraAlignmentEvRT_(ptr noundef nonnull align 16 dereferenceable(16)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.44310a9efdd79817.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPacked.Main(i32 %n) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 1, !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !85
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a"(ptr %a.var), !dbg !85
// CHECK:STDOUT:   %.loc42.n = getelementptr inbounds nuw [4 x i8], ptr %a.var, i32 0, i32 0, !dbg !87
// CHECK:STDOUT:   store i32 %n, ptr %.loc42.n, align 4, !dbg !87
// CHECK:STDOUT:   call void @_Z4TakeI6PackedEvRT_(ptr %a.var), !dbg !91
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.074b1e70fd3297fd(ptr %a.var), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6PackedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !145
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !145
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6PackedC1Ev.carbon_thunk.(ptr %return), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI6PackedEvRT_(ptr noundef nonnull align 1 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.3ef8b760b7b5d645.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.d4cda66b3640b619"(ptr sret([4 x i8]) %return) #0 !dbg !101 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.c846b89dd79c7366.Core:NaturallyAligned.Cpp"(ptr %return), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.75a2b523a742a601(ptr %t) #0 !dbg !106 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.42caf9316ea28965.Core:DestroyImpl.Core"(ptr %t), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645"(ptr sret([16 x i8]) %return) #0 !dbg !113 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp"(ptr %return), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.10abfe78d01d932a(ptr %t) #0 !dbg !118 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.44310a9efdd79817.Core:DestroyImpl.Core"(ptr %t), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a"(ptr sret([4 x i8]) %return) #0 !dbg !125 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp"(ptr %return), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.074b1e70fd3297fd(ptr %t) #0 !dbg !130 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.3ef8b760b7b5d645.Core:DestroyImpl.Core"(ptr %t), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!146, !147, !148, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!151}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "alignment.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{!39}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "NaturallyAligned", linkageName: "_CNaturallyAligned.Main", scope: null, file: !34, line: 23, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !55)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !36)
// CHECK:STDOUT: !47 = !DILocation(line: 24, column: 3, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 25, column: 3, scope: !44)
// CHECK:STDOUT: !53 = !DILocation(line: 26, column: 3, scope: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 23, column: 1, scope: !44)
// CHECK:STDOUT: !55 = !{!45}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.c846b89dd79c7366.Core:NaturallyAligned.Cpp", scope: null, file: !34, line: 7, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 8, scope: !56)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.42caf9316ea28965.Core:DestroyImpl.Core", scope: null, file: !34, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !39)
// CHECK:STDOUT: !61 = !DILocation(line: 0, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "ExtraAlignment", linkageName: "_CExtraAlignment.Main", scope: null, file: !34, line: 29, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !74)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !36)
// CHECK:STDOUT: !66 = !DILocation(line: 30, column: 3, scope: !63)
// CHECK:STDOUT: !68 = !DILocation(line: 33, column: 3, scope: !63)
// CHECK:STDOUT: !72 = !DILocation(line: 34, column: 3, scope: !63)
// CHECK:STDOUT: !73 = !DILocation(line: 29, column: 1, scope: !63)
// CHECK:STDOUT: !74 = !{!64}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp", scope: null, file: !34, line: 11, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !77 = !DILocation(line: 11, column: 20, scope: !75)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.44310a9efdd79817.Core:DestroyImpl.Core", scope: null, file: !34, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !39)
// CHECK:STDOUT: !80 = !DILocation(line: 0, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Packed", linkageName: "_CPacked.Main", scope: null, file: !34, line: 37, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !93)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !36)
// CHECK:STDOUT: !85 = !DILocation(line: 38, column: 3, scope: !82)
// CHECK:STDOUT: !87 = !DILocation(line: 42, column: 3, scope: !82)
// CHECK:STDOUT: !91 = !DILocation(line: 43, column: 3, scope: !82)
// CHECK:STDOUT: !92 = !DILocation(line: 37, column: 1, scope: !82)
// CHECK:STDOUT: !93 = !{!83}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp", scope: null, file: !34, line: 15, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !96 = !DILocation(line: 15, column: 32, scope: !94)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.3ef8b760b7b5d645.Core:DestroyImpl.Core", scope: null, file: !34, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !39)
// CHECK:STDOUT: !99 = !DILocation(line: 0, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.d4cda66b3640b619", scope: null, file: !24, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !104 = !DILocation(line: 9, column: 28, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 9, column: 21, scope: !101)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.75a2b523a742a601", scope: null, file: !28, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !112)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !39)
// CHECK:STDOUT: !109 = !DILocation(line: 27, column: 5, scope: !106)
// CHECK:STDOUT: !111 = !DILocation(line: 26, column: 1, scope: !106)
// CHECK:STDOUT: !112 = !{!107}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645", scope: null, file: !24, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !116 = !DILocation(line: 9, column: 28, scope: !113)
// CHECK:STDOUT: !117 = !DILocation(line: 9, column: 21, scope: !113)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.10abfe78d01d932a", scope: null, file: !28, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !39)
// CHECK:STDOUT: !121 = !DILocation(line: 27, column: 5, scope: !118)
// CHECK:STDOUT: !123 = !DILocation(line: 26, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a", scope: null, file: !24, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !128 = !DILocation(line: 9, column: 28, scope: !125)
// CHECK:STDOUT: !129 = !DILocation(line: 9, column: 21, scope: !125)
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.074b1e70fd3297fd", scope: null, file: !28, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !136)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !39)
// CHECK:STDOUT: !133 = !DILocation(line: 27, column: 5, scope: !130)
// CHECK:STDOUT: !135 = !DILocation(line: 26, column: 1, scope: !130)
// CHECK:STDOUT: !136 = !{!131}
// CHECK:STDOUT: !137 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !138 = !{!"omnipotent char", !137, i64 0}
// CHECK:STDOUT: !139 = !{!"any pointer", !138, i64 0}
// CHECK:STDOUT: !140 = !{!"p1 _ZTS16NaturallyAligned", !139, i64 0}
// CHECK:STDOUT: !141 = !{!140, !140, i64 0}
// CHECK:STDOUT: !142 = !{!"p1 _ZTS14ExtraAlignment", !139, i64 0}
// CHECK:STDOUT: !143 = !{!142, !142, i64 0}
// CHECK:STDOUT: !144 = !{!"p1 _ZTS6Packed", !139, i64 0}
// CHECK:STDOUT: !145 = !{!144, !144, i64 0}
// CHECK:STDOUT: !146 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !147 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !148 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !149 = !{!"int", !138, i64 0}
// CHECK:STDOUT: !150 = !{!"__libc_errno", !149, i64 0}
// CHECK:STDOUT: !151 = !{!150, !149, i64 0}
// CHECK:STDOUT:
