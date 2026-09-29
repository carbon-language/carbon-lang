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
// CHECK:STDOUT: define void @_CNaturallyAligned.Main(i32 %n) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 4, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !45
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.d4cda66b3640b619"(ptr %a.var), !dbg !45
// CHECK:STDOUT:   %.loc25.n = getelementptr inbounds nuw [4 x i8], ptr %a.var, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   store i32 %n, ptr %.loc25.n, align 4, !dbg !47
// CHECK:STDOUT:   call void @_Z4TakeI16NaturallyAlignedEvRT_(ptr %a.var), !dbg !51
// CHECK:STDOUT:   call void @"_CSelfDestruct.89af5f7a0692dfe8:core.Destroy.Core"(ptr %a.var), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN16NaturallyAlignedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !118
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !118
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.c846b89dd79c7366.Core:NaturallyAligned.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN16NaturallyAlignedC1Ev.carbon_thunk.(ptr %return), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI16NaturallyAlignedEvRT_(ptr noundef nonnull align 4 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.89af5f7a0692dfe8:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CExtraAlignment.Main(i32 %n) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [16 x i8], align 16, !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !64
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645"(ptr %a.var), !dbg !64
// CHECK:STDOUT:   %.loc33.n = getelementptr inbounds nuw [16 x i8], ptr %a.var, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   store i32 %n, ptr %.loc33.n, align 4, !dbg !66
// CHECK:STDOUT:   call void @_Z4TakeI14ExtraAlignmentEvRT_(ptr %a.var), !dbg !70
// CHECK:STDOUT:   call void @"_CSelfDestruct.77c39a3d53f40ae2:core.Destroy.Core"(ptr %a.var), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN14ExtraAlignmentC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp"(ptr sret([16 x i8]) %return) #2 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN14ExtraAlignmentC1Ev.carbon_thunk.(ptr %return), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI14ExtraAlignmentEvRT_(ptr noundef nonnull align 16 dereferenceable(16)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.77c39a3d53f40ae2:core.Destroy.Core"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPacked.Main(i32 %n) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 1, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !83
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a"(ptr %a.var), !dbg !83
// CHECK:STDOUT:   %.loc42.n = getelementptr inbounds nuw [4 x i8], ptr %a.var, i32 0, i32 0, !dbg !85
// CHECK:STDOUT:   store i32 %n, ptr %.loc42.n, align 4, !dbg !85
// CHECK:STDOUT:   call void @_Z4TakeI6PackedEvRT_(ptr %a.var), !dbg !89
// CHECK:STDOUT:   call void @"_CSelfDestruct.00b787a687567d9a:core.Destroy.Core"(ptr %a.var), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6PackedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !122
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !122
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6PackedC1Ev.carbon_thunk.(ptr %return), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI6PackedEvRT_(ptr noundef nonnull align 1 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.00b787a687567d9a:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.d4cda66b3640b619"(ptr sret([4 x i8]) %return) #0 !dbg !99 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.c846b89dd79c7366.Core:NaturallyAligned.Cpp"(ptr %return), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645"(ptr sret([16 x i8]) %return) #0 !dbg !104 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp"(ptr %return), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a"(ptr sret([4 x i8]) %return) #0 !dbg !109 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp"(ptr %return), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!123, !124, !125, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!128}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !32 = !DIFile(filename: "alignment.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{null, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{!37}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "NaturallyAligned", linkageName: "_CNaturallyAligned.Main", scope: null, file: !32, line: 23, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !53)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !34)
// CHECK:STDOUT: !45 = !DILocation(line: 24, column: 3, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 25, column: 3, scope: !42)
// CHECK:STDOUT: !51 = !DILocation(line: 26, column: 3, scope: !42)
// CHECK:STDOUT: !52 = !DILocation(line: 23, column: 1, scope: !42)
// CHECK:STDOUT: !53 = !{!43}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.c846b89dd79c7366.Core:NaturallyAligned.Cpp", scope: null, file: !32, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !56 = !DILocation(line: 7, column: 8, scope: !54)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.89af5f7a0692dfe8:core.Destroy.Core", scope: null, file: !32, line: 24, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !37)
// CHECK:STDOUT: !59 = !DILocation(line: 24, column: 3, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "ExtraAlignment", linkageName: "_CExtraAlignment.Main", scope: null, file: !32, line: 29, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !72)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !34)
// CHECK:STDOUT: !64 = !DILocation(line: 30, column: 3, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 33, column: 3, scope: !61)
// CHECK:STDOUT: !70 = !DILocation(line: 34, column: 3, scope: !61)
// CHECK:STDOUT: !71 = !DILocation(line: 29, column: 1, scope: !61)
// CHECK:STDOUT: !72 = !{!62}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp", scope: null, file: !32, line: 11, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !75 = !DILocation(line: 11, column: 20, scope: !73)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.77c39a3d53f40ae2:core.Destroy.Core", scope: null, file: !32, line: 30, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !79)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !37)
// CHECK:STDOUT: !78 = !DILocation(line: 30, column: 3, scope: !76)
// CHECK:STDOUT: !79 = !{!77}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Packed", linkageName: "_CPacked.Main", scope: null, file: !32, line: 37, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !91)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !34)
// CHECK:STDOUT: !83 = !DILocation(line: 38, column: 3, scope: !80)
// CHECK:STDOUT: !85 = !DILocation(line: 42, column: 3, scope: !80)
// CHECK:STDOUT: !89 = !DILocation(line: 43, column: 3, scope: !80)
// CHECK:STDOUT: !90 = !DILocation(line: 37, column: 1, scope: !80)
// CHECK:STDOUT: !91 = !{!81}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp", scope: null, file: !32, line: 15, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !94 = !DILocation(line: 15, column: 32, scope: !92)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.00b787a687567d9a:core.Destroy.Core", scope: null, file: !32, line: 38, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !37)
// CHECK:STDOUT: !97 = !DILocation(line: 38, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.d4cda66b3640b619", scope: null, file: !26, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !102 = !DILocation(line: 9, column: 28, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 9, column: 21, scope: !99)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645", scope: null, file: !26, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !107 = !DILocation(line: 9, column: 28, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 9, column: 21, scope: !104)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a", scope: null, file: !26, line: 9, type: !41, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !112 = !DILocation(line: 9, column: 28, scope: !109)
// CHECK:STDOUT: !113 = !DILocation(line: 9, column: 21, scope: !109)
// CHECK:STDOUT: !114 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !115 = !{!"omnipotent char", !114, i64 0}
// CHECK:STDOUT: !116 = !{!"any pointer", !115, i64 0}
// CHECK:STDOUT: !117 = !{!"p1 _ZTS16NaturallyAligned", !116, i64 0}
// CHECK:STDOUT: !118 = !{!117, !117, i64 0}
// CHECK:STDOUT: !119 = !{!"p1 _ZTS14ExtraAlignment", !116, i64 0}
// CHECK:STDOUT: !120 = !{!119, !119, i64 0}
// CHECK:STDOUT: !121 = !{!"p1 _ZTS6Packed", !116, i64 0}
// CHECK:STDOUT: !122 = !{!121, !121, i64 0}
// CHECK:STDOUT: !123 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !124 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !125 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !126 = !{!"int", !115, i64 0}
// CHECK:STDOUT: !127 = !{!"__libc_errno", !126, i64 0}
// CHECK:STDOUT: !128 = !{!127, !126, i64 0}
// CHECK:STDOUT:
