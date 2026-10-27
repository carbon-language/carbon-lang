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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9098afc6c06847a6(ptr %a.var), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN16NaturallyAlignedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !129
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !129
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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CExtraAlignment.Main(i32 %n) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [16 x i8], align 16, !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !62
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645"(ptr %a.var), !dbg !62
// CHECK:STDOUT:   %.loc33.n = getelementptr inbounds nuw [16 x i8], ptr %a.var, i32 0, i32 0, !dbg !64
// CHECK:STDOUT:   store i32 %n, ptr %.loc33.n, align 4, !dbg !64
// CHECK:STDOUT:   call void @_Z4TakeI14ExtraAlignmentEvRT_(ptr %a.var), !dbg !68
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.473a34350c101836(ptr %a.var), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN14ExtraAlignmentC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !131
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !131
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp"(ptr sret([16 x i8]) %return) #2 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN14ExtraAlignmentC1Ev.carbon_thunk.(ptr %return), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI14ExtraAlignmentEvRT_(ptr noundef nonnull align 16 dereferenceable(16)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPacked.Main(i32 %n) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 1, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !77
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a"(ptr %a.var), !dbg !77
// CHECK:STDOUT:   %.loc42.n = getelementptr inbounds nuw [4 x i8], ptr %a.var, i32 0, i32 0, !dbg !79
// CHECK:STDOUT:   store i32 %n, ptr %.loc42.n, align 4, !dbg !79
// CHECK:STDOUT:   call void @_Z4TakeI6PackedEvRT_(ptr %a.var), !dbg !83
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.327908d682917e7f(ptr %a.var), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6PackedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !133
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !133
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6PackedC1Ev.carbon_thunk.(ptr %return), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI6PackedEvRT_(ptr noundef nonnull align 1 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.d4cda66b3640b619"(ptr sret([4 x i8]) %return) #0 !dbg !89 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.c846b89dd79c7366.Core:NaturallyAligned.Cpp"(ptr %return), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9098afc6c06847a6(ptr %t) #0 !dbg !94 {
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645"(ptr sret([16 x i8]) %return) #0 !dbg !101 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp"(ptr %return), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.473a34350c101836(ptr %t) #0 !dbg !106 {
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a"(ptr sret([4 x i8]) %return) #0 !dbg !113 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp"(ptr %return), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.327908d682917e7f(ptr %t) #0 !dbg !118 {
// CHECK:STDOUT:   ret void, !dbg !123
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
// CHECK:STDOUT: !llvm.module.flags = !{!134, !135, !136, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!139}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "ExtraAlignment", linkageName: "_CExtraAlignment.Main", scope: null, file: !34, line: 29, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !70)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !36)
// CHECK:STDOUT: !62 = !DILocation(line: 30, column: 3, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 33, column: 3, scope: !59)
// CHECK:STDOUT: !68 = !DILocation(line: 34, column: 3, scope: !59)
// CHECK:STDOUT: !69 = !DILocation(line: 29, column: 1, scope: !59)
// CHECK:STDOUT: !70 = !{!60}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.52e7708693d29223.Core:ExtraAlignment.Cpp", scope: null, file: !34, line: 11, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !73 = !DILocation(line: 11, column: 20, scope: !71)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Packed", linkageName: "_CPacked.Main", scope: null, file: !34, line: 37, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !85)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !36)
// CHECK:STDOUT: !77 = !DILocation(line: 38, column: 3, scope: !74)
// CHECK:STDOUT: !79 = !DILocation(line: 42, column: 3, scope: !74)
// CHECK:STDOUT: !83 = !DILocation(line: 43, column: 3, scope: !74)
// CHECK:STDOUT: !84 = !DILocation(line: 37, column: 1, scope: !74)
// CHECK:STDOUT: !85 = !{!75}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.ca2b917a39954a06.Core:Packed.Cpp", scope: null, file: !34, line: 15, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !88 = !DILocation(line: 15, column: 32, scope: !86)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.d4cda66b3640b619", scope: null, file: !22, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !92 = !DILocation(line: 9, column: 28, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 9, column: 21, scope: !89)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9098afc6c06847a6", scope: null, file: !26, line: 32, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !100)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !39)
// CHECK:STDOUT: !99 = !DILocation(line: 32, column: 1, scope: !94)
// CHECK:STDOUT: !100 = !{!95}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.34933c1c343bd645", scope: null, file: !22, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !104 = !DILocation(line: 9, column: 28, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 9, column: 21, scope: !101)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.473a34350c101836", scope: null, file: !26, line: 32, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !112)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !39)
// CHECK:STDOUT: !111 = !DILocation(line: 32, column: 1, scope: !106)
// CHECK:STDOUT: !112 = !{!107}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.3e10857902c69c1a", scope: null, file: !22, line: 9, type: !43, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !116 = !DILocation(line: 9, column: 28, scope: !113)
// CHECK:STDOUT: !117 = !DILocation(line: 9, column: 21, scope: !113)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.327908d682917e7f", scope: null, file: !26, line: 32, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !39)
// CHECK:STDOUT: !123 = !DILocation(line: 32, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !126 = !{!"omnipotent char", !125, i64 0}
// CHECK:STDOUT: !127 = !{!"any pointer", !126, i64 0}
// CHECK:STDOUT: !128 = !{!"p1 _ZTS16NaturallyAligned", !127, i64 0}
// CHECK:STDOUT: !129 = !{!128, !128, i64 0}
// CHECK:STDOUT: !130 = !{!"p1 _ZTS14ExtraAlignment", !127, i64 0}
// CHECK:STDOUT: !131 = !{!130, !130, i64 0}
// CHECK:STDOUT: !132 = !{!"p1 _ZTS6Packed", !127, i64 0}
// CHECK:STDOUT: !133 = !{!132, !132, i64 0}
// CHECK:STDOUT: !134 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !135 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !136 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !137 = !{!"int", !126, i64 0}
// CHECK:STDOUT: !138 = !{!"__libc_errno", !137, i64 0}
// CHECK:STDOUT: !139 = !{!138, !137, i64 0}
// CHECK:STDOUT:
