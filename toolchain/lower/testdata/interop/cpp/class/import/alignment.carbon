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

// CHECK:STDOUT: ; ModuleID = 'alignment.carbon'
// CHECK:STDOUT: source_filename = "alignment.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CNaturallyAligned.Main(i32 %n) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 4, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !17
// CHECK:STDOUT:   call void @"_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.53704887462f6065"(ptr %a.var), !dbg !17
// CHECK:STDOUT:   %.loc25.n = getelementptr inbounds nuw [4 x i8], ptr %a.var, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   store i32 %n, ptr %.loc25.n, align 4, !dbg !18
// CHECK:STDOUT:   call void @_Z4TakeI16NaturallyAlignedEvRT_(ptr %a.var), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN16NaturallyAlignedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !21
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !21
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Default.381601f5a2bd2e81.Core:NaturallyAligned.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN16NaturallyAlignedC1Ev.carbon_thunk.(ptr %return), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI16NaturallyAlignedEvRT_(ptr noundef nonnull align 4 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CExtraAlignment.Main(i32 %n) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [16 x i8], align 16, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !32
// CHECK:STDOUT:   call void @"_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.f529dae788690cbe"(ptr %a.var), !dbg !32
// CHECK:STDOUT:   %.loc33.n = getelementptr inbounds nuw [16 x i8], ptr %a.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   store i32 %n, ptr %.loc33.n, align 4, !dbg !33
// CHECK:STDOUT:   call void @_Z4TakeI14ExtraAlignmentEvRT_(ptr %a.var), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN14ExtraAlignmentC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !36
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !36
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Default.6ca50d719a15fa24.Core:ExtraAlignment.Cpp"(ptr sret([16 x i8]) %return) #2 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN14ExtraAlignmentC1Ev.carbon_thunk.(ptr %return), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI14ExtraAlignmentEvRT_(ptr noundef nonnull align 16 dereferenceable(16)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPacked.Main(i32 %n) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 1, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !43
// CHECK:STDOUT:   call void @"_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.ee398c9a806dbb35"(ptr %a.var), !dbg !43
// CHECK:STDOUT:   %.loc42.n = getelementptr inbounds nuw [4 x i8], ptr %a.var, i32 0, i32 0, !dbg !44
// CHECK:STDOUT:   store i32 %n, ptr %.loc42.n, align 4, !dbg !44
// CHECK:STDOUT:   call void @_Z4TakeI6PackedEvRT_(ptr %a.var), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6PackedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Default.ba085c8bae0456bc.Core:Packed.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6PackedC1Ev.carbon_thunk.(ptr %return), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4TakeI6PackedEvRT_(ptr noundef nonnull align 1 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.53704887462f6065"(ptr sret([4 x i8]) %return) #0 !dbg !51 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.381601f5a2bd2e81.Core:NaturallyAligned.Cpp"(ptr %return), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.f529dae788690cbe"(ptr sret([16 x i8]) %return) #0 !dbg !55 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.6ca50d719a15fa24.Core:ExtraAlignment.Cpp"(ptr %return), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.ee398c9a806dbb35"(ptr sret([4 x i8]) %return) #0 !dbg !58 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.ba085c8bae0456bc.Core:Packed.Cpp"(ptr %return), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
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
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "alignment.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "NaturallyAligned", linkageName: "_CNaturallyAligned.Main", scope: null, file: !6, line: 23, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 24, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 25, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 26, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 1, scope: !11)
// CHECK:STDOUT: !21 = !{!22, !22, i64 0}
// CHECK:STDOUT: !22 = !{!"p1 _ZTS16NaturallyAligned", !23, i64 0}
// CHECK:STDOUT: !23 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.381601f5a2bd2e81.Core:NaturallyAligned.Cpp", scope: null, file: !6, line: 7, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !25 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !28 = !DILocation(line: 7, column: 8, scope: !24)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "ExtraAlignment", linkageName: "_CExtraAlignment.Main", scope: null, file: !6, line: 29, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !14)
// CHECK:STDOUT: !32 = !DILocation(line: 30, column: 3, scope: !29)
// CHECK:STDOUT: !33 = !DILocation(line: 33, column: 3, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 34, column: 3, scope: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 29, column: 1, scope: !29)
// CHECK:STDOUT: !36 = !{!37, !37, i64 0}
// CHECK:STDOUT: !37 = !{!"p1 _ZTS14ExtraAlignment", !23, i64 0}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.6ca50d719a15fa24.Core:ExtraAlignment.Cpp", scope: null, file: !6, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !39 = !DILocation(line: 11, column: 20, scope: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Packed", linkageName: "_CPacked.Main", scope: null, file: !6, line: 37, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !14)
// CHECK:STDOUT: !43 = !DILocation(line: 38, column: 3, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 42, column: 3, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 43, column: 3, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 37, column: 1, scope: !40)
// CHECK:STDOUT: !47 = !{!48, !48, i64 0}
// CHECK:STDOUT: !48 = !{!"p1 _ZTS6Packed", !23, i64 0}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.ba085c8bae0456bc.Core:Packed.Cpp", scope: null, file: !6, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !50 = !DILocation(line: 15, column: 32, scope: !49)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.53704887462f6065", scope: null, file: !52, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !52 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !53 = !DILocation(line: 9, column: 28, scope: !51)
// CHECK:STDOUT: !54 = !DILocation(line: 9, column: 21, scope: !51)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.f529dae788690cbe", scope: null, file: !52, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !56 = !DILocation(line: 9, column: 28, scope: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 9, column: 21, scope: !55)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbe5dd53b28eb3d1:DefaultOrUnformed.Core.ee398c9a806dbb35", scope: null, file: !52, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !59 = !DILocation(line: 9, column: 28, scope: !58)
// CHECK:STDOUT: !60 = !DILocation(line: 9, column: 21, scope: !58)
