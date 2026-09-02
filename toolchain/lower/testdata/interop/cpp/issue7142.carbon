// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/issue7142.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/issue7142.carbon

import Cpp;

class A { var price: i32; };
fn Get() -> A*;

inline Cpp '''
struct S {
  static int Vend() {
    return Carbon::Get()->price;
  }
};

void f() {
  S::Vend();
}
''';

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'issue7142.carbon'
// CHECK:STDOUT: source_filename = "issue7142.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::A" = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1S4VendEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef i32 @_ZN1S4VendEv()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef i32 @_ZN1S4VendEv() #0 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef ptr @_ZN6CarbonL3GetEv()
// CHECK:STDOUT:   %price = getelementptr inbounds nuw %"class.Carbon::A", ptr %call, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %price, align 4, !tbaa !12
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CGet.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGet__carbon_thunk.Main(ptr %_) #1 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Get.call = call ptr @_CGet.Main(), !dbg !20
// CHECK:STDOUT:   store ptr %Get.call, ptr %_, align 8, !dbg !20
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.A.Main"(ptr %self) #2 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.e649d414b9e7644a:core.Destroy.Core"(ptr %self), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #1 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7a7bf93ecb3607ea:core.Destroy.Core"(ptr %self) #1 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e649d414b9e7644a:core.Destroy.Core"(ptr %self) #1 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal noundef ptr @_ZN6CarbonL3GetEv() #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca ptr, align 8
// CHECK:STDOUT:   call void @_CGet__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %retval)
// CHECK:STDOUT:   %0 = load ptr, ptr %retval, align 8
// CHECK:STDOUT:   ret ptr %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "issue7142.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = !{!13, !9, i64 0}
// CHECK:STDOUT: !13 = !{!"_ZTSN6Carbon1AE", !9, i64 0}
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Get__carbon_thunk", linkageName: "_CGet__carbon_thunk.Main", scope: null, file: !1, line: 16, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 16, column: 1, scope: !14)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.A.Main", scope: null, file: !1, line: 15, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !21, type: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 15, column: 1, scope: !21)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 15, type: !26, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{null, !28}
// CHECK:STDOUT: !28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !25, type: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 15, column: 1, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7a7bf93ecb3607ea:core.Destroy.Core", scope: null, file: !1, line: 15, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !17)
// CHECK:STDOUT: !35 = !DILocation(line: 15, column: 1, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e649d414b9e7644a:core.Destroy.Core", scope: null, file: !1, line: 15, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !17)
// CHECK:STDOUT: !39 = !DILocation(line: 15, column: 1, scope: !36)
// CHECK:STDOUT:
