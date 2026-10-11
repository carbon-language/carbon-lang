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
// CHECK:STDOUT:   %0 = load i32, ptr %price, align 4, !tbaa !49
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CGet.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGet__carbon_thunk.Main(ptr %_) #1 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Get.call = call ptr @_CGet.Main(), !dbg !14
// CHECK:STDOUT:   store ptr %Get.call, ptr %_, align 8, !dbg !14
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.A.Main"(ptr %self) #2 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1ff1de8b251403c3(ptr %self), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a58e9b74ce410031:core.SubobjectDestroy.Core"(ptr %self) #1 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_9.1.price = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc15_9.1.price), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5a23cc09585c55d2:core.SubobjectDestroy.Core"(ptr %self) #1 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_9.1.price = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %.loc15_9.1.price), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1ff1de8b251403c3(ptr %t) #1 !dbg !31 {
// CHECK:STDOUT:   call void @"_COp.5a23cc09585c55d2:core.SubobjectDestroy.Core"(ptr %t), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #1 !dbg !38 {
// CHECK:STDOUT:   ret void, !dbg !43
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
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!56}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "issue7142.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !7}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Get__carbon_thunk", linkageName: "_CGet__carbon_thunk.Main", scope: null, file: !4, line: 16, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 16, column: 1, scope: !12)
// CHECK:STDOUT: !15 = !{!13}
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.A.Main", scope: null, file: !4, line: 15, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !19)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !16, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 1, scope: !16)
// CHECK:STDOUT: !19 = !{!17}
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a58e9b74ce410031:core.SubobjectDestroy.Core", scope: null, file: !4, line: 15, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !20, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 15, column: 1, scope: !20)
// CHECK:STDOUT: !23 = !{null, !6}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!21}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5a23cc09585c55d2:core.SubobjectDestroy.Core", scope: null, file: !4, line: 15, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !7)
// CHECK:STDOUT: !28 = !DILocation(line: 15, column: 1, scope: !26)
// CHECK:STDOUT: !29 = !{!27}
// CHECK:STDOUT: !30 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1ff1de8b251403c3", scope: null, file: !30, line: 30, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !7)
// CHECK:STDOUT: !35 = !DILocation(line: 32, column: 5, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 30, column: 1, scope: !31)
// CHECK:STDOUT: !37 = !{!32}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !30, line: 30, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !44)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !6)
// CHECK:STDOUT: !43 = !DILocation(line: 30, column: 1, scope: !38)
// CHECK:STDOUT: !44 = !{!39}
// CHECK:STDOUT: !45 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !46 = !{!"omnipotent char", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"int", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"_ZTSN6Carbon1AE", !47, i64 0}
// CHECK:STDOUT: !49 = !{!48, !47, i64 0}
// CHECK:STDOUT: !52 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !53 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !55 = !{!"__libc_errno", !47, i64 0}
// CHECK:STDOUT: !56 = !{!55, !47, i64 0}
// CHECK:STDOUT:
