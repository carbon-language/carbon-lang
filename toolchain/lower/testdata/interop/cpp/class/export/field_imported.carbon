// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/export/field_imported.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/export/field_imported.carbon

// Fields of a class imported from another Carbon library can be accessed from
// C++, at the correct offsets.

// --- other.carbon

package Other;

class Point {
  var x: i32;
  var y: i32;
}

// --- field_imported.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
int GetY(Carbon::Other::Point* p) { return p->y; }
void SetX(Carbon::Other::Point* p, int x) { p->x = x; }
''';

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'other.carbon'
// CHECK:STDOUT: source_filename = "other.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!29}
// CHECK:STDOUT: !llvm.module.flags = !{!12, !13}
// CHECK:STDOUT:
// CHECK:STDOUT: !12 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !13 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !29 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !28, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'field_imported.carbon'
// CHECK:STDOUT: source_filename = "field_imported.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::Other::Point" = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z4GetYPN6Carbon5Other5PointE(ptr noundef %p) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %p, ptr %p.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %0 = load ptr, ptr %p.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %"class.Carbon::Other::Point", ptr %0, i32 0, i32 1
// CHECK:STDOUT:   %1 = load i32, ptr %y, align 4, !tbaa !7
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z4SetXPN6Carbon5Other5PointEi(ptr noundef %p, i32 noundef %x) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p.addr = alloca ptr, align 8
// CHECK:STDOUT:   %x.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %p, ptr %p.addr, align 8, !tbaa !4
// CHECK:STDOUT:   store i32 %x, ptr %x.addr, align 4, !tbaa !8
// CHECK:STDOUT:   %0 = load i32, ptr %x.addr, align 4, !tbaa !8
// CHECK:STDOUT:   %1 = load ptr, ptr %p.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %x1 = getelementptr inbounds nuw %"class.Carbon::Other::Point", ptr %1, i32 0, i32 0
// CHECK:STDOUT:   store i32 %0, ptr %x1, align 4, !tbaa !9
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.Point.Other"(ptr %self) #1 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.3b703080df8b538e:core.Destroy.Core"(ptr %self), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %self) #2 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d8c271b64a9e6f5b:core.Destroy.Core"(ptr %self) #2 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1.y = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %.1.y), !dbg !48
// CHECK:STDOUT:   %.2.x = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %.2.x), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d8c271b64a9e6f5b:core.Destroy.Core"(ptr %self) #2 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d8c271b64a9e6f5b:core.Destroy.Core"(ptr %self), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3b703080df8b538e:core.Destroy.Core"(ptr %self) #2 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1.y = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %.1.y), !dbg !56
// CHECK:STDOUT:   %.2.x = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !56
// CHECK:STDOUT:   call void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %.2.x), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3b703080df8b538e:core.Destroy.Core"(ptr %self) #2 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3b703080df8b538e:core.Destroy.Core"(ptr %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!32}
// CHECK:STDOUT: !llvm.module.flags = !{!62, !63, !64, !12, !13}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!66}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !3 = !{!"p1 _ZTSN6Carbon5Other5PointE", !2, i64 0}
// CHECK:STDOUT: !4 = !{!3, !3, i64 0}
// CHECK:STDOUT: !5 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !6 = !{!"_ZTSN6Carbon5Other5PointE", !5, i64 0, !5, i64 4}
// CHECK:STDOUT: !7 = !{!6, !5, i64 4}
// CHECK:STDOUT: !8 = !{!5, !5, i64 0}
// CHECK:STDOUT: !9 = !{!6, !5, i64 0}
// CHECK:STDOUT: !12 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !13 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !DIFile(filename: "field_imported.carbon", directory: "")
// CHECK:STDOUT: !32 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !31, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !34 = !{null, !33}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !{null, !30}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Point.Other", scope: null, file: !28, line: 4, type: !35, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 4, column: 1, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core", scope: null, file: !28, line: 4, type: !37, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !30)
// CHECK:STDOUT: !44 = !DILocation(line: 4, column: 1, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d8c271b64a9e6f5b:core.Destroy.Core", scope: null, file: !28, line: 4, type: !35, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !33)
// CHECK:STDOUT: !48 = !DILocation(line: 4, column: 1, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d8c271b64a9e6f5b:core.Destroy.Core", scope: null, file: !28, line: 4, type: !35, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !33)
// CHECK:STDOUT: !52 = !DILocation(line: 4, column: 1, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3b703080df8b538e:core.Destroy.Core", scope: null, file: !28, line: 4, type: !35, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !33)
// CHECK:STDOUT: !56 = !DILocation(line: 4, column: 1, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3b703080df8b538e:core.Destroy.Core", scope: null, file: !28, line: 4, type: !35, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !33)
// CHECK:STDOUT: !60 = !DILocation(line: 4, column: 1, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT: !62 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !63 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !64 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !65 = !{!"__libc_errno", !5, i64 0}
// CHECK:STDOUT: !66 = !{!65, !5, i64 0}
// CHECK:STDOUT:
