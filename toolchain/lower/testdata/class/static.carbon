// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/static.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/static.carbon

// --- static.carbon
library "[[@TEST_NAME]]";

class C {
  static var static_field: i32;
  static var static_field_with_init: i32 = 123;
}

fn F() {
  C.static_field = 456;
  C.static_field_with_init = 789;
}

// --- static_export.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {
  static var x: i32 = 123;
}

inline Cpp '''
int F() {
  return Carbon::C::x;
}
''';

fn Run() -> i32 {
  return Cpp.F();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'static.carbon'
// CHECK:STDOUT: source_filename = "static.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cstatic_field.C.Main = global i32 0
// CHECK:STDOUT: @_Cstatic_field_with_init.C.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 456, ptr @_Cstatic_field.C.Main, align 4, !dbg !28
// CHECK:STDOUT:   store i32 789, ptr @_Cstatic_field_with_init.C.Main, align 4, !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 poison, ptr @_Cstatic_field.C.Main, align 4, !dbg !33
// CHECK:STDOUT:   store i32 123, ptr @_Cstatic_field_with_init.C.Main, align 4, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!6, !7}
// CHECK:STDOUT:
// CHECK:STDOUT: !6 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !7 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "static.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !22, line: 8, type: !26, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 9, column: 3, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 10, column: 3, scope: !27)
// CHECK:STDOUT: !30 = !DILocation(line: 8, column: 1, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !22, type: !26, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 4, column: 10, scope: !31)
// CHECK:STDOUT: !34 = !DILocation(line: 5, column: 10, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 0, scope: !31)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'static_export.carbon'
// CHECK:STDOUT: source_filename = "static_export.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cx.C.Main = global i32 0, align 4
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1Fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = load i32, ptr @_Cx.C.Main, align 4, !tbaa !3
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.C.Main"(ptr %self) #1 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #2 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #2 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_Z1Fv(), !dbg !54
// CHECK:STDOUT:   ret i32 %F.call, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #2 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 123, ptr @_Cx.C.Main, align 4, !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!37}
// CHECK:STDOUT: !llvm.module.flags = !{!59, !60, !61, !6, !7}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!63}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !3 = !{!2, !2, i64 0}
// CHECK:STDOUT: !6 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !7 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{null}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !36 = !DIFile(filename: "static_export.carbon", directory: "")
// CHECK:STDOUT: !37 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !36, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !39 = !{null, !38}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !{!24}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.C.Main", scope: null, file: !36, line: 4, type: !40, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 4, column: 1, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !36, line: 4, type: !40, spFlags: DISPFlagDefinition, unit: !37, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !38)
// CHECK:STDOUT: !49 = !DILocation(line: 4, column: 1, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !36, line: 14, type: !42, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !54 = !DILocation(line: 15, column: 10, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 15, column: 3, scope: !51)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !36, type: !26, spFlags: DISPFlagDefinition, unit: !37)
// CHECK:STDOUT: !57 = !DILocation(line: 5, column: 10, scope: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 0, scope: !56)
// CHECK:STDOUT: !59 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !60 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !61 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !62 = !{!"__libc_errno", !2, i64 0}
// CHECK:STDOUT: !63 = !{!62, !2, i64 0}
// CHECK:STDOUT:
