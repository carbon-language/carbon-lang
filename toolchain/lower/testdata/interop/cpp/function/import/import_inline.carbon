// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/import_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/import_inline.carbon

import Cpp inline '''

void F(int *_Nonnull p) {
  *p = 1;
}

''';

fn Call() -> i32 {
  var n: i32 = 0;
  Cpp.F(&n);
  return n;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_inline.carbon'
// CHECK:STDOUT: source_filename = "import_inline.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1FPi(ptr noundef %p) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %p, ptr %p.addr, align 8, !tbaa !5
// CHECK:STDOUT:   %0 = load ptr, ptr %p.addr, align 8, !tbaa !5
// CHECK:STDOUT:   store i32 1, ptr %0, align 4, !tbaa !6
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main() #1 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !38
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !38
// CHECK:STDOUT:   call void @_Z1FPi(ptr %n.var), !dbg !42
// CHECK:STDOUT:   %.loc24 = load i32, ptr %n.var, align 4, !dbg !43
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %n.var), !dbg !38
// CHECK:STDOUT:   ret i32 %.loc24, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #1 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #1 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #1 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!26}
// CHECK:STDOUT: !llvm.module.flags = !{!57, !58, !59, !9, !10}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!61}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !3 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !4 = !{!"p1 int", !3, i64 0}
// CHECK:STDOUT: !5 = !{!4, !4, i64 0}
// CHECK:STDOUT: !6 = !{!2, !2, i64 0}
// CHECK:STDOUT: !9 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !10 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !DIFile(filename: "import_inline.carbon", directory: "")
// CHECK:STDOUT: !26 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !25, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !28 = !{!27}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !33 = !{null, !27}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !25, line: 21, type: !29, spFlags: DISPFlagDefinition, unit: !26)
// CHECK:STDOUT: !38 = !DILocation(line: 22, column: 3, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 23, column: 3, scope: !35)
// CHECK:STDOUT: !43 = !DILocation(line: 24, column: 10, scope: !35)
// CHECK:STDOUT: !44 = !DILocation(line: 24, column: 3, scope: !35)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !25, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !26, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !27)
// CHECK:STDOUT: !47 = !DILocation(line: 22, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !25, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !26, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !27)
// CHECK:STDOUT: !51 = !DILocation(line: 22, column: 3, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !25, line: 22, type: !34, spFlags: DISPFlagDefinition, unit: !26, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !27)
// CHECK:STDOUT: !55 = !DILocation(line: 22, column: 3, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !59 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !60 = !{!"__libc_errno", !2, i64 0}
// CHECK:STDOUT: !61 = !{!60, !2, i64 0}
// CHECK:STDOUT:
