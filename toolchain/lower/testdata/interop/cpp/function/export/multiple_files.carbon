// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/multiple_files.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/multiple_files.carbon

// Each file that exports a Carbon function to C++ emits its own copy of the
// Carbon thunk for that function, so the thunk must have linkonce_odr linkage
// to avoid duplicate symbols when the files are linked together.

// --- other.carbon

package Other;

fn Identity(n: i32) -> i32 {
  return n;
}

// --- a.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
int UseA() { return Carbon::Other::Identity(3); }
''';

// --- b.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
int UseB() { return Carbon::Other::Identity(4); }
''';

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'other.carbon'
// CHECK:STDOUT: source_filename = "other.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CIdentity.Other(i32 %n) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %n, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Identity", linkageName: "_CIdentity.Other", scope: null, file: !18, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !30)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 5, column: 3, scope: !23)
// CHECK:STDOUT: !30 = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z4UseAv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef i32 @_ZN6Carbon5OtherL8IdentityEi(i32 noundef 3)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZN6Carbon5OtherL8IdentityEi(i32 noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca i32, align 4
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !43
// CHECK:STDOUT:   call void @_CIdentity__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %.addr, ptr noundef nonnull align 4 dereferenceable(4) %retval)
// CHECK:STDOUT:   %1 = load i32, ptr %retval, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CIdentity.Other(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CIdentity__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #2 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load i32, ptr %_, align 4, !dbg !38
// CHECK:STDOUT:   %Identity.call = call i32 @_CIdentity.Other(i32 %.1), !dbg !38
// CHECK:STDOUT:   store i32 %Identity.call, ptr %_1, align 4, !dbg !38
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!32}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !32 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !31, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !33 = !{null, !20, !20}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Identity__carbon_thunk", linkageName: "_CIdentity__carbon_thunk.Other", scope: null, file: !18, line: 4, type: !34, spFlags: DISPFlagDefinition, unit: !32, retainedNodes: !39)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !20)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 2, scope: !35, type: !20)
// CHECK:STDOUT: !38 = !DILocation(line: 4, column: 1, scope: !35)
// CHECK:STDOUT: !39 = !{!36, !37}
// CHECK:STDOUT: !40 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !41 = !{!"omnipotent char", !40, i64 0}
// CHECK:STDOUT: !42 = !{!"int", !41, i64 0}
// CHECK:STDOUT: !43 = !{!42, !42, i64 0}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !42, i64 0}
// CHECK:STDOUT: !48 = !{!47, !42, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z4UseBv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef i32 @_ZN6Carbon5OtherL8IdentityEi(i32 noundef 4)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZN6Carbon5OtherL8IdentityEi(i32 noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca i32, align 4
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !43
// CHECK:STDOUT:   call void @_CIdentity__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %.addr, ptr noundef nonnull align 4 dereferenceable(4) %retval)
// CHECK:STDOUT:   %1 = load i32, ptr %retval, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CIdentity.Other(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CIdentity__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #2 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load i32, ptr %_, align 4, !dbg !54
// CHECK:STDOUT:   %Identity.call = call i32 @_CIdentity.Other(i32 %.1), !dbg !54
// CHECK:STDOUT:   store i32 %Identity.call, ptr %_1, align 4, !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!50}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !33 = !{null, !20, !20}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !40 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !41 = !{!"omnipotent char", !40, i64 0}
// CHECK:STDOUT: !42 = !{!"int", !41, i64 0}
// CHECK:STDOUT: !43 = !{!42, !42, i64 0}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !42, i64 0}
// CHECK:STDOUT: !48 = !{!47, !42, i64 0}
// CHECK:STDOUT: !49 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !50 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !49, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Identity__carbon_thunk", linkageName: "_CIdentity__carbon_thunk.Other", scope: null, file: !18, line: 4, type: !34, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !55)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !20)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 2, scope: !51, type: !20)
// CHECK:STDOUT: !54 = !DILocation(line: 4, column: 1, scope: !51)
// CHECK:STDOUT: !55 = !{!52, !53}
// CHECK:STDOUT:
