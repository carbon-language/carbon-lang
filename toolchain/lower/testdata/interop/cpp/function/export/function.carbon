// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/function.carbon

// --- other.carbon
package Other;
fn NoArgs() {}
fn BoolArg(a: bool) {
  a;
}
fn IntArg(a: i32) {
  a;
}
fn FloatArg(a: f32) {
  a;
}
fn IntReturn() -> i32 {
  return 123;
}
// --- function.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
int G() {
  Carbon::Other::NoArgs();
  Carbon::Other::BoolArg(true);
  Carbon::Other::IntArg(123);
  Carbon::Other::FloatArg(1.5);
  return Carbon::Other::IntReturn();
}
''';

// --- single_file.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {}

inline Cpp '''
inline void G1() {
  Carbon::F();
}

// Call an inline function indirectly to ensure declarations are getting
// properly registered with the CodeGen consumer.
inline void G2() {
  G1();
}
''';

fn H() { Cpp.G2(); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'other.carbon'
// CHECK:STDOUT: source_filename = "other.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CNoArgs.Other() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CBoolArg.Other(i1 %a) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIntArg.Other(i32 %a) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFloatArg.Other(float %a) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CIntReturn.Other() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 123, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{!6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "NoArgs", linkageName: "_CNoArgs.Other", scope: null, file: !4, line: 2, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DILocation(line: 2, column: 1, scope: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "BoolArg", linkageName: "_CBoolArg.Other", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !23)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !18, type: !9)
// CHECK:STDOUT: !22 = !DILocation(line: 3, column: 1, scope: !18)
// CHECK:STDOUT: !23 = !{!19}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "IntArg", linkageName: "_CIntArg.Other", scope: null, file: !4, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !6)
// CHECK:STDOUT: !28 = !DILocation(line: 6, column: 1, scope: !24)
// CHECK:STDOUT: !29 = !{!25}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "FloatArg", linkageName: "_CFloatArg.Other", scope: null, file: !4, line: 9, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !9)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 1, scope: !30)
// CHECK:STDOUT: !35 = !{!31}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "IntReturn", linkageName: "_CIntReturn.Other", scope: null, file: !4, line: 12, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !39 = !DILocation(line: 13, column: 3, scope: !36)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'function.carbon'
// CHECK:STDOUT: source_filename = "function.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1Gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6Carbon5OtherL6NoArgsEv()
// CHECK:STDOUT:   call void @_ZN6Carbon5OtherL7BoolArgEb(i1 noundef zeroext true)
// CHECK:STDOUT:   call void @_ZN6Carbon5OtherL6IntArgEi(i32 noundef 123)
// CHECK:STDOUT:   call void @_ZN6Carbon5OtherL8FloatArgEf(float noundef 1.500000e+00)
// CHECK:STDOUT:   %call = call noundef i32 @_ZN6Carbon5OtherL9IntReturnEv()
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5OtherL6NoArgsEv() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNoArgs__carbon_thunk.Other()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5OtherL7BoolArgEb(i1 noundef zeroext %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i8, align 1
// CHECK:STDOUT:   %storedv = zext i1 %0 to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %.addr, align 1, !tbaa !63
// CHECK:STDOUT:   call void @_CBoolArg__carbon_thunk.Other(ptr noundef nonnull align 1 dereferenceable(1) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5OtherL6IntArgEi(i32 noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !65
// CHECK:STDOUT:   call void @_CIntArg__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5OtherL8FloatArgEf(float noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca float, align 4
// CHECK:STDOUT:   store float %0, ptr %.addr, align 4, !tbaa !67
// CHECK:STDOUT:   call void @_CFloatArg__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZN6Carbon5OtherL9IntReturnEv() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca i32, align 4
// CHECK:STDOUT:   call void @_CIntReturn__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %retval)
// CHECK:STDOUT:   %0 = load i32, ptr %retval, align 4
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CNoArgs.Other()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CNoArgs__carbon_thunk.Other() #2 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNoArgs.Other(), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CBoolArg.Other(i1)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CBoolArg__carbon_thunk.Other(ptr %_) #2 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load i8, ptr %_, align 1, !dbg !46
// CHECK:STDOUT:   %.11 = trunc i8 %.1 to i1, !dbg !46
// CHECK:STDOUT:   call void @_CBoolArg.Other(i1 %.11), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CIntArg.Other(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIntArg__carbon_thunk.Other(ptr %_) #2 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load i32, ptr %_, align 4, !dbg !50
// CHECK:STDOUT:   call void @_CIntArg.Other(i32 %.1), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFloatArg.Other(float)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFloatArg__carbon_thunk.Other(ptr %_) #2 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load float, ptr %_, align 4, !dbg !54
// CHECK:STDOUT:   call void @_CFloatArg.Other(float %.1), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CIntReturn.Other()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIntReturn__carbon_thunk.Other(ptr %_) #2 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %IntReturn.call = call i32 @_CIntReturn.Other(), !dbg !58
// CHECK:STDOUT:   store i32 %IntReturn.call, ptr %_, align 4, !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
// CHECK:STDOUT: !llvm.module.flags = !{!68, !69, !70, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!72}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !40 = !DIFile(filename: "function.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "NoArgs__carbon_thunk", linkageName: "_CNoArgs__carbon_thunk.Other", scope: null, file: !4, line: 2, type: !8, spFlags: DISPFlagDefinition, unit: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 2, column: 1, scope: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "BoolArg__carbon_thunk", linkageName: "_CBoolArg__carbon_thunk.Other", scope: null, file: !4, line: 3, type: !11, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !9)
// CHECK:STDOUT: !46 = !DILocation(line: 3, column: 1, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "IntArg__carbon_thunk", linkageName: "_CIntArg__carbon_thunk.Other", scope: null, file: !4, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !6)
// CHECK:STDOUT: !50 = !DILocation(line: 6, column: 1, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "FloatArg__carbon_thunk", linkageName: "_CFloatArg__carbon_thunk.Other", scope: null, file: !4, line: 9, type: !11, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !9)
// CHECK:STDOUT: !54 = !DILocation(line: 9, column: 1, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "IntReturn__carbon_thunk", linkageName: "_CIntReturn__carbon_thunk.Other", scope: null, file: !4, line: 12, type: !13, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !6)
// CHECK:STDOUT: !58 = !DILocation(line: 12, column: 1, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !61 = !{!"omnipotent char", !60, i64 0}
// CHECK:STDOUT: !62 = !{!"bool", !61, i64 0}
// CHECK:STDOUT: !63 = !{!62, !62, i64 0}
// CHECK:STDOUT: !64 = !{!"int", !61, i64 0}
// CHECK:STDOUT: !65 = !{!64, !64, i64 0}
// CHECK:STDOUT: !66 = !{!"float", !61, i64 0}
// CHECK:STDOUT: !67 = !{!66, !66, i64 0}
// CHECK:STDOUT: !68 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !69 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !70 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !71 = !{!"__libc_errno", !64, i64 0}
// CHECK:STDOUT: !72 = !{!71, !64, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'single_file.carbon'
// CHECK:STDOUT: source_filename = "single_file.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z2G2v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z2G1v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Main() #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z2G2v(), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z2G2v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z2G1v()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z2G1v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6CarbonL1FEv()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL1FEv() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF__carbon_thunk.Main()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!74}
// CHECK:STDOUT: !llvm.module.flags = !{!68, !69, !70, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!72}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !60 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !61 = !{!"omnipotent char", !60, i64 0}
// CHECK:STDOUT: !64 = !{!"int", !61, i64 0}
// CHECK:STDOUT: !68 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !69 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !70 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !71 = !{!"__libc_errno", !64, i64 0}
// CHECK:STDOUT: !72 = !{!71, !64, i64 0}
// CHECK:STDOUT: !73 = !DIFile(filename: "single_file.carbon", directory: "")
// CHECK:STDOUT: !74 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !73, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !73, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !74)
// CHECK:STDOUT: !76 = !DILocation(line: 6, column: 1, scope: !75)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Main", scope: null, file: !73, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 6, column: 1, scope: !77)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !73, line: 20, type: !8, spFlags: DISPFlagDefinition, unit: !74)
// CHECK:STDOUT: !80 = !DILocation(line: 20, column: 10, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 20, column: 1, scope: !79)
// CHECK:STDOUT:
