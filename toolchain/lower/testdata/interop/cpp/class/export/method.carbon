// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-fno-exceptions
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/export/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/export/method.carbon

// --- method.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {
  fn F(self) { self; }
}

inline Cpp '''
void CallF() {
  Carbon::A().F();
}
''';

// --- method_with_args_and_return.carbon
library "[[@TEST_NAME]]";

import Cpp;

class ArgsA {
  fn F(self, a: i32) -> i32 {
    self;
    return a;
  }
}

inline Cpp '''
int CallArgsF() {
  return Carbon::ArgsA().F(123);
}
''';

fn CallCallF() -> i32 { return Cpp.CallArgsF(); }

// --- static.carbon
library "[[@TEST_NAME]]";

import Cpp;

class StaticA {
  fn F() {}
}

inline Cpp '''
void CallStaticF() {
  Carbon::StaticA::F();
}
''';

fn CallCallF() { Cpp.CallStaticF(); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'method.carbon'
// CHECK:STDOUT: source_filename = "method.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::A" = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon1AD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z5CallFv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ref.tmp = alloca %"class.Carbon::A", align 1
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ref.tmp) #4
// CHECK:STDOUT:   call void @_ZNK6Carbon1A1FEv(ptr noundef nonnull align 1 %ref.tmp)
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %ref.tmp) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %ref.tmp) #4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNK6Carbon1A1FEv(ptr noundef nonnull align 1 %this) #2 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.A.Main(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.A.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.A.Main(ptr %self) #4 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.A.Main"(ptr %self) #5 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.1cabb3cc2a1d5b15(ptr %self), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5a23cc09585c55d2:core.Destroy.Core"(ptr %self) #4 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.A.Main(ptr %self) #4 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(ptr %self), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.1cabb3cc2a1d5b15(ptr %self) #4 !dbg !43 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5a23cc09585c55d2:core.Destroy.Core"(ptr %self), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nounwind }
// CHECK:STDOUT: attributes #5 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!55, !56, !57, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIFile(filename: "method.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main", scope: null, file: !20, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !30)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !22)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !{!26}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.A.Main", scope: null, file: !20, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !22)
// CHECK:STDOUT: !33 = !DILocation(line: 5, column: 1, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5a23cc09585c55d2:core.Destroy.Core", scope: null, file: !20, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !22)
// CHECK:STDOUT: !37 = !DILocation(line: 5, column: 1, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.A.Main", scope: null, file: !20, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !22)
// CHECK:STDOUT: !41 = !DILocation(line: 6, column: 3, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.1cabb3cc2a1d5b15", scope: null, file: !16, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !49)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !22)
// CHECK:STDOUT: !47 = !DILocation(line: 26, column: 5, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 24, column: 3, scope: !43)
// CHECK:STDOUT: !49 = !{!44}
// CHECK:STDOUT: !50 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !51 = !{!"omnipotent char", !50, i64 0}
// CHECK:STDOUT: !52 = !{!"any pointer", !51, i64 0}
// CHECK:STDOUT: !53 = !{!"p1 _ZTSN6Carbon1AE", !52, i64 0}
// CHECK:STDOUT: !54 = !{!53, !53, i64 0}
// CHECK:STDOUT: !55 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !56 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !58 = !{!"int", !51, i64 0}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !60 = !{!59, !58, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'method_with_args_and_return.carbon'
// CHECK:STDOUT: source_filename = "method_with_args_and_return.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::ArgsA" = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon5ArgsAD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z9CallArgsFv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ref.tmp = alloca %"class.Carbon::ArgsA", align 1
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ref.tmp) #4
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK6Carbon5ArgsA1FEi(ptr noundef nonnull align 1 %ref.tmp, i32 noundef 123)
// CHECK:STDOUT:   call void @_ZN6Carbon5ArgsAD2Ev(ptr noundef nonnull align 1 %ref.tmp) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %ref.tmp) #4
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK6Carbon5ArgsA1FEi(ptr noundef nonnull align 1 %this, i32 noundef %0) #2 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca i32, align 4
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !108
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !109
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.ArgsA.Main(ptr noundef nonnull align 1 %this1, ptr noundef nonnull align 4 dereferenceable(4) %.addr, ptr noundef nonnull align 4 dereferenceable(4) %retval)
// CHECK:STDOUT:   %1 = load i32, ptr %retval, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon5ArgsAD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !108
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.ArgsA.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.ArgsA.Main(ptr %self, i32 %a) #4 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.ArgsA.Main"(ptr %self) #5 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.93ac61ebaf93c951(ptr %self), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.eefdfd667b52bf5b:core.Destroy.Core"(ptr %self) #4 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.ArgsA.Main(ptr %self, ptr %_, ptr %_1) #4 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_29.2 = load i32, ptr %_, align 4, !dbg !93
// CHECK:STDOUT:   %ArgsA.F.call = call i32 @_CF.ArgsA.Main(ptr %self, i32 %.loc6_29.2), !dbg !93
// CHECK:STDOUT:   store i32 %ArgsA.F.call, ptr %_1, align 4, !dbg !93
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallCallF.Main() #4 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CallArgsF.call = call i32 @_Z9CallArgsFv(), !dbg !98
// CHECK:STDOUT:   ret i32 %CallArgsF.call, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.93ac61ebaf93c951(ptr %self) #4 !dbg !100 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.eefdfd667b52bf5b:core.Destroy.Core"(ptr %self), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nounwind }
// CHECK:STDOUT: attributes #5 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!62}
// CHECK:STDOUT: !llvm.module.flags = !{!55, !56, !57, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !50 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !51 = !{!"omnipotent char", !50, i64 0}
// CHECK:STDOUT: !52 = !{!"any pointer", !51, i64 0}
// CHECK:STDOUT: !55 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !56 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !58 = !{!"int", !51, i64 0}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !60 = !{!59, !58, i64 0}
// CHECK:STDOUT: !61 = !DIFile(filename: "method_with_args_and_return.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !61, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !63 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !64 = !{!63, !22, !63}
// CHECK:STDOUT: !65 = !DISubroutineType(types: !64)
// CHECK:STDOUT: !66 = !{null, !22, !63, !63}
// CHECK:STDOUT: !67 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = !DISubroutineType(types: !68)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "F", linkageName: "_CF.ArgsA.Main", scope: null, file: !61, line: 6, type: !65, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !80)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !22)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 2, scope: !70, type: !63)
// CHECK:STDOUT: !79 = !DILocation(line: 8, column: 5, scope: !70)
// CHECK:STDOUT: !80 = !{!71, !72}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.ArgsA.Main", scope: null, file: !61, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !84)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !22)
// CHECK:STDOUT: !83 = !DILocation(line: 5, column: 1, scope: !81)
// CHECK:STDOUT: !84 = !{!82}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.eefdfd667b52bf5b:core.Destroy.Core", scope: null, file: !61, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !22)
// CHECK:STDOUT: !87 = !DILocation(line: 5, column: 1, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.ArgsA.Main", scope: null, file: !61, line: 6, type: !67, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !94)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !22)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !89, type: !63)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 3, scope: !89, type: !63)
// CHECK:STDOUT: !93 = !DILocation(line: 6, column: 3, scope: !89)
// CHECK:STDOUT: !94 = !{!90, !91, !92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !61, line: 18, type: !69, spFlags: DISPFlagDefinition, unit: !62)
// CHECK:STDOUT: !98 = !DILocation(line: 18, column: 32, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 18, column: 25, scope: !95)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.93ac61ebaf93c951", scope: null, file: !16, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !106)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !22)
// CHECK:STDOUT: !104 = !DILocation(line: 26, column: 5, scope: !100)
// CHECK:STDOUT: !105 = !DILocation(line: 24, column: 3, scope: !100)
// CHECK:STDOUT: !106 = !{!101}
// CHECK:STDOUT: !107 = !{!"p1 _ZTSN6Carbon5ArgsAE", !52, i64 0}
// CHECK:STDOUT: !108 = !{!107, !107, i64 0}
// CHECK:STDOUT: !109 = !{!58, !58, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'static.carbon'
// CHECK:STDOUT: source_filename = "static.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z11CallStaticFv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6Carbon7StaticA1FEv()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon7StaticA1FEv() #1 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF__carbon_thunk.StaticA.Main()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.StaticA.Main() #2 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.StaticA.Main"(ptr %self) #3 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.57ed446d50680f19(ptr %self), !dbg !118
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.1be3804f96ff0532:core.Destroy.Core"(ptr %self) #2 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.StaticA.Main() #2 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.StaticA.Main(), !dbg !125
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main() #2 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z11CallStaticFv(), !dbg !127
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.57ed446d50680f19(ptr %self) #2 !dbg !129 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.1be3804f96ff0532:core.Destroy.Core"(ptr %self), !dbg !133
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!111}
// CHECK:STDOUT: !llvm.module.flags = !{!55, !56, !57, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !50 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !51 = !{!"omnipotent char", !50, i64 0}
// CHECK:STDOUT: !55 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !56 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !58 = !{!"int", !51, i64 0}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !60 = !{!59, !58, i64 0}
// CHECK:STDOUT: !110 = !DIFile(filename: "static.carbon", directory: "")
// CHECK:STDOUT: !111 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !110, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !112 = !{null}
// CHECK:STDOUT: !113 = !DISubroutineType(types: !112)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "F", linkageName: "_CF.StaticA.Main", scope: null, file: !110, line: 6, type: !113, spFlags: DISPFlagDefinition, unit: !111)
// CHECK:STDOUT: !115 = !DILocation(line: 6, column: 3, scope: !114)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.StaticA.Main", scope: null, file: !110, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !119)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !22)
// CHECK:STDOUT: !118 = !DILocation(line: 5, column: 1, scope: !116)
// CHECK:STDOUT: !119 = !{!117}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.1be3804f96ff0532:core.Destroy.Core", scope: null, file: !110, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !123)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !22)
// CHECK:STDOUT: !122 = !DILocation(line: 5, column: 1, scope: !120)
// CHECK:STDOUT: !123 = !{!121}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.StaticA.Main", scope: null, file: !110, line: 6, type: !113, spFlags: DISPFlagDefinition, unit: !111)
// CHECK:STDOUT: !125 = !DILocation(line: 6, column: 3, scope: !124)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !110, line: 15, type: !113, spFlags: DISPFlagDefinition, unit: !111)
// CHECK:STDOUT: !127 = !DILocation(line: 15, column: 18, scope: !126)
// CHECK:STDOUT: !128 = !DILocation(line: 15, column: 1, scope: !126)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.57ed446d50680f19", scope: null, file: !16, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !111, retainedNodes: !135)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !22)
// CHECK:STDOUT: !133 = !DILocation(line: 26, column: 5, scope: !129)
// CHECK:STDOUT: !134 = !DILocation(line: 24, column: 3, scope: !129)
// CHECK:STDOUT: !135 = !{!130}
// CHECK:STDOUT:
