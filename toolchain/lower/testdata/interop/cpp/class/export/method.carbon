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
// CHECK:STDOUT: %"class.Carbon::Main::A" = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon4Main1AD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z5CallFv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ref.tmp = alloca %"class.Carbon::Main::A", align 1
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ref.tmp) #4
// CHECK:STDOUT:   call void @_ZNK6Carbon4Main1A1FEv(ptr noundef nonnull align 1 %ref.tmp)
// CHECK:STDOUT:   call void @_ZN6Carbon4Main1AD2Ev(ptr noundef nonnull align 1 %ref.tmp) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %ref.tmp) #4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNK6Carbon4Main1A1FEv(ptr noundef nonnull align 1 %this) #2 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !51
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.A.Main(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main1AD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !51
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
// CHECK:STDOUT:   call void @"_CSelfDestruct.5a23cc09585c55d2:core.Destroy.Core"(ptr %self), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #4 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5a23cc09585c55d2:core.Destroy.Core"(ptr %self) #4 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.A.Main(ptr %self) #4 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(ptr %self), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
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
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
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
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !20, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !22)
// CHECK:STDOUT: !37 = !DILocation(line: 5, column: 1, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5a23cc09585c55d2:core.Destroy.Core", scope: null, file: !20, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !22)
// CHECK:STDOUT: !41 = !DILocation(line: 5, column: 1, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.A.Main", scope: null, file: !20, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !22)
// CHECK:STDOUT: !45 = !DILocation(line: 6, column: 3, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !50 = !{!"p1 _ZTSN6Carbon4Main1AE", !49, i64 0}
// CHECK:STDOUT: !51 = !{!50, !50, i64 0}
// CHECK:STDOUT: !52 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !53 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !55 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !57 = !{!56, !55, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'method_with_args_and_return.carbon'
// CHECK:STDOUT: source_filename = "method_with_args_and_return.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::Main::ArgsA" = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon4Main5ArgsAD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z9CallArgsFv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ref.tmp = alloca %"class.Carbon::Main::ArgsA", align 1
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ref.tmp) #4
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK6Carbon4Main5ArgsA1FEi(ptr noundef nonnull align 1 %ref.tmp, i32 noundef 123)
// CHECK:STDOUT:   call void @_ZN6Carbon4Main5ArgsAD2Ev(ptr noundef nonnull align 1 %ref.tmp) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %ref.tmp) #4
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK6Carbon4Main5ArgsA1FEi(ptr noundef nonnull align 1 %this, i32 noundef %0) #2 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca i32, align 4
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !102
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !103
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.ArgsA.Main(ptr noundef nonnull align 1 %this1, ptr noundef nonnull align 4 dereferenceable(4) %.addr, ptr noundef nonnull align 4 dereferenceable(4) %retval)
// CHECK:STDOUT:   %1 = load i32, ptr %retval, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main5ArgsAD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !102
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.ArgsA.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.ArgsA.Main(ptr %self, i32 %a) #4 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.ArgsA.Main"(ptr %self) #5 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.eefdfd667b52bf5b:core.Destroy.Core"(ptr %self), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #4 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.eefdfd667b52bf5b:core.Destroy.Core"(ptr %self) #4 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.ArgsA.Main(ptr %self, ptr %_, ptr %_1) #4 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_29.2 = load i32, ptr %_, align 4, !dbg !94
// CHECK:STDOUT:   %ArgsA.F.call = call i32 @_CF.ArgsA.Main(ptr %self, i32 %.loc6_29.2), !dbg !94
// CHECK:STDOUT:   store i32 %ArgsA.F.call, ptr %_1, align 4, !dbg !94
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallCallF.Main() #4 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CallArgsF.call = call i32 @_Z9CallArgsFv(), !dbg !99
// CHECK:STDOUT:   ret i32 %CallArgsF.call, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nounwind }
// CHECK:STDOUT: attributes #5 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!59}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !52 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !53 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !55 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !57 = !{!56, !55, i64 0}
// CHECK:STDOUT: !58 = !DIFile(filename: "method_with_args_and_return.carbon", directory: "")
// CHECK:STDOUT: !59 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !58, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !60 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !61 = !{!60, !22, !60}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !63 = !{null, !22, !60, !60}
// CHECK:STDOUT: !64 = !DISubroutineType(types: !63)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT: !66 = !DISubroutineType(types: !65)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "F", linkageName: "_CF.ArgsA.Main", scope: null, file: !58, line: 6, type: !62, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !77)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !22)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 2, scope: !67, type: !60)
// CHECK:STDOUT: !76 = !DILocation(line: 8, column: 5, scope: !67)
// CHECK:STDOUT: !77 = !{!68, !69}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.ArgsA.Main", scope: null, file: !58, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !22)
// CHECK:STDOUT: !80 = !DILocation(line: 5, column: 1, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !58, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !22)
// CHECK:STDOUT: !84 = !DILocation(line: 5, column: 1, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.eefdfd667b52bf5b:core.Destroy.Core", scope: null, file: !58, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !89)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !22)
// CHECK:STDOUT: !88 = !DILocation(line: 5, column: 1, scope: !86)
// CHECK:STDOUT: !89 = !{!87}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.ArgsA.Main", scope: null, file: !58, line: 6, type: !64, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !95)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !22)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 2, scope: !90, type: !60)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 3, scope: !90, type: !60)
// CHECK:STDOUT: !94 = !DILocation(line: 6, column: 3, scope: !90)
// CHECK:STDOUT: !95 = !{!91, !92, !93}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !58, line: 18, type: !66, spFlags: DISPFlagDefinition, unit: !59)
// CHECK:STDOUT: !99 = !DILocation(line: 18, column: 32, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 18, column: 25, scope: !96)
// CHECK:STDOUT: !101 = !{!"p1 _ZTSN6Carbon4Main5ArgsAE", !49, i64 0}
// CHECK:STDOUT: !102 = !{!101, !101, i64 0}
// CHECK:STDOUT: !103 = !{!55, !55, i64 0}
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
// CHECK:STDOUT:   call void @_ZN6Carbon4Main7StaticA1FEv()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon4Main7StaticA1FEv() #1 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF__carbon_thunk.StaticA.Main()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.StaticA.Main() #2 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.StaticA.Main"(ptr %self) #3 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.1be3804f96ff0532:core.Destroy.Core"(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #2 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1be3804f96ff0532:core.Destroy.Core"(ptr %self) #2 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.StaticA.Main() #2 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.StaticA.Main(), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main() #2 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z11CallStaticFv(), !dbg !125
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!105}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !52 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !53 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !55 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !57 = !{!56, !55, i64 0}
// CHECK:STDOUT: !104 = !DIFile(filename: "static.carbon", directory: "")
// CHECK:STDOUT: !105 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !104, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !106 = !{null}
// CHECK:STDOUT: !107 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "F", linkageName: "_CF.StaticA.Main", scope: null, file: !104, line: 6, type: !107, spFlags: DISPFlagDefinition, unit: !105)
// CHECK:STDOUT: !109 = !DILocation(line: 6, column: 3, scope: !108)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.StaticA.Main", scope: null, file: !104, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !105, retainedNodes: !113)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !22)
// CHECK:STDOUT: !112 = !DILocation(line: 5, column: 1, scope: !110)
// CHECK:STDOUT: !113 = !{!111}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !104, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !105, retainedNodes: !117)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !22)
// CHECK:STDOUT: !116 = !DILocation(line: 5, column: 1, scope: !114)
// CHECK:STDOUT: !117 = !{!115}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1be3804f96ff0532:core.Destroy.Core", scope: null, file: !104, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !105, retainedNodes: !121)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !22)
// CHECK:STDOUT: !120 = !DILocation(line: 5, column: 1, scope: !118)
// CHECK:STDOUT: !121 = !{!119}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.StaticA.Main", scope: null, file: !104, line: 6, type: !107, spFlags: DISPFlagDefinition, unit: !105)
// CHECK:STDOUT: !123 = !DILocation(line: 6, column: 3, scope: !122)
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !104, line: 15, type: !107, spFlags: DISPFlagDefinition, unit: !105)
// CHECK:STDOUT: !125 = !DILocation(line: 15, column: 18, scope: !124)
// CHECK:STDOUT: !126 = !DILocation(line: 15, column: 1, scope: !124)
// CHECK:STDOUT:
