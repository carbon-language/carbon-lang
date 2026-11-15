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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.A.Main(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main1AD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.A.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.A.Main(ptr %self) #4 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.A.Main"(ptr %self) #5 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF__carbon_thunk.A.Main(ptr noundef nonnull align 1 %self) #6 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(ptr %self), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nounwind }
// CHECK:STDOUT: attributes #5 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #6 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!46, !47, !48, !49, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "method.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.A.Main", scope: null, file: !22, line: 6, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !32)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 6, column: 3, scope: !27)
// CHECK:STDOUT: !32 = !{!28}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.A.Main", scope: null, file: !22, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !36)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !24)
// CHECK:STDOUT: !35 = !DILocation(line: 5, column: 1, scope: !33)
// CHECK:STDOUT: !36 = !{!34}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.A.Main", scope: null, file: !22, line: 6, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !24)
// CHECK:STDOUT: !39 = !DILocation(line: 6, column: 3, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !42 = !{!"omnipotent char", !41, i64 0}
// CHECK:STDOUT: !43 = !{!"any pointer", !42, i64 0}
// CHECK:STDOUT: !44 = !{!"p1 _ZTSN6Carbon4Main1AE", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !44, i64 0}
// CHECK:STDOUT: !46 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !47 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !50 = !{!"int", !42, i64 0}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !50, i64 0}
// CHECK:STDOUT: !52 = !{!51, !50, i64 0}
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !89
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !90
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.ArgsA.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.ArgsA.Main(ptr %self, i32 %a) #4 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.ArgsA.Main"(ptr %self) #5 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF__carbon_thunk.ArgsA.Main(ptr noundef nonnull align 1 %self, ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #6 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_29.2 = load i32, ptr %_, align 4, !dbg !81
// CHECK:STDOUT:   %ArgsA.F.call = call i32 @_CF.ArgsA.Main(ptr %self, i32 %.loc6_29.2), !dbg !81
// CHECK:STDOUT:   store i32 %ArgsA.F.call, ptr %_1, align 4, !dbg !81
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallCallF.Main() #4 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CallArgsF.call = call i32 @_Z9CallArgsFv(), !dbg !86
// CHECK:STDOUT:   ret i32 %CallArgsF.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nounwind }
// CHECK:STDOUT: attributes #5 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #6 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!54}
// CHECK:STDOUT: !llvm.module.flags = !{!46, !47, !48, !49, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !41 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !42 = !{!"omnipotent char", !41, i64 0}
// CHECK:STDOUT: !43 = !{!"any pointer", !42, i64 0}
// CHECK:STDOUT: !46 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !47 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !50 = !{!"int", !42, i64 0}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !50, i64 0}
// CHECK:STDOUT: !52 = !{!51, !50, i64 0}
// CHECK:STDOUT: !53 = !DIFile(filename: "method_with_args_and_return.carbon", directory: "")
// CHECK:STDOUT: !54 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !53, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !55 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !56 = !{!55, !24, !55}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !58 = !{null, !24, !55, !55}
// CHECK:STDOUT: !59 = !DISubroutineType(types: !58)
// CHECK:STDOUT: !60 = !{!55}
// CHECK:STDOUT: !61 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "F", linkageName: "_CF.ArgsA.Main", scope: null, file: !53, line: 6, type: !57, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !72)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !24)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !62, type: !55)
// CHECK:STDOUT: !71 = !DILocation(line: 8, column: 5, scope: !62)
// CHECK:STDOUT: !72 = !{!63, !64}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.ArgsA.Main", scope: null, file: !53, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !24)
// CHECK:STDOUT: !75 = !DILocation(line: 5, column: 1, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.ArgsA.Main", scope: null, file: !53, line: 6, type: !59, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !82)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !24)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 2, scope: !77, type: !55)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 3, scope: !77, type: !55)
// CHECK:STDOUT: !81 = !DILocation(line: 6, column: 3, scope: !77)
// CHECK:STDOUT: !82 = !{!78, !79, !80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !53, line: 18, type: !61, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !86 = !DILocation(line: 18, column: 32, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 18, column: 25, scope: !83)
// CHECK:STDOUT: !88 = !{!"p1 _ZTSN6Carbon4Main5ArgsAE", !43, i64 0}
// CHECK:STDOUT: !89 = !{!88, !88, i64 0}
// CHECK:STDOUT: !90 = !{!50, !50, i64 0}
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
// CHECK:STDOUT: define void @_CF.StaticA.Main() #2 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.StaticA.Main"(ptr %self) #3 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF__carbon_thunk.StaticA.Main() #4 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.StaticA.Main(), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main() #2 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z11CallStaticFv(), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #4 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!92}
// CHECK:STDOUT: !llvm.module.flags = !{!46, !47, !48, !49, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !41 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !42 = !{!"omnipotent char", !41, i64 0}
// CHECK:STDOUT: !46 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !47 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !50 = !{!"int", !42, i64 0}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !50, i64 0}
// CHECK:STDOUT: !52 = !{!51, !50, i64 0}
// CHECK:STDOUT: !91 = !DIFile(filename: "static.carbon", directory: "")
// CHECK:STDOUT: !92 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !91, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !93 = !{null}
// CHECK:STDOUT: !94 = !DISubroutineType(types: !93)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "F", linkageName: "_CF.StaticA.Main", scope: null, file: !91, line: 6, type: !94, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 6, column: 3, scope: !95)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.StaticA.Main", scope: null, file: !91, line: 5, type: !26, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !24)
// CHECK:STDOUT: !99 = !DILocation(line: 5, column: 1, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.StaticA.Main", scope: null, file: !91, line: 6, type: !94, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !102 = !DILocation(line: 6, column: 3, scope: !101)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !91, line: 15, type: !94, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !104 = !DILocation(line: 15, column: 18, scope: !103)
// CHECK:STDOUT: !105 = !DILocation(line: 15, column: 1, scope: !103)
// CHECK:STDOUT:
