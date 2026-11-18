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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !50
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.A.Main(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main1AD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !50
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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6b8b385dcf82cc70(ptr %self), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF__carbon_thunk.A.Main(ptr noundef nonnull align 1 %self) #6 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.A.Main(ptr %self), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6b8b385dcf82cc70(ptr %t) #4 !dbg !39 {
// CHECK:STDOUT:   ret void, !dbg !44
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!51, !52, !53, !54, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !12 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.A.Main", scope: null, file: !20, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !22)
// CHECK:STDOUT: !37 = !DILocation(line: 6, column: 3, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6b8b385dcf82cc70", scope: null, file: !12, line: 32, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !45)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !22)
// CHECK:STDOUT: !44 = !DILocation(line: 32, column: 1, scope: !39)
// CHECK:STDOUT: !45 = !{!40}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"any pointer", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"p1 _ZTSN6Carbon4Main1AE", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !49, i64 0}
// CHECK:STDOUT: !51 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !52 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !53 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !55 = !{!"int", !47, i64 0}
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !101
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !102
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !101
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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.cf5113b5b8a292cd(ptr %self), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF__carbon_thunk.ArgsA.Main(ptr noundef nonnull align 1 %self, ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #6 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_29.2 = load i32, ptr %_, align 4, !dbg !86
// CHECK:STDOUT:   %ArgsA.F.call = call i32 @_CF.ArgsA.Main(ptr %self, i32 %.loc6_29.2), !dbg !86
// CHECK:STDOUT:   store i32 %ArgsA.F.call, ptr %_1, align 4, !dbg !86
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallCallF.Main() #4 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CallArgsF.call = call i32 @_Z9CallArgsFv(), !dbg !91
// CHECK:STDOUT:   ret i32 %CallArgsF.call, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.cf5113b5b8a292cd(ptr %t) #4 !dbg !93 {
// CHECK:STDOUT:   ret void, !dbg !98
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!59}
// CHECK:STDOUT: !llvm.module.flags = !{!51, !52, !53, !54, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !12 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"any pointer", !47, i64 0}
// CHECK:STDOUT: !51 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !52 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !53 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !55 = !{!"int", !47, i64 0}
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
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.ArgsA.Main", scope: null, file: !58, line: 6, type: !64, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !87)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !22)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !82, type: !60)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 3, scope: !82, type: !60)
// CHECK:STDOUT: !86 = !DILocation(line: 6, column: 3, scope: !82)
// CHECK:STDOUT: !87 = !{!83, !84, !85}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !58, line: 18, type: !66, spFlags: DISPFlagDefinition, unit: !59)
// CHECK:STDOUT: !91 = !DILocation(line: 18, column: 32, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 18, column: 25, scope: !88)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.cf5113b5b8a292cd", scope: null, file: !12, line: 32, type: !24, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !99)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !22)
// CHECK:STDOUT: !98 = !DILocation(line: 32, column: 1, scope: !93)
// CHECK:STDOUT: !99 = !{!94}
// CHECK:STDOUT: !100 = !{!"p1 _ZTSN6Carbon4Main5ArgsAE", !48, i64 0}
// CHECK:STDOUT: !101 = !{!100, !100, i64 0}
// CHECK:STDOUT: !102 = !{!55, !55, i64 0}
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
// CHECK:STDOUT: define void @_CF.StaticA.Main() #2 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.StaticA.Main"(ptr %self) #3 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6cfb2148c863a5b2(ptr %self), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF__carbon_thunk.StaticA.Main() #4 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.StaticA.Main(), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCallF.Main() #2 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z11CallStaticFv(), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6cfb2148c863a5b2(ptr %t) #2 !dbg !118 {
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #4 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!104}
// CHECK:STDOUT: !llvm.module.flags = !{!51, !52, !53, !54, !4, !5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !12 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !51 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !52 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !53 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !55 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !57 = !{!56, !55, i64 0}
// CHECK:STDOUT: !103 = !DIFile(filename: "static.carbon", directory: "")
// CHECK:STDOUT: !104 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !103, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !105 = !{null}
// CHECK:STDOUT: !106 = !DISubroutineType(types: !105)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "F", linkageName: "_CF.StaticA.Main", scope: null, file: !103, line: 6, type: !106, spFlags: DISPFlagDefinition, unit: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 6, column: 3, scope: !107)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.StaticA.Main", scope: null, file: !103, line: 5, type: !24, spFlags: DISPFlagDefinition, unit: !104, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !22)
// CHECK:STDOUT: !111 = !DILocation(line: 5, column: 1, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.StaticA.Main", scope: null, file: !103, line: 6, type: !106, spFlags: DISPFlagDefinition, unit: !104)
// CHECK:STDOUT: !114 = !DILocation(line: 6, column: 3, scope: !113)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "CallCallF", linkageName: "_CCallCallF.Main", scope: null, file: !103, line: 15, type: !106, spFlags: DISPFlagDefinition, unit: !104)
// CHECK:STDOUT: !116 = !DILocation(line: 15, column: 18, scope: !115)
// CHECK:STDOUT: !117 = !DILocation(line: 15, column: 1, scope: !115)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6cfb2148c863a5b2", scope: null, file: !12, line: 32, type: !24, spFlags: DISPFlagDefinition, unit: !104, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !22)
// CHECK:STDOUT: !123 = !DILocation(line: 32, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT:
