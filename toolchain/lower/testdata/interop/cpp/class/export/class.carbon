// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-fno-exceptions
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/export/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/export/class.carbon

// --- pointer.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {
  var x: i32;
  var y: i32;
}

inline Cpp '''
Carbon::A *pass(Carbon::A *p) { return p; }
void call() { pass(nullptr); }
''';

fn DoIt() { Cpp.call(); }

// --- layout.carbon

library "[[@TEST_NAME]]";

import Cpp;

base class LayoutA {
  var x: i32*;
  var y: i32;
}

class LayoutB {
  var a: LayoutA;
  var b: i32;
}

class LayoutC {
  extend base: LayoutA;
  var c: i32;
}

fn CallLayoutA(a: LayoutA*);
fn CallLayoutB(b: LayoutB*);
fn CallLayoutC(b: LayoutC*);

inline Cpp '''
void f() {
  Carbon::LayoutA a;
  Carbon::CallLayoutA(&a);
}

void g() {
  Carbon::LayoutB b;
  Carbon::CallLayoutB(&b);
}

void h() {
  Carbon::LayoutC c;
  Carbon::CallLayoutC(&c);
  Carbon::CallLayoutA(&c);
}
''';

// --- field.carbon

library "[[@TEST_NAME]]";

import Cpp;

class FieldC {
  var a: i32;
  var b: i32;
}

inline Cpp '''
void F(Carbon::FieldC& c) {
  c.a = 12;
  c.b = 34;
}
''';

fn G() {
  var c: FieldC = {.a = 0, .b = 0};
  Cpp.F(ref c);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pointer.carbon'
// CHECK:STDOUT: source_filename = "pointer.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef ptr @_Z4passPN6Carbon4Main1AE(ptr noundef %p) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %p, ptr %p.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %0 = load ptr, ptr %p.addr, align 8, !tbaa !4
// CHECK:STDOUT:   ret ptr %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z4callv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef ptr @_Z4passPN6Carbon4Main1AE(ptr noundef null)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDoIt.Main() #1 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4callv(), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !19, !20}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !3 = !{!"p1 _ZTSN6Carbon4Main1AE", !2, i64 0}
// CHECK:STDOUT: !4 = !{!3, !3, i64 0}
// CHECK:STDOUT: !5 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !19 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !20 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "pointer.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !38 = !{null}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "DoIt", linkageName: "_CDoIt.Main", scope: null, file: !35, line: 15, type: !39, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 15, column: 13, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 15, column: 1, scope: !40)
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !5, i64 0}
// CHECK:STDOUT: !47 = !{!46, !5, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'layout.carbon'
// CHECK:STDOUT: source_filename = "layout.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::Main::LayoutA" = type <{ ptr, i32 }>
// CHECK:STDOUT: %"class.Carbon::Main::LayoutB" = type { %"class.Carbon::Main::LayoutA", i32 }
// CHECK:STDOUT: %"class.Carbon::Main::LayoutC" = type { %"class.Carbon::Main::LayoutA", i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon4Main7LayoutAD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon4Main7LayoutBD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon4Main7LayoutCD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a = alloca %"class.Carbon::Main::LayoutA", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a) #5
// CHECK:STDOUT:   call void @_ZN6Carbon4MainL11CallLayoutAEPNS0_7LayoutAE(ptr noundef %a)
// CHECK:STDOUT:   call void @_ZN6Carbon4Main7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %a) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon4MainL11CallLayoutAEPNS0_7LayoutAE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !99
// CHECK:STDOUT:   call void @_CCallLayoutA__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !99
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.LayoutA.Main"(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b = alloca %"class.Carbon::Main::LayoutB", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b) #5
// CHECK:STDOUT:   call void @_ZN6Carbon4MainL11CallLayoutBEPNS0_7LayoutBE(ptr noundef %b)
// CHECK:STDOUT:   call void @_ZN6Carbon4Main7LayoutBD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %b) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon4MainL11CallLayoutBEPNS0_7LayoutBE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !101
// CHECK:STDOUT:   call void @_CCallLayoutB__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutBD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.LayoutB.Main"(ptr noundef nonnull align 8 dereferenceable(16) %this1)
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %"class.Carbon::Main::LayoutB", ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   call void @_ZN6Carbon4Main7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %a) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1hv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c = alloca %"class.Carbon::Main::LayoutC", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c) #5
// CHECK:STDOUT:   call void @_ZN6Carbon4MainL11CallLayoutCEPNS0_7LayoutCE(ptr noundef %c)
// CHECK:STDOUT:   call void @_ZN6Carbon4MainL11CallLayoutAEPNS0_7LayoutAE(ptr noundef %c)
// CHECK:STDOUT:   call void @_ZN6Carbon4Main7LayoutCD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %c) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %c) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon4MainL11CallLayoutCEPNS0_7LayoutCE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !103
// CHECK:STDOUT:   call void @_CCallLayoutC__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutCD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !103
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.LayoutC.Main"(ptr noundef nonnull align 8 dereferenceable(16) %this1)
// CHECK:STDOUT:   call void @_ZN6Carbon4Main7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this1) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallLayoutA.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallLayoutB.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallLayoutC.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutA.Main"(ptr %self) #4 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %self), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutA__carbon_thunk.Main(ptr %_) #5 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %_, align 8, !dbg !59
// CHECK:STDOUT:   call void @_CCallLayoutA.Main(ptr %.loc21), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutB.Main"(ptr %self) #4 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd2d442e060c7dd0(ptr %self), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutB__carbon_thunk.Main(ptr %_) #5 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22 = load ptr, ptr %_, align 8, !dbg !67
// CHECK:STDOUT:   call void @_CCallLayoutB.Main(ptr %.loc22), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutC.Main"(ptr %self) #4 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d869c4fcbd4e85a2(ptr %self), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutC__carbon_thunk.Main(ptr %_) #5 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %_, align 8, !dbg !75
// CHECK:STDOUT:   call void @_CCallLayoutC.Main(ptr %.loc23), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %t) #5 !dbg !77 {
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dd2d442e060c7dd0(ptr %t) #5 !dbg !84 {
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d869c4fcbd4e85a2(ptr %t) #5 !dbg !91 {
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon4Main7LayoutAD2Ev, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #5 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!49}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !19, !20}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !5 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !19 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !20 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !27 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !5, i64 0}
// CHECK:STDOUT: !47 = !{!46, !5, i64 0}
// CHECK:STDOUT: !48 = !DIFile(filename: "layout.carbon", directory: "")
// CHECK:STDOUT: !49 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !48, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !50 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !51 = !{null, !50}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutA.Main", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !50)
// CHECK:STDOUT: !55 = !DILocation(line: 6, column: 1, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "CallLayoutA__carbon_thunk", linkageName: "_CCallLayoutA__carbon_thunk.Main", scope: null, file: !48, line: 21, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !50)
// CHECK:STDOUT: !59 = !DILocation(line: 21, column: 1, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutB.Main", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !50)
// CHECK:STDOUT: !63 = !DILocation(line: 11, column: 1, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "CallLayoutB__carbon_thunk", linkageName: "_CCallLayoutB__carbon_thunk.Main", scope: null, file: !48, line: 22, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !50)
// CHECK:STDOUT: !67 = !DILocation(line: 22, column: 1, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutC.Main", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !50)
// CHECK:STDOUT: !71 = !DILocation(line: 16, column: 1, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "CallLayoutC__carbon_thunk", linkageName: "_CCallLayoutC__carbon_thunk.Main", scope: null, file: !48, line: 23, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !50)
// CHECK:STDOUT: !75 = !DILocation(line: 23, column: 1, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ec5c5666af23bee2", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !83)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !50)
// CHECK:STDOUT: !82 = !DILocation(line: 32, column: 1, scope: !77)
// CHECK:STDOUT: !83 = !{!78}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dd2d442e060c7dd0", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !90)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !50)
// CHECK:STDOUT: !89 = !DILocation(line: 32, column: 1, scope: !84)
// CHECK:STDOUT: !90 = !{!85}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d869c4fcbd4e85a2", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !97)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !50)
// CHECK:STDOUT: !96 = !DILocation(line: 32, column: 1, scope: !91)
// CHECK:STDOUT: !97 = !{!92}
// CHECK:STDOUT: !98 = !{!"p1 _ZTSN6Carbon4Main7LayoutAE", !2, i64 0}
// CHECK:STDOUT: !99 = !{!98, !98, i64 0}
// CHECK:STDOUT: !100 = !{!"p1 _ZTSN6Carbon4Main7LayoutBE", !2, i64 0}
// CHECK:STDOUT: !101 = !{!100, !100, i64 0}
// CHECK:STDOUT: !102 = !{!"p1 _ZTSN6Carbon4Main7LayoutCE", !2, i64 0}
// CHECK:STDOUT: !103 = !{!102, !102, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::Main::FieldC" = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: @FieldC.val.loc19_3.1 = internal constant { i32, i32 } zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1FRN6Carbon4Main6FieldCE(ptr noundef nonnull align 4 dereferenceable(8) %c) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %c.addr, align 8, !tbaa !11, !nonnull !13, !align !14
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %"class.Carbon::Main::FieldC", ptr %0, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %a, align 4, !tbaa !15
// CHECK:STDOUT:   %1 = load ptr, ptr %c.addr, align 8, !tbaa !11, !nonnull !13, !align !14
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %"class.Carbon::Main::FieldC", ptr %1, i32 0, i32 1
// CHECK:STDOUT:   store i32 34, ptr %b, align 4, !tbaa !16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.FieldC.Main"(ptr %self) #1 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %self), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #2 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, i32 }, align 4, !dbg !111
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !111
// CHECK:STDOUT:   %.loc19_34.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 0, !dbg !112
// CHECK:STDOUT:   %.loc19_34.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 1, !dbg !112
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @FieldC.val.loc19_3.1, i64 8, i1 false), !dbg !111
// CHECK:STDOUT:   call void @_Z1FRN6Carbon4Main6FieldCE(ptr %c.var), !dbg !116
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %c.var), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %t) #2 !dbg !118 {
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0f9f2a833d58b7bf, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!105}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !19, !20}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !5 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !10 = !{!"p1 _ZTSN6Carbon4Main6FieldCE", !2, i64 0}
// CHECK:STDOUT: !11 = !{!10, !10, i64 0}
// CHECK:STDOUT: !12 = !{!"_ZTSN6Carbon4Main6FieldCE", !5, i64 0, !5, i64 4}
// CHECK:STDOUT: !13 = !{}
// CHECK:STDOUT: !14 = !{i64 4}
// CHECK:STDOUT: !15 = !{!12, !5, i64 0}
// CHECK:STDOUT: !16 = !{!12, !5, i64 4}
// CHECK:STDOUT: !19 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !20 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !27 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !38 = !{null}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !5, i64 0}
// CHECK:STDOUT: !47 = !{!46, !5, i64 0}
// CHECK:STDOUT: !50 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !51 = !{null, !50}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !104 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !105 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !104, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.FieldC.Main", scope: null, file: !104, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !105, retainedNodes: !109)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !50)
// CHECK:STDOUT: !108 = !DILocation(line: 6, column: 1, scope: !106)
// CHECK:STDOUT: !109 = !{!107}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !104, line: 18, type: !39, spFlags: DISPFlagDefinition, unit: !105)
// CHECK:STDOUT: !111 = !DILocation(line: 19, column: 3, scope: !110)
// CHECK:STDOUT: !112 = !DILocation(line: 19, column: 19, scope: !110)
// CHECK:STDOUT: !116 = !DILocation(line: 20, column: 3, scope: !110)
// CHECK:STDOUT: !117 = !DILocation(line: 18, column: 1, scope: !110)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0f9f2a833d58b7bf", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !105, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !50)
// CHECK:STDOUT: !123 = !DILocation(line: 32, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT:
