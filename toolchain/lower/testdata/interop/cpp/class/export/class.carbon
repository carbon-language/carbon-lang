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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !139
// CHECK:STDOUT:   call void @_CCallLayoutA__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !139
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !141
// CHECK:STDOUT:   call void @_CCallLayoutB__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutBD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !141
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !143
// CHECK:STDOUT:   call void @_CCallLayoutC__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutCD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !143
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
// CHECK:STDOUT: define weak_odr void @"_COp.7811fb2f9dc58511:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_20.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !59
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_20.1.y), !dbg !59
// CHECK:STDOUT:   %.loc6_20.4.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !59
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %.loc6_20.4.x), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5a318796badc8ab7:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_20.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !65
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_20.1.y), !dbg !65
// CHECK:STDOUT:   %.loc6_20.4.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !65
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %.loc6_20.4.x), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutA__carbon_thunk.Main(ptr %_) #5 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %_, align 8, !dbg !69
// CHECK:STDOUT:   call void @_CCallLayoutA.Main(ptr %.loc21), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutB.Main"(ptr %self) #4 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd2d442e060c7dd0(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9ec3885d851a710b:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_15.1.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !77
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc11_15.1.b), !dbg !77
// CHECK:STDOUT:   %.loc11_15.4.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !77
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %.loc11_15.4.a), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.98b9397038fd95a2:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_15.1.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !81
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc11_15.1.b), !dbg !81
// CHECK:STDOUT:   %.loc11_15.4.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !81
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %.loc11_15.4.a), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutB__carbon_thunk.Main(ptr %_) #5 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22 = load ptr, ptr %_, align 8, !dbg !85
// CHECK:STDOUT:   call void @_CCallLayoutB.Main(ptr %.loc22), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutC.Main"(ptr %self) #4 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d869c4fcbd4e85a2(ptr %self), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fdd48c8f5e73335b:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_15.1.c = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !93
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc16_15.1.c), !dbg !93
// CHECK:STDOUT:   %.loc16_15.4.base = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !93
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %.loc16_15.4.base), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bb6bd82e06513595:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_15.1.c = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !97
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc16_15.1.c), !dbg !97
// CHECK:STDOUT:   %.loc16_15.4.base = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !97
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %.loc16_15.4.base), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutC__carbon_thunk.Main(ptr %_) #5 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %_, align 8, !dbg !101
// CHECK:STDOUT:   call void @_CCallLayoutC.Main(ptr %.loc23), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %t) #5 !dbg !103 {
// CHECK:STDOUT:   call void @"_COp.5a318796badc8ab7:core.SubobjectDestroy.Core"(ptr %t), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #5 !dbg !110 {
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #5 !dbg !117 {
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dd2d442e060c7dd0(ptr %t) #5 !dbg !124 {
// CHECK:STDOUT:   call void @"_COp.98b9397038fd95a2:core.SubobjectDestroy.Core"(ptr %t), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d869c4fcbd4e85a2(ptr %t) #5 !dbg !131 {
// CHECK:STDOUT:   call void @"_COp.bb6bd82e06513595:core.SubobjectDestroy.Core"(ptr %t), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon4Main7LayoutAD2Ev, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.ec5c5666af23bee2, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.4f29d15a20894e9b, { 1, 0 }
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
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
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
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7811fb2f9dc58511:core.SubobjectDestroy.Core", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !62)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !50)
// CHECK:STDOUT: !59 = !DILocation(line: 6, column: 1, scope: !57)
// CHECK:STDOUT: !60 = !{null, !37}
// CHECK:STDOUT: !61 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !62 = !{!58}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5a318796badc8ab7:core.SubobjectDestroy.Core", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !50)
// CHECK:STDOUT: !65 = !DILocation(line: 6, column: 1, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "CallLayoutA__carbon_thunk", linkageName: "_CCallLayoutA__carbon_thunk.Main", scope: null, file: !48, line: 21, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !50)
// CHECK:STDOUT: !69 = !DILocation(line: 21, column: 1, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutB.Main", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !50)
// CHECK:STDOUT: !73 = !DILocation(line: 11, column: 1, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9ec3885d851a710b:core.SubobjectDestroy.Core", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !50)
// CHECK:STDOUT: !77 = !DILocation(line: 11, column: 1, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Op", linkageName: "_COp.98b9397038fd95a2:core.SubobjectDestroy.Core", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !50)
// CHECK:STDOUT: !81 = !DILocation(line: 11, column: 1, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "CallLayoutB__carbon_thunk", linkageName: "_CCallLayoutB__carbon_thunk.Main", scope: null, file: !48, line: 22, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !50)
// CHECK:STDOUT: !85 = !DILocation(line: 22, column: 1, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutC.Main", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !50)
// CHECK:STDOUT: !89 = !DILocation(line: 16, column: 1, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd48c8f5e73335b:core.SubobjectDestroy.Core", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !50)
// CHECK:STDOUT: !93 = !DILocation(line: 16, column: 1, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bb6bd82e06513595:core.SubobjectDestroy.Core", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !50)
// CHECK:STDOUT: !97 = !DILocation(line: 16, column: 1, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "CallLayoutC__carbon_thunk", linkageName: "_CCallLayoutC__carbon_thunk.Main", scope: null, file: !48, line: 23, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !50)
// CHECK:STDOUT: !101 = !DILocation(line: 23, column: 1, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ec5c5666af23bee2", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !109)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !50)
// CHECK:STDOUT: !107 = !DILocation(line: 34, column: 5, scope: !103)
// CHECK:STDOUT: !108 = !DILocation(line: 32, column: 1, scope: !103)
// CHECK:STDOUT: !109 = !{!104}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !27, line: 32, type: !61, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !116)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !37)
// CHECK:STDOUT: !115 = !DILocation(line: 32, column: 1, scope: !110)
// CHECK:STDOUT: !116 = !{!111}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !123)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !50)
// CHECK:STDOUT: !122 = !DILocation(line: 32, column: 1, scope: !117)
// CHECK:STDOUT: !123 = !{!118}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dd2d442e060c7dd0", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !130)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !50)
// CHECK:STDOUT: !128 = !DILocation(line: 34, column: 5, scope: !124)
// CHECK:STDOUT: !129 = !DILocation(line: 32, column: 1, scope: !124)
// CHECK:STDOUT: !130 = !{!125}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d869c4fcbd4e85a2", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !137)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !50)
// CHECK:STDOUT: !135 = !DILocation(line: 34, column: 5, scope: !131)
// CHECK:STDOUT: !136 = !DILocation(line: 32, column: 1, scope: !131)
// CHECK:STDOUT: !137 = !{!132}
// CHECK:STDOUT: !138 = !{!"p1 _ZTSN6Carbon4Main7LayoutAE", !2, i64 0}
// CHECK:STDOUT: !139 = !{!138, !138, i64 0}
// CHECK:STDOUT: !140 = !{!"p1 _ZTSN6Carbon4Main7LayoutBE", !2, i64 0}
// CHECK:STDOUT: !141 = !{!140, !140, i64 0}
// CHECK:STDOUT: !142 = !{!"p1 _ZTSN6Carbon4Main7LayoutCE", !2, i64 0}
// CHECK:STDOUT: !143 = !{!142, !142, i64 0}
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
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.FieldC.Main"(ptr %self) #1 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %self), !dbg !148
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2508ed710019eba8:core.SubobjectDestroy.Core"(ptr %self) #2 !dbg !150 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_14.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !152
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_14.1.b), !dbg !152
// CHECK:STDOUT:   %.loc6_14.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !152
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_14.4.a), !dbg !152
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.718874c3c75b803e:core.SubobjectDestroy.Core"(ptr %self) #2 !dbg !154 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_14.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !156
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_14.1.b), !dbg !156
// CHECK:STDOUT:   %.loc6_14.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !156
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_14.4.a), !dbg !156
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #2 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, i32 }, align 4, !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !159
// CHECK:STDOUT:   %.loc19_34.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 0, !dbg !160
// CHECK:STDOUT:   %.loc19_34.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 1, !dbg !160
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @FieldC.val.loc19_3.1, i64 8, i1 false), !dbg !159
// CHECK:STDOUT:   call void @_Z1FRN6Carbon4Main6FieldCE(ptr %c.var), !dbg !164
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %c.var), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %t) #2 !dbg !166 {
// CHECK:STDOUT:   call void @"_COp.718874c3c75b803e:core.SubobjectDestroy.Core"(ptr %t), !dbg !170
// CHECK:STDOUT:   ret void, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #2 !dbg !173 {
// CHECK:STDOUT:   ret void, !dbg !178
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!145}
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
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
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
// CHECK:STDOUT: !60 = !{null, !37}
// CHECK:STDOUT: !61 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !144 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !145 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !144, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.FieldC.Main", scope: null, file: !144, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !149)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !50)
// CHECK:STDOUT: !148 = !DILocation(line: 6, column: 1, scope: !146)
// CHECK:STDOUT: !149 = !{!147}
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2508ed710019eba8:core.SubobjectDestroy.Core", scope: null, file: !144, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !153)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !150, type: !50)
// CHECK:STDOUT: !152 = !DILocation(line: 6, column: 1, scope: !150)
// CHECK:STDOUT: !153 = !{!151}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Op", linkageName: "_COp.718874c3c75b803e:core.SubobjectDestroy.Core", scope: null, file: !144, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !157)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !50)
// CHECK:STDOUT: !156 = !DILocation(line: 6, column: 1, scope: !154)
// CHECK:STDOUT: !157 = !{!155}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !144, line: 18, type: !39, spFlags: DISPFlagDefinition, unit: !145)
// CHECK:STDOUT: !159 = !DILocation(line: 19, column: 3, scope: !158)
// CHECK:STDOUT: !160 = !DILocation(line: 19, column: 19, scope: !158)
// CHECK:STDOUT: !164 = !DILocation(line: 20, column: 3, scope: !158)
// CHECK:STDOUT: !165 = !DILocation(line: 18, column: 1, scope: !158)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0f9f2a833d58b7bf", scope: null, file: !27, line: 32, type: !52, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !172)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !166, type: !50)
// CHECK:STDOUT: !170 = !DILocation(line: 34, column: 5, scope: !166)
// CHECK:STDOUT: !171 = !DILocation(line: 32, column: 1, scope: !166)
// CHECK:STDOUT: !172 = !{!167}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !27, line: 32, type: !61, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !179)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !37)
// CHECK:STDOUT: !178 = !DILocation(line: 32, column: 1, scope: !173)
// CHECK:STDOUT: !179 = !{!174}
// CHECK:STDOUT:
