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
// CHECK:STDOUT: define dso_local noundef ptr @_Z4passPN6Carbon1AE(ptr noundef %p) #0 {
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
// CHECK:STDOUT:   %call = call noundef ptr @_Z4passPN6Carbon1AE(ptr noundef null)
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
// CHECK:STDOUT: !3 = !{!"p1 _ZTSN6Carbon1AE", !2, i64 0}
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
// CHECK:STDOUT: %"class.Carbon::LayoutA" = type <{ ptr, i32 }>
// CHECK:STDOUT: %"class.Carbon::LayoutB" = type { %"class.Carbon::LayoutA", i32 }
// CHECK:STDOUT: %"class.Carbon::LayoutC" = type { %"class.Carbon::LayoutA", i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon7LayoutAD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon7LayoutBD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon7LayoutCD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a = alloca %"class.Carbon::LayoutA", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a) #5
// CHECK:STDOUT:   call void @_ZN6CarbonL11CallLayoutAEPNS_7LayoutAE(ptr noundef %a)
// CHECK:STDOUT:   call void @_ZN6Carbon7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %a) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL11CallLayoutAEPNS_7LayoutAE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !144
// CHECK:STDOUT:   call void @_CCallLayoutA__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !144
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
// CHECK:STDOUT:   %b = alloca %"class.Carbon::LayoutB", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b) #5
// CHECK:STDOUT:   call void @_ZN6CarbonL11CallLayoutBEPNS_7LayoutBE(ptr noundef %b)
// CHECK:STDOUT:   call void @_ZN6Carbon7LayoutBD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %b) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL11CallLayoutBEPNS_7LayoutBE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !146
// CHECK:STDOUT:   call void @_CCallLayoutB__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon7LayoutBD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !146
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.LayoutB.Main"(ptr noundef nonnull align 8 dereferenceable(16) %this1)
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %"class.Carbon::LayoutB", ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   call void @_ZN6Carbon7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %a) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1hv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c = alloca %"class.Carbon::LayoutC", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c) #5
// CHECK:STDOUT:   call void @_ZN6CarbonL11CallLayoutCEPNS_7LayoutCE(ptr noundef %c)
// CHECK:STDOUT:   call void @_ZN6CarbonL11CallLayoutAEPNS_7LayoutAE(ptr noundef %c)
// CHECK:STDOUT:   call void @_ZN6Carbon7LayoutCD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %c) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %c) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL11CallLayoutCEPNS_7LayoutCE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !148
// CHECK:STDOUT:   call void @_CCallLayoutC__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon7LayoutCD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !148
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.LayoutC.Main"(ptr noundef nonnull align 8 dereferenceable(16) %this1)
// CHECK:STDOUT:   call void @_ZN6Carbon7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this1) #5
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
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutA.Main"(ptr %self) #4 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.5a318796badc8ab7:core.Destroy.Core"(ptr %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #5 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #5 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #5 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #5 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.63f0d36eb067b612:core.Destroy.Core"(ptr %self) #5 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_20.1.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !77
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %.loc6_20.1.x), !dbg !77
// CHECK:STDOUT:   %.loc6_20.2.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !77
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc6_20.2.y), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.63f0d36eb067b612:core.Destroy.Core"(ptr %self) #5 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.63f0d36eb067b612:core.Destroy.Core"(ptr %self), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5a318796badc8ab7:core.Destroy.Core"(ptr %self) #5 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5a318796badc8ab7:core.Destroy.Core"(ptr %self) #5 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5a318796badc8ab7:core.Destroy.Core"(ptr %self), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutA__carbon_thunk.Main(ptr %_) #5 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %_, align 8, !dbg !93
// CHECK:STDOUT:   call void @_CCallLayoutA.Main(ptr %.loc21), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutB.Main"(ptr %self) #4 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.98b9397038fd95a2:core.Destroy.Core"(ptr %self), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.680fc8e588bd03e0:core.Destroy.Core"(ptr %self) #5 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_15.1.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !101
// CHECK:STDOUT:   call void @"_CSelfDestruct.5a318796badc8ab7:core.Destroy.Core"(ptr %.loc11_15.1.a), !dbg !101
// CHECK:STDOUT:   %.loc11_15.2.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !101
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc11_15.2.b), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.680fc8e588bd03e0:core.Destroy.Core"(ptr %self) #5 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.680fc8e588bd03e0:core.Destroy.Core"(ptr %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.98b9397038fd95a2:core.Destroy.Core"(ptr %self) #5 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.98b9397038fd95a2:core.Destroy.Core"(ptr %self) #5 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.98b9397038fd95a2:core.Destroy.Core"(ptr %self), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutB__carbon_thunk.Main(ptr %_) #5 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22 = load ptr, ptr %_, align 8, !dbg !117
// CHECK:STDOUT:   call void @_CCallLayoutB.Main(ptr %.loc22), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutC.Main"(ptr %self) #4 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.bb6bd82e06513595:core.Destroy.Core"(ptr %self), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d1ce4198d9e72e02:core.Destroy.Core"(ptr %self) #5 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_15.1.base = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !125
// CHECK:STDOUT:   call void @"_CSelfDestruct.5a318796badc8ab7:core.Destroy.Core"(ptr %.loc16_15.1.base), !dbg !125
// CHECK:STDOUT:   %.loc16_15.2.c = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !125
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc16_15.2.c), !dbg !125
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d1ce4198d9e72e02:core.Destroy.Core"(ptr %self) #5 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d1ce4198d9e72e02:core.Destroy.Core"(ptr %self), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bb6bd82e06513595:core.Destroy.Core"(ptr %self) #5 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb6bd82e06513595:core.Destroy.Core"(ptr %self) #5 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.bb6bd82e06513595:core.Destroy.Core"(ptr %self), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallLayoutC__carbon_thunk.Main(ptr %_) #5 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %_, align 8, !dbg !141
// CHECK:STDOUT:   call void @_CCallLayoutC.Main(ptr %.loc23), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon7LayoutAD2Ev, { 2, 1, 0 }
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
// CHECK:STDOUT: !53 = !{null, !37}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutA.Main", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !50)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 1, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !50)
// CHECK:STDOUT: !61 = !DILocation(line: 6, column: 1, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !48, line: 6, type: !54, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !37)
// CHECK:STDOUT: !65 = !DILocation(line: 6, column: 1, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !48, line: 6, type: !54, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !37)
// CHECK:STDOUT: !69 = !DILocation(line: 6, column: 1, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !48, line: 6, type: !54, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !37)
// CHECK:STDOUT: !73 = !DILocation(line: 6, column: 1, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.63f0d36eb067b612:core.Destroy.Core", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !50)
// CHECK:STDOUT: !77 = !DILocation(line: 6, column: 1, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.63f0d36eb067b612:core.Destroy.Core", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !50)
// CHECK:STDOUT: !81 = !DILocation(line: 6, column: 1, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5a318796badc8ab7:core.Destroy.Core", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !50)
// CHECK:STDOUT: !85 = !DILocation(line: 6, column: 1, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5a318796badc8ab7:core.Destroy.Core", scope: null, file: !48, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !50)
// CHECK:STDOUT: !89 = !DILocation(line: 6, column: 1, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "CallLayoutA__carbon_thunk", linkageName: "_CCallLayoutA__carbon_thunk.Main", scope: null, file: !48, line: 21, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !50)
// CHECK:STDOUT: !93 = !DILocation(line: 21, column: 1, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutB.Main", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !50)
// CHECK:STDOUT: !97 = !DILocation(line: 11, column: 1, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.680fc8e588bd03e0:core.Destroy.Core", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !50)
// CHECK:STDOUT: !101 = !DILocation(line: 11, column: 1, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.680fc8e588bd03e0:core.Destroy.Core", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !50)
// CHECK:STDOUT: !105 = !DILocation(line: 11, column: 1, scope: !103)
// CHECK:STDOUT: !106 = !{!104}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.98b9397038fd95a2:core.Destroy.Core", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !50)
// CHECK:STDOUT: !109 = !DILocation(line: 11, column: 1, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.98b9397038fd95a2:core.Destroy.Core", scope: null, file: !48, line: 11, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !50)
// CHECK:STDOUT: !113 = !DILocation(line: 11, column: 1, scope: !111)
// CHECK:STDOUT: !114 = !{!112}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "CallLayoutB__carbon_thunk", linkageName: "_CCallLayoutB__carbon_thunk.Main", scope: null, file: !48, line: 22, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !118)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !50)
// CHECK:STDOUT: !117 = !DILocation(line: 22, column: 1, scope: !115)
// CHECK:STDOUT: !118 = !{!116}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutC.Main", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !122)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !50)
// CHECK:STDOUT: !121 = !DILocation(line: 16, column: 1, scope: !119)
// CHECK:STDOUT: !122 = !{!120}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d1ce4198d9e72e02:core.Destroy.Core", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !126)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !50)
// CHECK:STDOUT: !125 = !DILocation(line: 16, column: 1, scope: !123)
// CHECK:STDOUT: !126 = !{!124}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d1ce4198d9e72e02:core.Destroy.Core", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !130)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !50)
// CHECK:STDOUT: !129 = !DILocation(line: 16, column: 1, scope: !127)
// CHECK:STDOUT: !130 = !{!128}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bb6bd82e06513595:core.Destroy.Core", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !50)
// CHECK:STDOUT: !133 = !DILocation(line: 16, column: 1, scope: !131)
// CHECK:STDOUT: !134 = !{!132}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb6bd82e06513595:core.Destroy.Core", scope: null, file: !48, line: 16, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !138)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !50)
// CHECK:STDOUT: !137 = !DILocation(line: 16, column: 1, scope: !135)
// CHECK:STDOUT: !138 = !{!136}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "CallLayoutC__carbon_thunk", linkageName: "_CCallLayoutC__carbon_thunk.Main", scope: null, file: !48, line: 23, type: !52, spFlags: DISPFlagDefinition, unit: !49, retainedNodes: !142)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !50)
// CHECK:STDOUT: !141 = !DILocation(line: 23, column: 1, scope: !139)
// CHECK:STDOUT: !142 = !{!140}
// CHECK:STDOUT: !143 = !{!"p1 _ZTSN6Carbon7LayoutAE", !2, i64 0}
// CHECK:STDOUT: !144 = !{!143, !143, i64 0}
// CHECK:STDOUT: !145 = !{!"p1 _ZTSN6Carbon7LayoutBE", !2, i64 0}
// CHECK:STDOUT: !146 = !{!145, !145, i64 0}
// CHECK:STDOUT: !147 = !{!"p1 _ZTSN6Carbon7LayoutCE", !2, i64 0}
// CHECK:STDOUT: !148 = !{!147, !147, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::FieldC" = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: @FieldC.val.loc19_3 = internal constant { i32, i32 } zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1FRN6Carbon6FieldCE(ptr noundef nonnull align 4 dereferenceable(8) %c) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %c.addr, align 8, !tbaa !11, !nonnull !13, !align !14
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %"class.Carbon::FieldC", ptr %0, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %a, align 4, !tbaa !15
// CHECK:STDOUT:   %1 = load ptr, ptr %c.addr, align 8, !tbaa !11, !nonnull !13, !align !14
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %"class.Carbon::FieldC", ptr %1, i32 0, i32 1
// CHECK:STDOUT:   store i32 34, ptr %b, align 4, !tbaa !16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.FieldC.Main"(ptr %self) #1 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.718874c3c75b803e:core.Destroy.Core"(ptr %self), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core"(ptr %self) #2 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #2 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #2 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core"(ptr %self) #2 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_14.1.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !169
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc6_14.1.a), !dbg !169
// CHECK:STDOUT:   %.loc6_14.2.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !169
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc6_14.2.b), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b4c762864cd9a02b:core.Destroy.Core"(ptr %self) #2 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core"(ptr %self), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.718874c3c75b803e:core.Destroy.Core"(ptr %self) #2 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.718874c3c75b803e:core.Destroy.Core"(ptr %self) #2 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.718874c3c75b803e:core.Destroy.Core"(ptr %self), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #2 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, i32 }, align 4, !dbg !184
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !184
// CHECK:STDOUT:   %.loc19_34.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 0, !dbg !185
// CHECK:STDOUT:   %.loc19_34.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 1, !dbg !185
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @FieldC.val.loc19_3, i64 8, i1 false), !dbg !184
// CHECK:STDOUT:   call void @_Z1FRN6Carbon6FieldCE(ptr %c.var), !dbg !189
// CHECK:STDOUT:   call void @"_CSelfDestruct.718874c3c75b803e:core.Destroy.Core"(ptr %c.var), !dbg !184
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!150}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !19, !20}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !5 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !10 = !{!"p1 _ZTSN6Carbon6FieldCE", !2, i64 0}
// CHECK:STDOUT: !11 = !{!10, !10, i64 0}
// CHECK:STDOUT: !12 = !{!"_ZTSN6Carbon6FieldCE", !5, i64 0, !5, i64 4}
// CHECK:STDOUT: !13 = !{}
// CHECK:STDOUT: !14 = !{i64 4}
// CHECK:STDOUT: !15 = !{!12, !5, i64 0}
// CHECK:STDOUT: !16 = !{!12, !5, i64 4}
// CHECK:STDOUT: !19 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !20 = !{i32 2, !"Debug Info Version", i32 3}
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
// CHECK:STDOUT: !53 = !{null, !37}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !149 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !150 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !149, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.FieldC.Main", scope: null, file: !149, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !154)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !50)
// CHECK:STDOUT: !153 = !DILocation(line: 6, column: 1, scope: !151)
// CHECK:STDOUT: !154 = !{!152}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6dae29ece9a708c4:core.Destroy.Core", scope: null, file: !149, line: 6, type: !54, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !37)
// CHECK:STDOUT: !157 = !DILocation(line: 6, column: 1, scope: !155)
// CHECK:STDOUT: !158 = !{!156}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !149, line: 6, type: !54, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !37)
// CHECK:STDOUT: !161 = !DILocation(line: 6, column: 1, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !149, line: 6, type: !54, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !37)
// CHECK:STDOUT: !165 = !DILocation(line: 6, column: 1, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core", scope: null, file: !149, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !50)
// CHECK:STDOUT: !169 = !DILocation(line: 6, column: 1, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b4c762864cd9a02b:core.Destroy.Core", scope: null, file: !149, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !50)
// CHECK:STDOUT: !173 = !DILocation(line: 6, column: 1, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.718874c3c75b803e:core.Destroy.Core", scope: null, file: !149, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !178)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !50)
// CHECK:STDOUT: !177 = !DILocation(line: 6, column: 1, scope: !175)
// CHECK:STDOUT: !178 = !{!176}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.718874c3c75b803e:core.Destroy.Core", scope: null, file: !149, line: 6, type: !52, spFlags: DISPFlagDefinition, unit: !150, retainedNodes: !182)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !50)
// CHECK:STDOUT: !181 = !DILocation(line: 6, column: 1, scope: !179)
// CHECK:STDOUT: !182 = !{!180}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !149, line: 18, type: !39, spFlags: DISPFlagDefinition, unit: !150)
// CHECK:STDOUT: !184 = !DILocation(line: 19, column: 3, scope: !183)
// CHECK:STDOUT: !185 = !DILocation(line: 19, column: 19, scope: !183)
// CHECK:STDOUT: !189 = !DILocation(line: 20, column: 3, scope: !183)
// CHECK:STDOUT: !190 = !DILocation(line: 18, column: 1, scope: !183)
// CHECK:STDOUT:
