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
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !46, !19, !20}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
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
// CHECK:STDOUT: !43 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !5, i64 0}
// CHECK:STDOUT: !48 = !{!47, !5, i64 0}
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !140
// CHECK:STDOUT:   call void @_CCallLayoutA__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutAD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !140
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !142
// CHECK:STDOUT:   call void @_CCallLayoutB__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutBD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !142
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !144
// CHECK:STDOUT:   call void @_CCallLayoutC__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4Main7LayoutCD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !144
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
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutA.Main"(ptr %self) #4 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %self), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7811fb2f9dc58511:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_20.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !60
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_20.1.y), !dbg !60
// CHECK:STDOUT:   %.loc6_20.4.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !60
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %.loc6_20.4.x), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5a318796badc8ab7:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_20.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_20.1.y), !dbg !66
// CHECK:STDOUT:   %.loc6_20.4.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %.loc6_20.4.x), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallLayoutA__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %_) #6 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %_, align 8, !dbg !70
// CHECK:STDOUT:   call void @_CCallLayoutA.Main(ptr %.loc21), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutB.Main"(ptr %self) #4 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd2d442e060c7dd0(ptr %self), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9ec3885d851a710b:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_15.1.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !78
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc11_15.1.b), !dbg !78
// CHECK:STDOUT:   %.loc11_15.4.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !78
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %.loc11_15.4.a), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.98b9397038fd95a2:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_15.1.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !82
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc11_15.1.b), !dbg !82
// CHECK:STDOUT:   %.loc11_15.4.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !82
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %.loc11_15.4.a), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallLayoutB__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %_) #6 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22 = load ptr, ptr %_, align 8, !dbg !86
// CHECK:STDOUT:   call void @_CCallLayoutB.Main(ptr %.loc22), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.LayoutC.Main"(ptr %self) #4 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d869c4fcbd4e85a2(ptr %self), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fdd48c8f5e73335b:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_15.1.c = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc16_15.1.c), !dbg !94
// CHECK:STDOUT:   %.loc16_15.4.base = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %.loc16_15.4.base), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bb6bd82e06513595:core.SubobjectDestroy.Core"(ptr %self) #5 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_15.1.c = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc16_15.1.c), !dbg !98
// CHECK:STDOUT:   %.loc16_15.4.base = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %.loc16_15.4.base), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallLayoutC__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %_) #6 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %_, align 8, !dbg !102
// CHECK:STDOUT:   call void @_CCallLayoutC.Main(ptr %.loc23), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ec5c5666af23bee2(ptr %t) #5 !dbg !104 {
// CHECK:STDOUT:   call void @"_COp.5a318796badc8ab7:core.SubobjectDestroy.Core"(ptr %t), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #5 !dbg !111 {
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #5 !dbg !118 {
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dd2d442e060c7dd0(ptr %t) #5 !dbg !125 {
// CHECK:STDOUT:   call void @"_COp.98b9397038fd95a2:core.SubobjectDestroy.Core"(ptr %t), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d869c4fcbd4e85a2(ptr %t) #5 !dbg !132 {
// CHECK:STDOUT:   call void @"_COp.bb6bd82e06513595:core.SubobjectDestroy.Core"(ptr %t), !dbg !136
// CHECK:STDOUT:   ret void, !dbg !137
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
// CHECK:STDOUT: attributes #6 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!50}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !46, !19, !20}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !5 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !19 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !20 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !27 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !5, i64 0}
// CHECK:STDOUT: !48 = !{!47, !5, i64 0}
// CHECK:STDOUT: !49 = !DIFile(filename: "layout.carbon", directory: "")
// CHECK:STDOUT: !50 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !49, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !51 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !52 = !{null, !51}
// CHECK:STDOUT: !53 = !DISubroutineType(types: !52)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutA.Main", scope: null, file: !49, line: 6, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !51)
// CHECK:STDOUT: !56 = !DILocation(line: 6, column: 1, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7811fb2f9dc58511:core.SubobjectDestroy.Core", scope: null, file: !49, line: 6, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !63)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !51)
// CHECK:STDOUT: !60 = !DILocation(line: 6, column: 1, scope: !58)
// CHECK:STDOUT: !61 = !{null, !37}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !63 = !{!59}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5a318796badc8ab7:core.SubobjectDestroy.Core", scope: null, file: !49, line: 6, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !67)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !51)
// CHECK:STDOUT: !66 = !DILocation(line: 6, column: 1, scope: !64)
// CHECK:STDOUT: !67 = !{!65}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "CallLayoutA__carbon_thunk", linkageName: "_CCallLayoutA__carbon_thunk.Main", scope: null, file: !49, line: 21, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !51)
// CHECK:STDOUT: !70 = !DILocation(line: 21, column: 1, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutB.Main", scope: null, file: !49, line: 11, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !51)
// CHECK:STDOUT: !74 = !DILocation(line: 11, column: 1, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9ec3885d851a710b:core.SubobjectDestroy.Core", scope: null, file: !49, line: 11, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !79)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !51)
// CHECK:STDOUT: !78 = !DILocation(line: 11, column: 1, scope: !76)
// CHECK:STDOUT: !79 = !{!77}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "Op", linkageName: "_COp.98b9397038fd95a2:core.SubobjectDestroy.Core", scope: null, file: !49, line: 11, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !83)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !51)
// CHECK:STDOUT: !82 = !DILocation(line: 11, column: 1, scope: !80)
// CHECK:STDOUT: !83 = !{!81}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "CallLayoutB__carbon_thunk", linkageName: "_CCallLayoutB__carbon_thunk.Main", scope: null, file: !49, line: 22, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !87)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !51)
// CHECK:STDOUT: !86 = !DILocation(line: 22, column: 1, scope: !84)
// CHECK:STDOUT: !87 = !{!85}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.LayoutC.Main", scope: null, file: !49, line: 16, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !91)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !51)
// CHECK:STDOUT: !90 = !DILocation(line: 16, column: 1, scope: !88)
// CHECK:STDOUT: !91 = !{!89}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd48c8f5e73335b:core.SubobjectDestroy.Core", scope: null, file: !49, line: 16, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !51)
// CHECK:STDOUT: !94 = !DILocation(line: 16, column: 1, scope: !92)
// CHECK:STDOUT: !95 = !{!93}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bb6bd82e06513595:core.SubobjectDestroy.Core", scope: null, file: !49, line: 16, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !51)
// CHECK:STDOUT: !98 = !DILocation(line: 16, column: 1, scope: !96)
// CHECK:STDOUT: !99 = !{!97}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "CallLayoutC__carbon_thunk", linkageName: "_CCallLayoutC__carbon_thunk.Main", scope: null, file: !49, line: 23, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !103)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !51)
// CHECK:STDOUT: !102 = !DILocation(line: 23, column: 1, scope: !100)
// CHECK:STDOUT: !103 = !{!101}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ec5c5666af23bee2", scope: null, file: !27, line: 32, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !51)
// CHECK:STDOUT: !108 = !DILocation(line: 34, column: 5, scope: !104)
// CHECK:STDOUT: !109 = !DILocation(line: 32, column: 1, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !27, line: 32, type: !62, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !117)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !37)
// CHECK:STDOUT: !116 = !DILocation(line: 32, column: 1, scope: !111)
// CHECK:STDOUT: !117 = !{!112}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !27, line: 32, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !51)
// CHECK:STDOUT: !123 = !DILocation(line: 32, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dd2d442e060c7dd0", scope: null, file: !27, line: 32, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !131)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !51)
// CHECK:STDOUT: !129 = !DILocation(line: 34, column: 5, scope: !125)
// CHECK:STDOUT: !130 = !DILocation(line: 32, column: 1, scope: !125)
// CHECK:STDOUT: !131 = !{!126}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d869c4fcbd4e85a2", scope: null, file: !27, line: 32, type: !53, spFlags: DISPFlagDefinition, unit: !50, retainedNodes: !138)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !51)
// CHECK:STDOUT: !136 = !DILocation(line: 34, column: 5, scope: !132)
// CHECK:STDOUT: !137 = !DILocation(line: 32, column: 1, scope: !132)
// CHECK:STDOUT: !138 = !{!133}
// CHECK:STDOUT: !139 = !{!"p1 _ZTSN6Carbon4Main7LayoutAE", !2, i64 0}
// CHECK:STDOUT: !140 = !{!139, !139, i64 0}
// CHECK:STDOUT: !141 = !{!"p1 _ZTSN6Carbon4Main7LayoutBE", !2, i64 0}
// CHECK:STDOUT: !142 = !{!141, !141, i64 0}
// CHECK:STDOUT: !143 = !{!"p1 _ZTSN6Carbon4Main7LayoutCE", !2, i64 0}
// CHECK:STDOUT: !144 = !{!143, !143, i64 0}
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
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.FieldC.Main"(ptr %self) #1 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %self), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2508ed710019eba8:core.SubobjectDestroy.Core"(ptr %self) #2 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_14.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !153
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_14.1.b), !dbg !153
// CHECK:STDOUT:   %.loc6_14.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !153
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_14.4.a), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.718874c3c75b803e:core.SubobjectDestroy.Core"(ptr %self) #2 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_14.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !157
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_14.1.b), !dbg !157
// CHECK:STDOUT:   %.loc6_14.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !157
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc6_14.4.a), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #2 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, i32 }, align 4, !dbg !160
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !160
// CHECK:STDOUT:   %.loc19_34.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 0, !dbg !161
// CHECK:STDOUT:   %.loc19_34.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 1, !dbg !161
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @FieldC.val.loc19_3.1, i64 8, i1 false), !dbg !160
// CHECK:STDOUT:   call void @_Z1FRN6Carbon4Main6FieldCE(ptr %c.var), !dbg !165
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %c.var), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0f9f2a833d58b7bf(ptr %t) #2 !dbg !167 {
// CHECK:STDOUT:   call void @"_COp.718874c3c75b803e:core.SubobjectDestroy.Core"(ptr %t), !dbg !171
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #2 !dbg !174 {
// CHECK:STDOUT:   ret void, !dbg !179
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!146}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !46, !19, !20}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!48}
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
// CHECK:STDOUT: !43 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"__libc_errno", !5, i64 0}
// CHECK:STDOUT: !48 = !{!47, !5, i64 0}
// CHECK:STDOUT: !51 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !52 = !{null, !51}
// CHECK:STDOUT: !53 = !DISubroutineType(types: !52)
// CHECK:STDOUT: !61 = !{null, !37}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !145 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !146 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !145, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.FieldC.Main", scope: null, file: !145, line: 6, type: !53, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !150)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !51)
// CHECK:STDOUT: !149 = !DILocation(line: 6, column: 1, scope: !147)
// CHECK:STDOUT: !150 = !{!148}
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2508ed710019eba8:core.SubobjectDestroy.Core", scope: null, file: !145, line: 6, type: !53, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !154)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !51)
// CHECK:STDOUT: !153 = !DILocation(line: 6, column: 1, scope: !151)
// CHECK:STDOUT: !154 = !{!152}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "Op", linkageName: "_COp.718874c3c75b803e:core.SubobjectDestroy.Core", scope: null, file: !145, line: 6, type: !53, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !51)
// CHECK:STDOUT: !157 = !DILocation(line: 6, column: 1, scope: !155)
// CHECK:STDOUT: !158 = !{!156}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !145, line: 18, type: !39, spFlags: DISPFlagDefinition, unit: !146)
// CHECK:STDOUT: !160 = !DILocation(line: 19, column: 3, scope: !159)
// CHECK:STDOUT: !161 = !DILocation(line: 19, column: 19, scope: !159)
// CHECK:STDOUT: !165 = !DILocation(line: 20, column: 3, scope: !159)
// CHECK:STDOUT: !166 = !DILocation(line: 18, column: 1, scope: !159)
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0f9f2a833d58b7bf", scope: null, file: !27, line: 32, type: !53, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !173)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !51)
// CHECK:STDOUT: !171 = !DILocation(line: 34, column: 5, scope: !167)
// CHECK:STDOUT: !172 = !DILocation(line: 32, column: 1, scope: !167)
// CHECK:STDOUT: !173 = !{!168}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !27, line: 32, type: !62, spFlags: DISPFlagDefinition, unit: !146, retainedNodes: !180)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !37)
// CHECK:STDOUT: !179 = !DILocation(line: 32, column: 1, scope: !174)
// CHECK:STDOUT: !180 = !{!175}
// CHECK:STDOUT:
