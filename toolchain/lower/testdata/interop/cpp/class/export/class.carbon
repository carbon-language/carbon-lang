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

base class A {
  var x: i32*;
  var y: i32;
}

class B {
  var a: A;
  var b: i32;
}

class C {
  extend base: A;
  var c: i32;
}

fn CallA(a: A*);
fn CallB(b: B*);
fn CallC(b: C*);

inline Cpp '''
void f() {
  Carbon::A a;
  Carbon::CallA(&a);
}

void g() {
  Carbon::B b;
  Carbon::CallB(&b);
}

void h() {
  Carbon::C c;
  Carbon::CallC(&c);
  Carbon::CallA(&c);
}
''';

// --- field.carbon

library "[[@TEST_NAME]]";

import Cpp;

class C {
  var a: i32;
  var b: i32;
}

inline Cpp '''
void F(Carbon::C& c) {
  c.a = 12;
  c.b = 34;
}
''';

fn G() {
  var c: C = {.a = 0, .b = 0};
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
// CHECK:STDOUT:   store ptr %p, ptr %p.addr, align 8, !tbaa !12
// CHECK:STDOUT:   %0 = load ptr, ptr %p.addr, align 8, !tbaa !12
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
// CHECK:STDOUT: define void @_CDoIt.Main() #1 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4callv(), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "pointer.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = !{!13, !13, i64 0}
// CHECK:STDOUT: !13 = !{!"p1 _ZTSN6Carbon1AE", !14, i64 0}
// CHECK:STDOUT: !14 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "DoIt", linkageName: "_CDoIt.Main", scope: null, file: !1, line: 15, type: !16, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null}
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 13, scope: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 15, column: 1, scope: !15)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'layout.carbon'
// CHECK:STDOUT: source_filename = "layout.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::A" = type <{ ptr, i32 }>
// CHECK:STDOUT: %"class.Carbon::B" = type { %"class.Carbon::A", i32 }
// CHECK:STDOUT: %"class.Carbon::C" = type { %"class.Carbon::A", i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon1AD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon1BD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon1CD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a = alloca %"class.Carbon::A", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a) #5
// CHECK:STDOUT:   call void @_ZN6CarbonL5CallAEPNS_1AE(ptr noundef %a)
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %a) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL5CallAEPNS_1AE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !12
// CHECK:STDOUT:   call void @_CCallA__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !12
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.A.Main"(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b = alloca %"class.Carbon::B", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b) #5
// CHECK:STDOUT:   call void @_ZN6CarbonL5CallBEPNS_1BE(ptr noundef %b)
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %b) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL5CallBEPNS_1BE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !15
// CHECK:STDOUT:   call void @_CCallB__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !15
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.B.Main"(ptr noundef nonnull align 8 dereferenceable(16) %this1)
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %"class.Carbon::B", ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %a) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1hv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c = alloca %"class.Carbon::C", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c) #5
// CHECK:STDOUT:   call void @_ZN6CarbonL5CallCEPNS_1CE(ptr noundef %c)
// CHECK:STDOUT:   call void @_ZN6CarbonL5CallAEPNS_1AE(ptr noundef %c)
// CHECK:STDOUT:   call void @_ZN6Carbon1CD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %c) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %c) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL5CallCEPNS_1CE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !17
// CHECK:STDOUT:   call void @_CCallC__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon1CD2Ev(ptr noundef nonnull align 8 dead_on_return(16) dereferenceable(16) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !17
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.C.Main"(ptr noundef nonnull align 8 dereferenceable(16) %this1)
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this1) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallA.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallB.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallC.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.A.Main"(ptr %self) #4 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.e649d414b9e7644a:core.Destroy.Core"(ptr %self), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #5 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.21390e61793948a2:core.Destroy.Core"(ptr %self) #5 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e649d414b9e7644a:core.Destroy.Core"(ptr %self) #5 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallA__carbon_thunk.Main(ptr %_) #5 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %_, align 8, !dbg !44
// CHECK:STDOUT:   call void @_CCallA.Main(ptr %.loc21), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.B.Main"(ptr %self) #4 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.535580a8a7a96b3b:core.Destroy.Core"(ptr %self), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8e58843eeaf37d77:core.Destroy.Core"(ptr %self) #5 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.535580a8a7a96b3b:core.Destroy.Core"(ptr %self) #5 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallB__carbon_thunk.Main(ptr %_) #5 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22 = load ptr, ptr %_, align 8, !dbg !60
// CHECK:STDOUT:   call void @_CCallB.Main(ptr %.loc22), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.C.Main"(ptr %self) #4 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cf9748c540fa2b51:core.Destroy.Core"(ptr %self) #5 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self) #5 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallC__carbon_thunk.Main(ptr %_) #5 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %_, align 8, !dbg !76
// CHECK:STDOUT:   call void @_CCallC.Main(ptr %.loc23), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon1AD2Ev, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #5 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "layout.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = !{!13, !13, i64 0}
// CHECK:STDOUT: !13 = !{!"p1 _ZTSN6Carbon1AE", !14, i64 0}
// CHECK:STDOUT: !14 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !15 = !{!16, !16, i64 0}
// CHECK:STDOUT: !16 = !{!"p1 _ZTSN6Carbon1BE", !14, i64 0}
// CHECK:STDOUT: !17 = !{!18, !18, i64 0}
// CHECK:STDOUT: !18 = !{!"p1 _ZTSN6Carbon1CE", !14, i64 0}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.A.Main", scope: null, file: !1, line: 6, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 6, column: 1, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 6, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{null, !29}
// CHECK:STDOUT: !29 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !26, type: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 6, column: 1, scope: !26)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.21390e61793948a2:core.Destroy.Core", scope: null, file: !1, line: 6, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 6, column: 1, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e649d414b9e7644a:core.Destroy.Core", scope: null, file: !1, line: 6, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !22)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 1, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "CallA__carbon_thunk", linkageName: "_CCallA__carbon_thunk.Main", scope: null, file: !1, line: 21, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !22)
// CHECK:STDOUT: !44 = !DILocation(line: 21, column: 1, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.B.Main", scope: null, file: !1, line: 11, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !22)
// CHECK:STDOUT: !48 = !DILocation(line: 11, column: 1, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8e58843eeaf37d77:core.Destroy.Core", scope: null, file: !1, line: 11, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !22)
// CHECK:STDOUT: !52 = !DILocation(line: 11, column: 1, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.535580a8a7a96b3b:core.Destroy.Core", scope: null, file: !1, line: 11, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !22)
// CHECK:STDOUT: !56 = !DILocation(line: 11, column: 1, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "CallB__carbon_thunk", linkageName: "_CCallB__carbon_thunk.Main", scope: null, file: !1, line: 22, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !22)
// CHECK:STDOUT: !60 = !DILocation(line: 22, column: 1, scope: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.C.Main", scope: null, file: !1, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !22)
// CHECK:STDOUT: !64 = !DILocation(line: 16, column: 1, scope: !61)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cf9748c540fa2b51:core.Destroy.Core", scope: null, file: !1, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !66)
// CHECK:STDOUT: !66 = !{!67}
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !65, type: !22)
// CHECK:STDOUT: !68 = !DILocation(line: 16, column: 1, scope: !65)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbd14a388b4a1c25:core.Destroy.Core", scope: null, file: !1, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !70)
// CHECK:STDOUT: !70 = !{!71}
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !69, type: !22)
// CHECK:STDOUT: !72 = !DILocation(line: 16, column: 1, scope: !69)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "CallC__carbon_thunk", linkageName: "_CCallC__carbon_thunk.Main", scope: null, file: !1, line: 23, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !22)
// CHECK:STDOUT: !76 = !DILocation(line: 23, column: 1, scope: !73)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::C.0" = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc19_3 = internal constant { i32, i32 } zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1FRN6Carbon1CE(ptr noundef nonnull align 4 dereferenceable(8) %c) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !12
// CHECK:STDOUT:   %0 = load ptr, ptr %c.addr, align 8, !tbaa !12, !nonnull !15, !align !16
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %"class.Carbon::C.0", ptr %0, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %a, align 4, !tbaa !17
// CHECK:STDOUT:   %1 = load ptr, ptr %c.addr, align 8, !tbaa !12, !nonnull !15, !align !16
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %"class.Carbon::C.0", ptr %1, i32 0, i32 1
// CHECK:STDOUT:   store i32 34, ptr %b, align 4, !tbaa !19
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.C.Main"(ptr %self) #1 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #2 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d0e8d3e870241846:core.Destroy.Core"(ptr %self) #2 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %self) #2 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #2 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, i32 }, align 4, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !45
// CHECK:STDOUT:   %.loc19_29.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   %.loc19_29.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 1, !dbg !46
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.loc19_3, i64 8, i1 false), !dbg !45
// CHECK:STDOUT:   call void @_Z1FRN6Carbon1CE(ptr %c.var), !dbg !47
// CHECK:STDOUT:   call void @"_COp.cbd14a388b4a1c25:core.Destroy.Core"(ptr %c.var), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !48
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = !{!13, !13, i64 0}
// CHECK:STDOUT: !13 = !{!"p1 _ZTSN6Carbon1CE", !14, i64 0}
// CHECK:STDOUT: !14 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !15 = !{}
// CHECK:STDOUT: !16 = !{i64 4}
// CHECK:STDOUT: !17 = !{!18, !9, i64 0}
// CHECK:STDOUT: !18 = !{!"_ZTSN6Carbon1CE", !9, i64 0, !9, i64 4}
// CHECK:STDOUT: !19 = !{!18, !9, i64 4}
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.C.Main", scope: null, file: !1, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 6, column: 1, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 6, type: !28, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !30}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !27, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 6, column: 1, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d0e8d3e870241846:core.Destroy.Core", scope: null, file: !1, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !23)
// CHECK:STDOUT: !37 = !DILocation(line: 6, column: 1, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cbd14a388b4a1c25:core.Destroy.Core", scope: null, file: !1, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !23)
// CHECK:STDOUT: !41 = !DILocation(line: 6, column: 1, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !1, line: 18, type: !43, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !43 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !44 = !{null}
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 3, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 14, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 20, column: 3, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 1, scope: !42)
// CHECK:STDOUT:
