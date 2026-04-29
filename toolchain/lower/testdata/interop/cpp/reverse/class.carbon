// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/reverse/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/reverse/class.carbon

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

// CHECK:STDOUT: ; ModuleID = 'pointer.carbon'
// CHECK:STDOUT: source_filename = "pointer.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef ptr @_Z4passPN6Carbon1AE(ptr noundef %p) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %p, ptr %p.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %p.addr, align 8, !tbaa !11
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
// CHECK:STDOUT: define void @_CDoIt.Main() #1 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4callv(), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "pointer.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTSN6Carbon1AE", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "DoIt", linkageName: "_CDoIt.Main", scope: null, file: !6, line: 15, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DILocation(line: 15, column: 13, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 1, scope: !14)
// CHECK:STDOUT: ; ModuleID = 'layout.carbon'
// CHECK:STDOUT: source_filename = "layout.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::A" = type <{ ptr, i32 }>
// CHECK:STDOUT: %"class.Carbon::B" = type { %"class.Carbon::A", i32 }
// CHECK:STDOUT: %"class.Carbon::C" = type { %"class.Carbon::A", i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a = alloca %"class.Carbon::A", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a) #3
// CHECK:STDOUT:   call void @_ZN6CarbonL5CallAEPNS_1AE(ptr noundef %a)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #3
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_CCallA__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b = alloca %"class.Carbon::B", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b) #3
// CHECK:STDOUT:   call void @_ZN6CarbonL5CallBEPNS_1BE(ptr noundef %b)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #3
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL5CallBEPNS_1BE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !14
// CHECK:STDOUT:   call void @_CCallB__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1hv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c = alloca %"class.Carbon::C", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c) #3
// CHECK:STDOUT:   call void @_ZN6CarbonL5CallCEPNS_1CE(ptr noundef %c)
// CHECK:STDOUT:   call void @_ZN6CarbonL5CallAEPNS_1AE(ptr noundef %c)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %c) #3
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL5CallCEPNS_1CE(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !16
// CHECK:STDOUT:   call void @_CCallC__carbon_thunk.Main(ptr noundef nonnull align 8 dereferenceable(8) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallA.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallB.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CCallC.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallA__carbon_thunk.Main(ptr %_) #3 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21 = load ptr, ptr %_, align 8, !dbg !24
// CHECK:STDOUT:   call void @_CCallA.Main(ptr %.loc21), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallB__carbon_thunk.Main(ptr %_) #3 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22 = load ptr, ptr %_, align 8, !dbg !28
// CHECK:STDOUT:   call void @_CCallB.Main(ptr %.loc22), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallC__carbon_thunk.Main(ptr %_) #3 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %_, align 8, !dbg !32
// CHECK:STDOUT:   call void @_CCallC.Main(ptr %.loc23), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "layout.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTSN6Carbon1AE", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = !{!15, !15, i64 0}
// CHECK:STDOUT: !15 = !{!"p1 _ZTSN6Carbon1BE", !13, i64 0}
// CHECK:STDOUT: !16 = !{!17, !17, i64 0}
// CHECK:STDOUT: !17 = !{!"p1 _ZTSN6Carbon1CE", !13, i64 0}
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "CallA__carbon_thunk", linkageName: "_CCallA__carbon_thunk.Main", scope: null, file: !6, line: 21, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null, !21}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !18, type: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 21, column: 1, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "CallB__carbon_thunk", linkageName: "_CCallB__carbon_thunk.Main", scope: null, file: !6, line: 22, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 22, column: 1, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "CallC__carbon_thunk", linkageName: "_CCallC__carbon_thunk.Main", scope: null, file: !6, line: 23, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !21)
// CHECK:STDOUT: !32 = !DILocation(line: 23, column: 1, scope: !29)
// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::C" = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc19_3 = internal constant { i32, i32 } zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z1FRN6Carbon1CE(ptr noundef nonnull align 4 dereferenceable(8) %c) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %c.addr, align 8, !tbaa !11, !nonnull !14, !align !15
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %"class.Carbon::C", ptr %0, i32 0, i32 0
// CHECK:STDOUT:   store i32 12, ptr %a, align 4, !tbaa !16
// CHECK:STDOUT:   %1 = load ptr, ptr %c.addr, align 8, !tbaa !11, !nonnull !14, !align !15
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %"class.Carbon::C", ptr %1, i32 0, i32 1
// CHECK:STDOUT:   store i32 34, ptr %b, align 4, !tbaa !18
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #1 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { i32, i32 }, align 8, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !22
// CHECK:STDOUT:   %.loc19_29.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 0, !dbg !23
// CHECK:STDOUT:   %.loc19_29.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %c.var, i32 0, i32 1, !dbg !23
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c.var, ptr align 4 @C.val.loc19_3, i64 8, i1 false), !dbg !22
// CHECK:STDOUT:   call void @_Z1FRN6Carbon1CE(ptr %c.var), !dbg !24
// CHECK:STDOUT:   call void @"_COp.71e25083d63c4d6c:core.Destroy.Core"(ptr %c.var), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #1 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fb32c4dfca65e378:core.Destroy.Core"(ptr %self) #1 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.71e25083d63c4d6c:core.Destroy.Core"(ptr %self) #1 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "field.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTSN6Carbon1CE", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = !{}
// CHECK:STDOUT: !15 = !{i64 4}
// CHECK:STDOUT: !16 = !{!17, !8, i64 0}
// CHECK:STDOUT: !17 = !{!"_ZTSN6Carbon1CE", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !18 = !{!17, !8, i64 4}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !6, line: 18, type: !20, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DILocation(line: 19, column: 3, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 14, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 20, column: 3, scope: !19)
// CHECK:STDOUT: !25 = !DILocation(line: 18, column: 1, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !6, line: 19, type: !27, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{null, !29}
// CHECK:STDOUT: !29 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !26, type: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 3, scope: !26)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fb32c4dfca65e378:core.Destroy.Core", scope: null, file: !6, line: 19, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !34 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !35 = !{null, !36}
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !33, type: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 19, column: 3, scope: !33)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.71e25083d63c4d6c:core.Destroy.Core", scope: null, file: !6, line: 19, type: !34, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 19, column: 3, scope: !40)
