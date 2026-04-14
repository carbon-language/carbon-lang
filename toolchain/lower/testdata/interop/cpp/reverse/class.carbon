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
  var x: ()*;
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
// CHECK:STDOUT: %"class.Carbon::A" = type { [12 x i8] }
// CHECK:STDOUT: %"class.Carbon::B" = type { [16 x i8] }
// CHECK:STDOUT: %"class.Carbon::C" = type { [16 x i8] }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a = alloca %"class.Carbon::A", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a) #3
// CHECK:STDOUT:   call void @_ZN6CarbonL16CallA__cpp_thunkEPNS_1AE(ptr noundef %a)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #3
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL16CallA__cpp_thunkEPNS_1AE(ptr noundef %0) #2 {
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
// CHECK:STDOUT:   call void @_ZN6CarbonL16CallB__cpp_thunkEPNS_1BE(ptr noundef %b)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #3
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL16CallB__cpp_thunkEPNS_1BE(ptr noundef %0) #2 {
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
// CHECK:STDOUT:   call void @_ZN6CarbonL16CallC__cpp_thunkEPNS_1CE(ptr noundef %c)
// CHECK:STDOUT:   call void @_ZN6CarbonL16CallA__cpp_thunkEPNS_1AE(ptr noundef %c)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %c) #3
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL16CallC__cpp_thunkEPNS_1CE(ptr noundef %0) #2 {
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
