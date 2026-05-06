// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-std=c++26
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/method.carbon

// --- methods.h

struct Base {
  virtual void virt0();
};

struct A: Base {
  int by_val() const { return n; }
  int by_ref() { return n; }
  virtual void virt1();
  int n;
};

// --- call_by_val.carbon

library "[[@TEST_NAME]]";

import Cpp library "methods.h";

fn UseVal(a: Cpp.A) -> i32 {
  return a.by_val();
}

// --- call_by_ref.carbon

library "[[@TEST_NAME]]";

import Cpp library "methods.h";

fn UseVal(a: Cpp.A*) -> i32 {
  return a->by_ref();
}

// --- call_virtual.carbon

library "[[@TEST_NAME]]";

import Cpp library "methods.h";

fn UseVal(a: Cpp.A*) {
  a->virt0();
  a->virt1();
}

// --- thunk.h

struct NeedThunk {
  void Implicit(signed char c) const;
  void Explicit(this NeedThunk, signed char c);
};

// --- call_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp library "thunk.h";

fn Call(n: Cpp.NeedThunk) {
  n.Implicit(1);
  n.Explicit(1);
}

// CHECK:STDOUT: ; ModuleID = 'call_by_val.carbon'
// CHECK:STDOUT: source_filename = "call_by_val.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A = type <{ %struct.Base, i32, [4 x i8] }>
// CHECK:STDOUT: %struct.Base = type { ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZNK1A6by_valEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUseVal.Main(ptr %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %by_val__carbon_thunk.call = call i32 @_ZNK1A6by_valEv.carbon_thunk(ptr %a), !dbg !18
// CHECK:STDOUT:   ret i32 %by_val__carbon_thunk.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4Base5virt0Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1A5virt1Ev(ptr noundef nonnull align 8 dereferenceable(12)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK1A6by_valEv.carbon_thunk(ptr noundef nonnull align 8 dereferenceable(12) %this) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !20
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !20, !nonnull !23, !align !24
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK1A6by_valEv(ptr noundef nonnull align 8 dereferenceable(12) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef i32 @_ZNK1A6by_valEv(ptr noundef nonnull align 8 dereferenceable(12) %this) #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !20
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %n = getelementptr inbounds nuw %struct.A, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   %0 = load i32, ptr %n, align 8, !tbaa !25
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "call_by_val.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !16)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !15}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !11, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 10, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS1A", !22, i64 0}
// CHECK:STDOUT: !22 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !23 = !{}
// CHECK:STDOUT: !24 = !{i64 8}
// CHECK:STDOUT: !25 = !{!26, !8, i64 8}
// CHECK:STDOUT: !26 = !{!"_ZTS1A", !27, i64 0, !8, i64 8}
// CHECK:STDOUT: !27 = !{!"_ZTS4Base"}
// CHECK:STDOUT: ; ModuleID = 'call_by_ref.carbon'
// CHECK:STDOUT: source_filename = "call_by_ref.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A = type <{ %struct.Base, i32, [4 x i8] }>
// CHECK:STDOUT: %struct.Base = type { ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1A6by_refEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUseVal.Main(ptr %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.by_ref.call = call i32 @_ZN1A6by_refEv(ptr %a), !dbg !18
// CHECK:STDOUT:   ret i32 %A.by_ref.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4Base5virt0Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1A5virt1Ev(ptr noundef nonnull align 8 dereferenceable(12)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef i32 @_ZN1A6by_refEv(ptr noundef nonnull align 8 dereferenceable(12) %this) #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !20
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %n = getelementptr inbounds nuw %struct.A, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   %0 = load i32, ptr %n, align 8, !tbaa !23
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "call_by_ref.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !16)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !15}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !11, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 7, column: 10, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS1A", !22, i64 0}
// CHECK:STDOUT: !22 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !23 = !{!24, !8, i64 8}
// CHECK:STDOUT: !24 = !{!"_ZTS1A", !25, i64 0, !8, i64 8}
// CHECK:STDOUT: !25 = !{!"_ZTS4Base"}
// CHECK:STDOUT: ; ModuleID = 'call_virtual.carbon'
// CHECK:STDOUT: source_filename = "call_virtual.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUseVal.Main(ptr %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.base = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   %Base.virt0.call.vtable = load ptr, ptr %.loc7_4.2.base, align 8, !dbg !17
// CHECK:STDOUT:   %Base.virt0.call = getelementptr ptr, ptr %Base.virt0.call.vtable, i32 0, !dbg !17
// CHECK:STDOUT:   %Base.virt0.call.memptr.virtualfn = load ptr, ptr %Base.virt0.call, align 8, !dbg !17
// CHECK:STDOUT:   call void %Base.virt0.call.memptr.virtualfn(ptr %.loc7_4.2.base), !dbg !17
// CHECK:STDOUT:   %A.virt1.call.vtable = load ptr, ptr %a, align 8, !dbg !18
// CHECK:STDOUT:   %A.virt1.call = getelementptr ptr, ptr %A.virt1.call.vtable, i32 1, !dbg !18
// CHECK:STDOUT:   %A.virt1.call.memptr.virtualfn = load ptr, ptr %A.virt1.call, align 8, !dbg !18
// CHECK:STDOUT:   call void %A.virt1.call.memptr.virtualfn(ptr %a), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4Base5virt0Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1A5virt1Ev(ptr noundef nonnull align 8 dereferenceable(12)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "call_virtual.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 8, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.NeedThunk = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @int_1.30e = internal constant i8 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %n) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_14.3.temp = alloca i8, align 1, !dbg !17
// CHECK:STDOUT:   %.loc8_14.3.temp = alloca i8, align 1, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_14.3.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZNK9NeedThunk8ImplicitEa.carbon_thunk(ptr %n, ptr @int_1.30e), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_14.3.temp), !dbg !18
// CHECK:STDOUT:   call void @_ZNH9NeedThunk8ExplicitES_a.carbon_thunk(ptr %n, ptr @int_1.30e), !dbg !20
// CHECK:STDOUT:   call void @"_COp.43bece552030817c:core.Destroy.Core"(ptr @int_1.30e), !dbg !18
// CHECK:STDOUT:   call void @"_COp.43bece552030817c:core.Destroy.Core"(ptr @int_1.30e), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK9NeedThunk8ImplicitEa.carbon_thunk(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %c) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !22
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !25
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !22, !nonnull !27
// CHECK:STDOUT:   %1 = load ptr, ptr %c.addr, align 8, !tbaa !25
// CHECK:STDOUT:   %2 = load i8, ptr %1, align 1, !tbaa !28
// CHECK:STDOUT:   call void @_ZNK9NeedThunk8ImplicitEa(ptr noundef nonnull align 1 dereferenceable(1) %0, i8 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNH9NeedThunk8ExplicitES_a.carbon_thunk(ptr noundef %0, ptr noundef %c) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.NeedThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !22
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !25
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %2 = load ptr, ptr %c.addr, align 8, !tbaa !25
// CHECK:STDOUT:   %3 = load i8, ptr %2, align 1, !tbaa !28
// CHECK:STDOUT:   call void @_ZNH9NeedThunk8ExplicitES_a(i8 noundef signext %3)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.43bece552030817c:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK9NeedThunk8ImplicitEa(ptr noundef nonnull align 1 dereferenceable(1), i8 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNH9NeedThunk8ExplicitES_a(i8 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.43bece552030817c:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 14, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 8, column: 14, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 8, column: 3, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !22 = !{!23, !23, i64 0}
// CHECK:STDOUT: !23 = !{!"p1 _ZTS9NeedThunk", !24, i64 0}
// CHECK:STDOUT: !24 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !25 = !{!26, !26, i64 0}
// CHECK:STDOUT: !26 = !{!"p1 omnipotent char", !24, i64 0}
// CHECK:STDOUT: !27 = !{}
// CHECK:STDOUT: !28 = !{!9, !9, i64 0}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.43bece552030817c:core.Destroy.Core", scope: null, file: !6, line: 8, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{null, !32}
// CHECK:STDOUT: !32 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !29, type: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 8, column: 14, scope: !29)
