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
// CHECK:STDOUT: define i32 @_CUseVal.Main(ptr %a) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %by_val__carbon_thunk.call = call i32 @_ZNK1A6by_valEv.carbon_thunk(ptr %a), !dbg !10
// CHECK:STDOUT:   ret i32 %by_val__carbon_thunk.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local i32 @_ZNK1A6by_valEv.carbon_thunk(ptr %this) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %call = call i32 @_ZNK1A6by_valEv(ptr nonnull align 8 dereferenceable(12) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local i32 @_ZNK1A6by_valEv(ptr nonnull align 8 dereferenceable(12) %this) #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %n = getelementptr inbounds nuw %struct.A, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   %0 = load i32, ptr %n, align 8
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_by_val.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 10, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 3, scope: !7)
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
// CHECK:STDOUT: define i32 @_CUseVal.Main(ptr %a) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.by_ref.call = call i32 @_ZN1A6by_refEv(ptr %a), !dbg !10
// CHECK:STDOUT:   ret i32 %A.by_ref.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local i32 @_ZN1A6by_refEv(ptr nonnull align 8 dereferenceable(12) %this) #0 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %n = getelementptr inbounds nuw %struct.A, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   %0 = load i32, ptr %n, align 8
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_by_ref.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 10, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'call_virtual.carbon'
// CHECK:STDOUT: source_filename = "call_virtual.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUseVal.Main(ptr %a) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.3.base = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %Base.virt0.call.vtable = load ptr, ptr %.loc7_4.3.base, align 8, !dbg !10
// CHECK:STDOUT:   %Base.virt0.call = getelementptr ptr, ptr %Base.virt0.call.vtable, i32 0, !dbg !10
// CHECK:STDOUT:   %Base.virt0.call.memptr.virtualfn = load ptr, ptr %Base.virt0.call, align 8, !dbg !10
// CHECK:STDOUT:   call void %Base.virt0.call.memptr.virtualfn(ptr %.loc7_4.3.base), !dbg !10
// CHECK:STDOUT:   %A.virt1.call.vtable = load ptr, ptr %a, align 8, !dbg !11
// CHECK:STDOUT:   %A.virt1.call = getelementptr ptr, ptr %A.virt1.call.vtable, i32 1, !dbg !11
// CHECK:STDOUT:   %A.virt1.call.memptr.virtualfn = load ptr, ptr %A.virt1.call, align 8, !dbg !11
// CHECK:STDOUT:   call void %A.virt1.call.memptr.virtualfn(ptr %a), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4Base5virt0Ev(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1A5virt1Ev(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_virtual.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 3, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.NeedThunk = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CCall.Main(ptr %n) !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_14.3.temp = alloca i8, align 1, !dbg !10
// CHECK:STDOUT:   %.loc8_14.3.temp = alloca i8, align 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_14.3.temp), !dbg !10
// CHECK:STDOUT:   store i8 1, ptr %.loc7_14.3.temp, align 1, !dbg !10
// CHECK:STDOUT:   call void @_ZNK9NeedThunk8ImplicitEa.carbon_thunk(ptr %n, ptr %.loc7_14.3.temp), !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_14.3.temp), !dbg !11
// CHECK:STDOUT:   store i8 1, ptr %.loc8_14.3.temp, align 1, !dbg !11
// CHECK:STDOUT:   call void @_ZNH9NeedThunk8ExplicitES_a.carbon_thunk(ptr %n, ptr %.loc8_14.3.temp), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_ZNK9NeedThunk8ImplicitEa.carbon_thunk(ptr %this, ptr %c) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %c.addr, align 8
// CHECK:STDOUT:   %2 = load i8, ptr %1, align 1
// CHECK:STDOUT:   call void @_ZNK9NeedThunk8ImplicitEa(ptr nonnull align 1 dereferenceable(1) %0, i8 signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK9NeedThunk8ImplicitEa(ptr nonnull align 1 dereferenceable(1), i8 signext) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_ZNH9NeedThunk8ExplicitES_a.carbon_thunk(ptr %0, ptr %c) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.NeedThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   %2 = load ptr, ptr %c.addr, align 8
// CHECK:STDOUT:   %3 = load i8, ptr %2, align 1
// CHECK:STDOUT:   call void @_ZNH9NeedThunk8ExplicitES_a(i8 signext %3)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNH9NeedThunk8ExplicitES_a(i8 signext) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 14, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 14, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 8, column: 3, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 6, column: 1, scope: !7)
