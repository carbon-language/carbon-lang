// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-std=c++26
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/method.carbon

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

// --- call_with_ref_self.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct RefSelfA {
  void f(int x, int y);
};
''';

fn F(ref r: Cpp.RefSelfA) {
  r.f(1, 2);
}

// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT: @_ZTV1A = available_externally constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN4Base5virt0Ev, ptr @_ZN1A5virt1Ev] }, align 8
// CHECK:STDOUT: @_ZTV4Base = available_externally constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN4Base5virt0Ev] }, align 8
// CHECK:STDOUT: @_ZTI1A = external constant ptr
// CHECK:STDOUT: @_ZTI4Base = external constant ptr
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUseVal.Main(ptr %a) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %by_val__carbon_thunk.call = call i32 @_ZNK1A6by_valEv.carbon_thunk._(ptr %a), !dbg !31
// CHECK:STDOUT:   ret i32 %by_val__carbon_thunk.call, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4Base5virt0Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1A5virt1Ev(ptr noundef nonnull align 8 dereferenceable(12)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK1A6by_valEv.carbon_thunk._(ptr noundef nonnull align 8 dereferenceable(12) %this) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !38
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !38, !nonnull !42, !align !43
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK1A6by_valEv(ptr noundef nonnull align 8 dereferenceable(12) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef i32 @_ZNK1A6by_valEv(ptr noundef nonnull align 8 dereferenceable(12) %this) #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !38
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %n = getelementptr inbounds nuw %struct.A, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   %0 = load i32, ptr %n, align 8, !tbaa !44
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN4Base5virt0Ev, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "call_by_val.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !18, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !21)
// CHECK:STDOUT: !31 = !DILocation(line: 7, column: 10, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 7, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !36 = !{!"any pointer", !35, i64 0}
// CHECK:STDOUT: !37 = !{!"p1 _ZTS1A", !36, i64 0}
// CHECK:STDOUT: !38 = !{!37, !37, i64 0}
// CHECK:STDOUT: !39 = !{!"_ZTS4Base"}
// CHECK:STDOUT: !40 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !41 = !{!"_ZTS1A", !39, i64 0, !40, i64 8}
// CHECK:STDOUT: !42 = !{}
// CHECK:STDOUT: !43 = !{i64 8}
// CHECK:STDOUT: !44 = !{!41, !40, i64 8}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !49 = !{!48, !40, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_by_ref.carbon'
// CHECK:STDOUT: source_filename = "call_by_ref.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A.18 = type <{ %struct.Base.19, i32, [4 x i8] }>
// CHECK:STDOUT: %struct.Base.19 = type { ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1A6by_refEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV1A = available_externally constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN4Base5virt0Ev, ptr @_ZN1A5virt1Ev] }, align 8
// CHECK:STDOUT: @_ZTV4Base = available_externally constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN4Base5virt0Ev] }, align 8
// CHECK:STDOUT: @_ZTI1A = external constant ptr
// CHECK:STDOUT: @_ZTI4Base = external constant ptr
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUseVal.Main(ptr %a) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.by_ref.call = call i32 @_ZN1A6by_refEv(ptr %a), !dbg !57
// CHECK:STDOUT:   ret i32 %A.by_ref.call, !dbg !58
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !38
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %n = getelementptr inbounds nuw %struct.A.18, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   %0 = load i32, ptr %n, align 8, !tbaa !44
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN4Base5virt0Ev, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!51}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!20, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !36 = !{!"any pointer", !35, i64 0}
// CHECK:STDOUT: !37 = !{!"p1 _ZTS1A", !36, i64 0}
// CHECK:STDOUT: !38 = !{!37, !37, i64 0}
// CHECK:STDOUT: !39 = !{!"_ZTS4Base"}
// CHECK:STDOUT: !40 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !41 = !{!"_ZTS1A", !39, i64 0, !40, i64 8}
// CHECK:STDOUT: !44 = !{!41, !40, i64 8}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !49 = !{!48, !40, i64 0}
// CHECK:STDOUT: !50 = !DIFile(filename: "call_by_ref.carbon", directory: "")
// CHECK:STDOUT: !51 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !50, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !50, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !51, retainedNodes: !59)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !21)
// CHECK:STDOUT: !57 = !DILocation(line: 7, column: 10, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 3, scope: !52)
// CHECK:STDOUT: !59 = !{!53}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_virtual.carbon'
// CHECK:STDOUT: source_filename = "call_virtual.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV1A = available_externally constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN4Base5virt0Ev, ptr @_ZN1A5virt1Ev] }, align 8
// CHECK:STDOUT: @_ZTV4Base = available_externally constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN4Base5virt0Ev] }, align 8
// CHECK:STDOUT: @_ZTI1A = external constant ptr
// CHECK:STDOUT: @_ZTI4Base = external constant ptr
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUseVal.Main(ptr %a) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.base = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 0, !dbg !65
// CHECK:STDOUT:   %Base.virt0.call.vtable = load ptr, ptr %.loc7_4.2.base, align 8, !dbg !65
// CHECK:STDOUT:   %Base.virt0.call = getelementptr ptr, ptr %Base.virt0.call.vtable, i32 0, !dbg !65
// CHECK:STDOUT:   %Base.virt0.call.memptr.virtualfn = load ptr, ptr %Base.virt0.call, align 8, !dbg !65
// CHECK:STDOUT:   call void %Base.virt0.call.memptr.virtualfn(ptr %.loc7_4.2.base), !dbg !65
// CHECK:STDOUT:   %A.virt1.call.vtable = load ptr, ptr %a, align 8, !dbg !66
// CHECK:STDOUT:   %A.virt1.call = getelementptr ptr, ptr %A.virt1.call.vtable, i32 1, !dbg !66
// CHECK:STDOUT:   %A.virt1.call.memptr.virtualfn = load ptr, ptr %A.virt1.call, align 8, !dbg !66
// CHECK:STDOUT:   call void %A.virt1.call.memptr.virtualfn(ptr %a), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4Base5virt0Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1A5virt1Ev(ptr noundef nonnull align 8 dereferenceable(12)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN4Base5virt0Ev, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!61}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !40 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !49 = !{!48, !40, i64 0}
// CHECK:STDOUT: !60 = !DIFile(filename: "call_virtual.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !60, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !60, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !61, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !21)
// CHECK:STDOUT: !65 = !DILocation(line: 7, column: 3, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 8, column: 3, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 6, column: 1, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.NeedThunk = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @int_1.f1e = internal constant i8 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %n) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_14.3.temp = alloca i8, align 1, !dbg !80
// CHECK:STDOUT:   %.loc8_14.3.temp = alloca i8, align 1, !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_14.3.temp), !dbg !80
// CHECK:STDOUT:   call void @_ZNK9NeedThunk8ImplicitEa.carbon_thunk.__(ptr %n, ptr @int_1.f1e), !dbg !79
// CHECK:STDOUT:   call void @"_CSelfDestruct.4954785a8b134801:core.Destroy.Core"(ptr @int_1.f1e), !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_14.3.temp), !dbg !82
// CHECK:STDOUT:   call void @_ZNH9NeedThunk8ExplicitES_a.carbon_thunk.__(ptr %n, ptr @int_1.f1e), !dbg !81
// CHECK:STDOUT:   call void @"_CSelfDestruct.4954785a8b134801:core.Destroy.Core"(ptr @int_1.f1e), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK9NeedThunk8ImplicitEa.carbon_thunk.__(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %c) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !98
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !98, !nonnull !42
// CHECK:STDOUT:   %1 = load ptr, ptr %c.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %2 = load i8, ptr %1, align 1, !tbaa !102
// CHECK:STDOUT:   call void @_ZNK9NeedThunk8ImplicitEa(ptr noundef nonnull align 1 dereferenceable(1) %0, i8 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c4a4fcc00062c7f0:core.Destroy.Core"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core"(ptr %self) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4954785a8b134801:core.Destroy.Core"(ptr %self) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core"(ptr %self), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNH9NeedThunk8ExplicitES_a.carbon_thunk.__(ptr noundef %0, ptr noundef %c) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.NeedThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !98
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !98
// CHECK:STDOUT:   %2 = load ptr, ptr %c.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %3 = load i8, ptr %2, align 1, !tbaa !102
// CHECK:STDOUT:   call void @_ZNH9NeedThunk8ExplicitES_a(i8 noundef signext %3)
// CHECK:STDOUT:   ret void
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
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.4954785a8b134801:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !36 = !{!"any pointer", !35, i64 0}
// CHECK:STDOUT: !40 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !42 = !{}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !49 = !{!48, !40, i64 0}
// CHECK:STDOUT: !69 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !74 = !{null, !71}
// CHECK:STDOUT: !75 = !DISubroutineType(types: !74)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !69, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !84)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !21)
// CHECK:STDOUT: !79 = !DILocation(line: 7, column: 3, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 7, column: 14, scope: !76)
// CHECK:STDOUT: !81 = !DILocation(line: 8, column: 3, scope: !76)
// CHECK:STDOUT: !82 = !DILocation(line: 8, column: 14, scope: !76)
// CHECK:STDOUT: !83 = !DILocation(line: 6, column: 1, scope: !76)
// CHECK:STDOUT: !84 = !{!77}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c4a4fcc00062c7f0:core.Destroy.Core", scope: null, file: !69, line: 7, type: !75, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !71)
// CHECK:STDOUT: !87 = !DILocation(line: 7, column: 14, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4954785a8b134801:core.Destroy.Core", scope: null, file: !69, line: 7, type: !75, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !71)
// CHECK:STDOUT: !91 = !DILocation(line: 7, column: 14, scope: !89)
// CHECK:STDOUT: !92 = !{!90}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4954785a8b134801:core.Destroy.Core", scope: null, file: !69, line: 7, type: !75, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !71)
// CHECK:STDOUT: !95 = !DILocation(line: 7, column: 14, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = !{!"p1 _ZTS9NeedThunk", !36, i64 0}
// CHECK:STDOUT: !98 = !{!97, !97, i64 0}
// CHECK:STDOUT: !99 = !{!"p1 omnipotent char", !36, i64 0}
// CHECK:STDOUT: !100 = !{!99, !99, i64 0}
// CHECK:STDOUT: !102 = !{!35, !35, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_with_ref_self.carbon'
// CHECK:STDOUT: source_filename = "call_with_ref_self.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %r) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN8RefSelfA1fEii(ptr %r, i32 1, i32 2), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN8RefSelfA1fEii(ptr noundef nonnull align 1 dereferenceable(1), i32 noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!104}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !40 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !49 = !{!48, !40, i64 0}
// CHECK:STDOUT: !103 = !DIFile(filename: "call_with_ref_self.carbon", directory: "")
// CHECK:STDOUT: !104 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !103, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !103, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !104, retainedNodes: !112)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !21)
// CHECK:STDOUT: !110 = !DILocation(line: 12, column: 3, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 11, column: 1, scope: !107)
// CHECK:STDOUT: !112 = !{!108}
// CHECK:STDOUT:
