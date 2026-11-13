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
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
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
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !50 = !{!49, !40, i64 0}
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
// CHECK:STDOUT: define i32 @_CUseVal.Main(ptr %a) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.by_ref.call = call i32 @_ZN1A6by_refEv(ptr %a), !dbg !58
// CHECK:STDOUT:   ret i32 %A.by_ref.call, !dbg !59
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!52}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
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
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !50 = !{!49, !40, i64 0}
// CHECK:STDOUT: !51 = !DIFile(filename: "call_by_ref.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !51, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !51, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !60)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !21)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 10, scope: !53)
// CHECK:STDOUT: !59 = !DILocation(line: 7, column: 3, scope: !53)
// CHECK:STDOUT: !60 = !{!54}
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
// CHECK:STDOUT: define void @_CUseVal.Main(ptr %a) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_4.2.base = getelementptr inbounds nuw [16 x i8], ptr %a, i32 0, i32 0, !dbg !66
// CHECK:STDOUT:   %Base.virt0.call.vtable = load ptr, ptr %.loc7_4.2.base, align 8, !dbg !66
// CHECK:STDOUT:   %Base.virt0.call = getelementptr ptr, ptr %Base.virt0.call.vtable, i32 0, !dbg !66
// CHECK:STDOUT:   %Base.virt0.call.memptr.virtualfn = load ptr, ptr %Base.virt0.call, align 8, !dbg !66
// CHECK:STDOUT:   call void %Base.virt0.call.memptr.virtualfn(ptr %.loc7_4.2.base), !dbg !66
// CHECK:STDOUT:   %A.virt1.call.vtable = load ptr, ptr %a, align 8, !dbg !67
// CHECK:STDOUT:   %A.virt1.call = getelementptr ptr, ptr %A.virt1.call.vtable, i32 1, !dbg !67
// CHECK:STDOUT:   %A.virt1.call.memptr.virtualfn = load ptr, ptr %A.virt1.call, align 8, !dbg !67
// CHECK:STDOUT:   call void %A.virt1.call.memptr.virtualfn(ptr %a), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!62}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !40 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !50 = !{!49, !40, i64 0}
// CHECK:STDOUT: !61 = !DIFile(filename: "call_virtual.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !61, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "UseVal", linkageName: "_CUseVal.Main", scope: null, file: !61, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !69)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 7, column: 3, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 8, column: 3, scope: !63)
// CHECK:STDOUT: !68 = !DILocation(line: 6, column: 1, scope: !63)
// CHECK:STDOUT: !69 = !{!64}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.NeedThunk = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @int_1.2eb = internal constant i8 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %n) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_14.3.temp = alloca i8, align 1, !dbg !79
// CHECK:STDOUT:   %.loc8_14.3.temp = alloca i8, align 1, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_14.3.temp), !dbg !79
// CHECK:STDOUT:   call void @_ZNK9NeedThunk8ImplicitEa.carbon_thunk.__(ptr %n, ptr @int_1.2eb), !dbg !78
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6fb12317fdf5c9b3(ptr @int_1.2eb), !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_14.3.temp), !dbg !83
// CHECK:STDOUT:   call void @_ZNH9NeedThunk8ExplicitES_a.carbon_thunk.__(ptr %n, ptr @int_1.2eb), !dbg !82
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6fb12317fdf5c9b3(ptr @int_1.2eb), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK9NeedThunk8ImplicitEa.carbon_thunk.__(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %c) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !94
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !96
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !94, !nonnull !42
// CHECK:STDOUT:   %1 = load ptr, ptr %c.addr, align 8, !tbaa !96
// CHECK:STDOUT:   %2 = load i8, ptr %1, align 1, !tbaa !98
// CHECK:STDOUT:   call void @_ZNK9NeedThunk8ImplicitEa(ptr noundef nonnull align 1 dereferenceable(1) %0, i8 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNH9NeedThunk8ExplicitES_a.carbon_thunk.__(ptr noundef %0, ptr noundef %c) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %c.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.NeedThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !94
// CHECK:STDOUT:   store ptr %c, ptr %c.addr, align 8, !tbaa !96
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !94
// CHECK:STDOUT:   %2 = load ptr, ptr %c.addr, align 8, !tbaa !96
// CHECK:STDOUT:   %3 = load i8, ptr %2, align 1, !tbaa !98
// CHECK:STDOUT:   call void @_ZNH9NeedThunk8ExplicitES_a(i8 noundef signext %3)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6fb12317fdf5c9b3(ptr %t) #0 !dbg !86 {
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK9NeedThunk8ImplicitEa(ptr noundef nonnull align 1 dereferenceable(1), i8 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNH9NeedThunk8ExplicitES_a(i8 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6fb12317fdf5c9b3, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!71}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !36 = !{!"any pointer", !35, i64 0}
// CHECK:STDOUT: !40 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !42 = !{}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !50 = !{!49, !40, i64 0}
// CHECK:STDOUT: !70 = !DIFile(filename: "call_thunk.carbon", directory: "")
// CHECK:STDOUT: !71 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !70, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !72 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !70, line: 6, type: !25, spFlags: DISPFlagDefinition, unit: !71, retainedNodes: !85)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !21)
// CHECK:STDOUT: !78 = !DILocation(line: 7, column: 3, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 7, column: 14, scope: !75)
// CHECK:STDOUT: !80 = !{null, !72}
// CHECK:STDOUT: !81 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !82 = !DILocation(line: 8, column: 3, scope: !75)
// CHECK:STDOUT: !83 = !DILocation(line: 8, column: 14, scope: !75)
// CHECK:STDOUT: !84 = !DILocation(line: 6, column: 1, scope: !75)
// CHECK:STDOUT: !85 = !{!76}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6fb12317fdf5c9b3", scope: null, file: !10, line: 32, type: !81, spFlags: DISPFlagDefinition, unit: !71, retainedNodes: !92)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !72)
// CHECK:STDOUT: !91 = !DILocation(line: 32, column: 1, scope: !86)
// CHECK:STDOUT: !92 = !{!87}
// CHECK:STDOUT: !93 = !{!"p1 _ZTS9NeedThunk", !36, i64 0}
// CHECK:STDOUT: !94 = !{!93, !93, i64 0}
// CHECK:STDOUT: !95 = !{!"p1 omnipotent char", !36, i64 0}
// CHECK:STDOUT: !96 = !{!95, !95, i64 0}
// CHECK:STDOUT: !98 = !{!35, !35, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_with_ref_self.carbon'
// CHECK:STDOUT: source_filename = "call_with_ref_self.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main(ptr %r) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN8RefSelfA1fEii(ptr %r, i32 1, i32 2), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN8RefSelfA1fEii(ptr noundef nonnull align 1 dereferenceable(1), i32 noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!100}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !21}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !40 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !40, i64 0}
// CHECK:STDOUT: !50 = !{!49, !40, i64 0}
// CHECK:STDOUT: !99 = !DIFile(filename: "call_with_ref_self.carbon", directory: "")
// CHECK:STDOUT: !100 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !99, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !99, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !100, retainedNodes: !108)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !21)
// CHECK:STDOUT: !106 = !DILocation(line: 12, column: 3, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 11, column: 1, scope: !103)
// CHECK:STDOUT: !108 = !{!104}
// CHECK:STDOUT:
