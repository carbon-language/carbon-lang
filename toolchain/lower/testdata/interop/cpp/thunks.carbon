// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --clang-arg=-std=c++26
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/thunks.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/thunks.carbon

// --- need_thunk.h

struct NeedThunk {
  NeedThunk();
  NeedThunk(const NeedThunk&);
  ~NeedThunk() = default;
};

// --- multiple_thunks_in_one_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "need_thunk.h";

// We intend to create multiple thunks named `Op` here to demonstrate the need
// for the mangling to include the signature function.
fn Call(n: Cpp.NeedThunk) {
  var unused n2: Cpp.NeedThunk = n;
}

// --- box.h

struct A {};
struct B {};

template <typename T>
struct Box {
  Box();
  Box(const Box&);
  ~Box();
  T value;
};

// --- thunks_for_distinct_specializations.carbon

library "[[@TEST_NAME]]";

import Cpp library "box.h";

// Distinct specializations of the same class template share a Carbon name and
// parent scope, so their `Copy` (and `Destroy`) thunks need distinct mangled
// names; otherwise they would collide into a single LLVM function. The
// distinguishing key is the Clang mangled type name recorded when importing a
// C++ class.
fn Call(a: Cpp.Box(Cpp.A), b: Cpp.Box(Cpp.B)) {
  var unused a2: Cpp.Box(Cpp.A) = a;
  var unused b2: Cpp.Box(Cpp.B) = b;
}

// --- anonymous_namespace_collision.h

namespace {
// A type in the anonymous namespace. We expose it under an alias because, once
// the global `Collide` below is declared, the unqualified name `Collide`
// refers to that global type.
struct Collide {
  Collide() {}
  Collide(const Collide&) {}
  ~Collide() {}
};
using AnonCollide = Collide;
}  // namespace

// A *different* type with the same name in the global namespace.
struct Collide {
  Collide() {}
  Collide(const Collide&) {}
  ~Collide() {}
};

// --- thunks_for_anonymous_namespace_collision.carbon

library "[[@TEST_NAME]]";

import Cpp library "anonymous_namespace_collision.h";

// `Cpp.Collide` is the global type and `Cpp.AnonCollide` aliases a distinct
// type of the same name in the anonymous namespace. Carbon sees both as
// `Collide` with the same parent scope, so their `Copy` thunks would collide
// into a single LLVM function without the Clang mangled type name (which
// encodes the anonymous namespace) in the fingerprint.
fn Call(a: Cpp.Collide, b: Cpp.AnonCollide) {
  var unused a2: Cpp.Collide = a;
  var unused b2: Cpp.AnonCollide = b;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'multiple_thunks_in_one_class.carbon'
// CHECK:STDOUT: source_filename = "multiple_thunks_in_one_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %n) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n2.var = alloca [1 x i8], align 1, !dbg !46
// CHECK:STDOUT:   %.loc9_34.1.temp = alloca [1 x i8], align 1, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n2.var), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_34.1.temp), !dbg !47
// CHECK:STDOUT:   call void @_ZN9NeedThunkC1ERKS_.carbon_thunk._(ptr %n, ptr %n2.var), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a232ab5552c0c8a5(ptr %n2.var), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9NeedThunkC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !71
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !71
// CHECK:STDOUT:   call void @_ZN9NeedThunkC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.60e3c6e2415a2ad6.Core:NeedThunk.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9NeedThunkC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:SubobjectDestroy.6b4f1dd80a288d4e.Core:NeedThunk.Cpp"(ptr %self) #2 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a232ab5552c0c8a5(ptr %t) #0 !dbg !60 {
// CHECK:STDOUT:   call void @"_COp:thunk:SubobjectDestroy.6b4f1dd80a288d4e.Core:NeedThunk.Cpp"(ptr %t), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN9NeedThunkC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!73, !74, !75, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!78}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "multiple_thunks_in_one_class.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !41 = !{!36, !36}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !34, line: 8, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !50)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !36)
// CHECK:STDOUT: !46 = !DILocation(line: 9, column: 3, scope: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 9, column: 34, scope: !43)
// CHECK:STDOUT: !49 = !DILocation(line: 8, column: 1, scope: !43)
// CHECK:STDOUT: !50 = !{!44}
// CHECK:STDOUT: !51 = !DIFile(filename: "./need_thunk.h", directory: "")
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.60e3c6e2415a2ad6.Core:NeedThunk.Cpp", scope: null, file: !51, line: 4, type: !42, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !36)
// CHECK:STDOUT: !54 = !DILocation(line: 4, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:SubobjectDestroy.6b4f1dd80a288d4e.Core:NeedThunk.Cpp", scope: null, file: !51, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !36)
// CHECK:STDOUT: !58 = !DILocation(line: 5, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a232ab5552c0c8a5", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !66)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !36)
// CHECK:STDOUT: !64 = !DILocation(line: 34, column: 5, scope: !60)
// CHECK:STDOUT: !65 = !DILocation(line: 32, column: 1, scope: !60)
// CHECK:STDOUT: !66 = !{!61}
// CHECK:STDOUT: !67 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !68 = !{!"omnipotent char", !67, i64 0}
// CHECK:STDOUT: !69 = !{!"any pointer", !68, i64 0}
// CHECK:STDOUT: !70 = !{!"p1 _ZTS9NeedThunk", !69, i64 0}
// CHECK:STDOUT: !71 = !{!70, !70, i64 0}
// CHECK:STDOUT: !73 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !74 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !75 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !76 = !{!"int", !68, i64 0}
// CHECK:STDOUT: !77 = !{!"__libc_errno", !76, i64 0}
// CHECK:STDOUT: !78 = !{!77, !76, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunks_for_distinct_specializations.carbon'
// CHECK:STDOUT: source_filename = "thunks_for_distinct_specializations.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %a, ptr %b) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a2.var = alloca [1 x i8], align 1, !dbg !86
// CHECK:STDOUT:   %.loc12_35.1.temp = alloca [1 x i8], align 1, !dbg !87
// CHECK:STDOUT:   %b2.var = alloca [1 x i8], align 1, !dbg !89
// CHECK:STDOUT:   %.loc13_35.1.temp = alloca [1 x i8], align 1, !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a2.var), !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_35.1.temp), !dbg !87
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr %a, ptr %a2.var), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_35.1.temp), !dbg !90
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr %b, ptr %b2.var), !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d1e44ef097900801(ptr %b2.var), !dbg !89
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5fbb58cefed751bc(ptr %a2.var), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !118
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !118
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.c0735c73762b4d1c.Core:Box.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr %self, ptr %return), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !122
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !122
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !122
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !122
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.2c1dd62357f2b914.Core:Box.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr %self, ptr %return), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN3BoxI1BED1Ev(ptr noundef nonnull align 1 dead_on_return(1) dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN3BoxI1AED1Ev(ptr noundef nonnull align 1 dead_on_return(1) dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d1e44ef097900801(ptr %t) #0 !dbg !103 {
// CHECK:STDOUT:   call void @_ZN3BoxI1BED1Ev(ptr %t), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5fbb58cefed751bc(ptr %t) #0 !dbg !110 {
// CHECK:STDOUT:   call void @_ZN3BoxI1AED1Ev(ptr %t), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN3BoxI1AEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #5
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN3BoxI1BEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!80}
// CHECK:STDOUT: !llvm.module.flags = !{!73, !74, !75, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!78}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{null, !36, !36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !{!36, !36}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !67 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !68 = !{!"omnipotent char", !67, i64 0}
// CHECK:STDOUT: !69 = !{!"any pointer", !68, i64 0}
// CHECK:STDOUT: !73 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !74 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !75 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !76 = !{!"int", !68, i64 0}
// CHECK:STDOUT: !77 = !{!"__libc_errno", !76, i64 0}
// CHECK:STDOUT: !78 = !{!77, !76, i64 0}
// CHECK:STDOUT: !79 = !DIFile(filename: "thunks_for_distinct_specializations.carbon", directory: "")
// CHECK:STDOUT: !80 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !79, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !79, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !93)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !36)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 2, scope: !81, type: !36)
// CHECK:STDOUT: !86 = !DILocation(line: 12, column: 3, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 12, column: 35, scope: !81)
// CHECK:STDOUT: !89 = !DILocation(line: 13, column: 3, scope: !81)
// CHECK:STDOUT: !90 = !DILocation(line: 13, column: 35, scope: !81)
// CHECK:STDOUT: !92 = !DILocation(line: 11, column: 1, scope: !81)
// CHECK:STDOUT: !93 = !{!82, !83}
// CHECK:STDOUT: !94 = !DIFile(filename: "./box.h", directory: "")
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.c0735c73762b4d1c.Core:Box.Cpp", scope: null, file: !94, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !36)
// CHECK:STDOUT: !97 = !DILocation(line: 8, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.2c1dd62357f2b914.Core:Box.Cpp", scope: null, file: !94, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !36)
// CHECK:STDOUT: !101 = !DILocation(line: 8, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d1e44ef097900801", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !109)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !36)
// CHECK:STDOUT: !107 = !DILocation(line: 34, column: 5, scope: !103)
// CHECK:STDOUT: !108 = !DILocation(line: 32, column: 1, scope: !103)
// CHECK:STDOUT: !109 = !{!104}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5fbb58cefed751bc", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !80, retainedNodes: !116)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !36)
// CHECK:STDOUT: !114 = !DILocation(line: 34, column: 5, scope: !110)
// CHECK:STDOUT: !115 = !DILocation(line: 32, column: 1, scope: !110)
// CHECK:STDOUT: !116 = !{!111}
// CHECK:STDOUT: !117 = !{!"p1 _ZTS3BoxI1AE", !69, i64 0}
// CHECK:STDOUT: !118 = !{!117, !117, i64 0}
// CHECK:STDOUT: !121 = !{!"p1 _ZTS3BoxI1BE", !69, i64 0}
// CHECK:STDOUT: !122 = !{!121, !121, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunks_for_anonymous_namespace_collision.carbon'
// CHECK:STDOUT: source_filename = "thunks_for_anonymous_namespace_collision.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN7CollideD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN7CollideC2ERKS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %a, ptr %b) #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a2.var = alloca [1 x i8], align 1, !dbg !132
// CHECK:STDOUT:   %.loc12_32.1.temp = alloca [1 x i8], align 1, !dbg !133
// CHECK:STDOUT:   %b2.var = alloca [1 x i8], align 1, !dbg !135
// CHECK:STDOUT:   %.loc13_36.1.temp = alloca [1 x i8], align 1, !dbg !136
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a2.var), !dbg !132
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_32.1.temp), !dbg !133
// CHECK:STDOUT:   call void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr %a, ptr %a2.var), !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_36.1.temp), !dbg !136
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr %b, ptr %b2.var), !dbg !136
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c89f00da2163eb95(ptr %b2.var), !dbg !135
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0b81af486af557dc(ptr %a2.var), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !164
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !164
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !164
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !164
// CHECK:STDOUT:   call void @_ZN7CollideC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.9e522bb4dbf82e5c.Core:Collide.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !167
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !167
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !167
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !167
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC2ERKS0_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.d1fe27f3241437fa.Core:Collide..Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr %self, ptr %return), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !167
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7CollideD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !164
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c89f00da2163eb95(ptr %t) #0 !dbg !149 {
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideD2Ev(ptr %t), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0b81af486af557dc(ptr %t) #0 !dbg !156 {
// CHECK:STDOUT:   call void @_ZN7CollideD2Ev(ptr %t), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7CollideC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !164
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !164
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideC2ERKS0_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !167
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !167
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!126}
// CHECK:STDOUT: !llvm.module.flags = !{!73, !74, !75, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!78}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{null, !36, !36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !{!36, !36}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !67 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !68 = !{!"omnipotent char", !67, i64 0}
// CHECK:STDOUT: !69 = !{!"any pointer", !68, i64 0}
// CHECK:STDOUT: !73 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !74 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !75 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !76 = !{!"int", !68, i64 0}
// CHECK:STDOUT: !77 = !{!"__libc_errno", !76, i64 0}
// CHECK:STDOUT: !78 = !{!77, !76, i64 0}
// CHECK:STDOUT: !125 = !DIFile(filename: "thunks_for_anonymous_namespace_collision.carbon", directory: "")
// CHECK:STDOUT: !126 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !125, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !125, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !139)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !36)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 2, scope: !127, type: !36)
// CHECK:STDOUT: !132 = !DILocation(line: 12, column: 3, scope: !127)
// CHECK:STDOUT: !133 = !DILocation(line: 12, column: 32, scope: !127)
// CHECK:STDOUT: !135 = !DILocation(line: 13, column: 3, scope: !127)
// CHECK:STDOUT: !136 = !DILocation(line: 13, column: 36, scope: !127)
// CHECK:STDOUT: !138 = !DILocation(line: 11, column: 1, scope: !127)
// CHECK:STDOUT: !139 = !{!128, !129}
// CHECK:STDOUT: !140 = !DIFile(filename: "./anonymous_namespace_collision.h", directory: "")
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.9e522bb4dbf82e5c.Core:Collide.Cpp", scope: null, file: !140, line: 17, type: !42, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !144)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !36)
// CHECK:STDOUT: !143 = !DILocation(line: 17, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !{!142}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.d1fe27f3241437fa.Core:Collide..Cpp", scope: null, file: !140, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !148)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !36)
// CHECK:STDOUT: !147 = !DILocation(line: 8, column: 3, scope: !145)
// CHECK:STDOUT: !148 = !{!146}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c89f00da2163eb95", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !155)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !36)
// CHECK:STDOUT: !153 = !DILocation(line: 34, column: 5, scope: !149)
// CHECK:STDOUT: !154 = !DILocation(line: 32, column: 1, scope: !149)
// CHECK:STDOUT: !155 = !{!150}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0b81af486af557dc", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !162)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !36)
// CHECK:STDOUT: !160 = !DILocation(line: 34, column: 5, scope: !156)
// CHECK:STDOUT: !161 = !DILocation(line: 32, column: 1, scope: !156)
// CHECK:STDOUT: !162 = !{!157}
// CHECK:STDOUT: !163 = !{!"p1 _ZTS7Collide", !69, i64 0}
// CHECK:STDOUT: !164 = !{!163, !163, i64 0}
// CHECK:STDOUT: !166 = !{!"p1 _ZTSN12_GLOBAL__N_17CollideE", !69, i64 0}
// CHECK:STDOUT: !167 = !{!166, !166, i64 0}
// CHECK:STDOUT:
