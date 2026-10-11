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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.44b676589690b86a(ptr %n2.var), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9NeedThunkC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !75
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !75
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !75
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !75
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
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.86629adb6c63d273.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:SubobjectDestroy.6b4f1dd80a288d4e.Core:NeedThunk.Cpp"(ptr %self) #2 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.44b676589690b86a(ptr %t) #0 !dbg !64 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.86629adb6c63d273.Core:DestroyImpl.Core"(ptr %t), !dbg !67
// CHECK:STDOUT:   call void @"_COp:thunk:SubobjectDestroy.6b4f1dd80a288d4e.Core:NeedThunk.Cpp"(ptr %t), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !69
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
// CHECK:STDOUT: !llvm.module.flags = !{!77, !78, !79, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!82}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.86629adb6c63d273.Core:DestroyImpl.Core", scope: null, file: !34, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !36)
// CHECK:STDOUT: !58 = !DILocation(line: 0, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:SubobjectDestroy.6b4f1dd80a288d4e.Core:NeedThunk.Cpp", scope: null, file: !51, line: 5, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !36)
// CHECK:STDOUT: !62 = !DILocation(line: 5, column: 3, scope: !60)
// CHECK:STDOUT: !63 = !{!61}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.44b676589690b86a", scope: null, file: !28, line: 26, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !70)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !36)
// CHECK:STDOUT: !67 = !DILocation(line: 27, column: 5, scope: !64)
// CHECK:STDOUT: !68 = !DILocation(line: 28, column: 5, scope: !64)
// CHECK:STDOUT: !69 = !DILocation(line: 26, column: 1, scope: !64)
// CHECK:STDOUT: !70 = !{!65}
// CHECK:STDOUT: !71 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !72 = !{!"omnipotent char", !71, i64 0}
// CHECK:STDOUT: !73 = !{!"any pointer", !72, i64 0}
// CHECK:STDOUT: !74 = !{!"p1 _ZTS9NeedThunk", !73, i64 0}
// CHECK:STDOUT: !75 = !{!74, !74, i64 0}
// CHECK:STDOUT: !77 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !78 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !79 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !80 = !{!"int", !72, i64 0}
// CHECK:STDOUT: !81 = !{!"__libc_errno", !80, i64 0}
// CHECK:STDOUT: !82 = !{!81, !80, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunks_for_distinct_specializations.carbon'
// CHECK:STDOUT: source_filename = "thunks_for_distinct_specializations.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %a, ptr %b) #0 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a2.var = alloca [1 x i8], align 1, !dbg !90
// CHECK:STDOUT:   %.loc12_35.1.temp = alloca [1 x i8], align 1, !dbg !91
// CHECK:STDOUT:   %b2.var = alloca [1 x i8], align 1, !dbg !93
// CHECK:STDOUT:   %.loc13_35.1.temp = alloca [1 x i8], align 1, !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a2.var), !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_35.1.temp), !dbg !91
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr %a, ptr %a2.var), !dbg !91
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_35.1.temp), !dbg !94
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr %b, ptr %b2.var), !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9c1f17b7b970807d(ptr %b2.var), !dbg !93
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.81b1ae2a3a919430(ptr %a2.var), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !130
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !130
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !130
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !130
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.c0735c73762b4d1c.Core:Box.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr %self, ptr %return), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !134
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !134
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !134
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !134
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.2c1dd62357f2b914.Core:Box.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr %self, ptr %return), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.7d6eb391bc2cf3d7.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN3BoxI1BED1Ev(ptr noundef nonnull align 1 dead_on_return(1) dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.d653779b9f221b29.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN3BoxI1AED1Ev(ptr noundef nonnull align 1 dead_on_return(1) dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9c1f17b7b970807d(ptr %t) #0 !dbg !115 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.7d6eb391bc2cf3d7.Core:DestroyImpl.Core"(ptr %t), !dbg !118
// CHECK:STDOUT:   call void @_ZN3BoxI1BED1Ev(ptr %t), !dbg !119
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.81b1ae2a3a919430(ptr %t) #0 !dbg !122 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.d653779b9f221b29.Core:DestroyImpl.Core"(ptr %t), !dbg !125
// CHECK:STDOUT:   call void @_ZN3BoxI1AED1Ev(ptr %t), !dbg !126
// CHECK:STDOUT:   ret void, !dbg !127
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!84}
// CHECK:STDOUT: !llvm.module.flags = !{!77, !78, !79, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!82}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{null, !36, !36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !{!36, !36}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !71 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !72 = !{!"omnipotent char", !71, i64 0}
// CHECK:STDOUT: !73 = !{!"any pointer", !72, i64 0}
// CHECK:STDOUT: !77 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !78 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !79 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !80 = !{!"int", !72, i64 0}
// CHECK:STDOUT: !81 = !{!"__libc_errno", !80, i64 0}
// CHECK:STDOUT: !82 = !{!81, !80, i64 0}
// CHECK:STDOUT: !83 = !DIFile(filename: "thunks_for_distinct_specializations.carbon", directory: "")
// CHECK:STDOUT: !84 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !83, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !83, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !97)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !36)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 2, scope: !85, type: !36)
// CHECK:STDOUT: !90 = !DILocation(line: 12, column: 3, scope: !85)
// CHECK:STDOUT: !91 = !DILocation(line: 12, column: 35, scope: !85)
// CHECK:STDOUT: !93 = !DILocation(line: 13, column: 3, scope: !85)
// CHECK:STDOUT: !94 = !DILocation(line: 13, column: 35, scope: !85)
// CHECK:STDOUT: !96 = !DILocation(line: 11, column: 1, scope: !85)
// CHECK:STDOUT: !97 = !{!86, !87}
// CHECK:STDOUT: !98 = !DIFile(filename: "./box.h", directory: "")
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.c0735c73762b4d1c.Core:Box.Cpp", scope: null, file: !98, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !36)
// CHECK:STDOUT: !101 = !DILocation(line: 8, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.2c1dd62357f2b914.Core:Box.Cpp", scope: null, file: !98, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !36)
// CHECK:STDOUT: !105 = !DILocation(line: 8, column: 3, scope: !103)
// CHECK:STDOUT: !106 = !{!104}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.7d6eb391bc2cf3d7.Core:DestroyImpl.Core", scope: null, file: !83, type: !38, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !36)
// CHECK:STDOUT: !109 = !DILocation(line: 0, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.d653779b9f221b29.Core:DestroyImpl.Core", scope: null, file: !83, type: !38, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !36)
// CHECK:STDOUT: !113 = !DILocation(line: 0, scope: !111)
// CHECK:STDOUT: !114 = !{!112}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9c1f17b7b970807d", scope: null, file: !28, line: 26, type: !38, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !121)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !36)
// CHECK:STDOUT: !118 = !DILocation(line: 27, column: 5, scope: !115)
// CHECK:STDOUT: !119 = !DILocation(line: 28, column: 5, scope: !115)
// CHECK:STDOUT: !120 = !DILocation(line: 26, column: 1, scope: !115)
// CHECK:STDOUT: !121 = !{!116}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.81b1ae2a3a919430", scope: null, file: !28, line: 26, type: !38, spFlags: DISPFlagDefinition, unit: !84, retainedNodes: !128)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !36)
// CHECK:STDOUT: !125 = !DILocation(line: 27, column: 5, scope: !122)
// CHECK:STDOUT: !126 = !DILocation(line: 28, column: 5, scope: !122)
// CHECK:STDOUT: !127 = !DILocation(line: 26, column: 1, scope: !122)
// CHECK:STDOUT: !128 = !{!123}
// CHECK:STDOUT: !129 = !{!"p1 _ZTS3BoxI1AE", !73, i64 0}
// CHECK:STDOUT: !130 = !{!129, !129, i64 0}
// CHECK:STDOUT: !133 = !{!"p1 _ZTS3BoxI1BE", !73, i64 0}
// CHECK:STDOUT: !134 = !{!133, !133, i64 0}
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
// CHECK:STDOUT: define void @_CCall.Main(ptr %a, ptr %b) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a2.var = alloca [1 x i8], align 1, !dbg !144
// CHECK:STDOUT:   %.loc12_32.1.temp = alloca [1 x i8], align 1, !dbg !145
// CHECK:STDOUT:   %b2.var = alloca [1 x i8], align 1, !dbg !147
// CHECK:STDOUT:   %.loc13_36.1.temp = alloca [1 x i8], align 1, !dbg !148
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a2.var), !dbg !144
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_32.1.temp), !dbg !145
// CHECK:STDOUT:   call void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr %a, ptr %a2.var), !dbg !145
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !147
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_36.1.temp), !dbg !148
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr %b, ptr %b2.var), !dbg !148
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.59ab7877a93f58b8(ptr %b2.var), !dbg !147
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bcd607a73108c5ec(ptr %a2.var), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !184
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !184
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !184
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !184
// CHECK:STDOUT:   call void @_ZN7CollideC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.9e522bb4dbf82e5c.Core:Collide.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !187
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !187
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !187
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !187
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC2ERKS0_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.d1fe27f3241437fa.Core:Collide..Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr %self, ptr %return), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.ca1c08705916a9b7.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !187
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.80ba2e027e5b3e23.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7CollideD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !184
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.59ab7877a93f58b8(ptr %t) #0 !dbg !169 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.ca1c08705916a9b7.Core:DestroyImpl.Core"(ptr %t), !dbg !172
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideD2Ev(ptr %t), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bcd607a73108c5ec(ptr %t) #0 !dbg !176 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.80ba2e027e5b3e23.Core:DestroyImpl.Core"(ptr %t), !dbg !179
// CHECK:STDOUT:   call void @_ZN7CollideD2Ev(ptr %t), !dbg !180
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7CollideC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !184
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !184
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideC2ERKS0_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !187
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !187
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!138}
// CHECK:STDOUT: !llvm.module.flags = !{!77, !78, !79, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!82}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{null, !36, !36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !{!36, !36}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !71 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !72 = !{!"omnipotent char", !71, i64 0}
// CHECK:STDOUT: !73 = !{!"any pointer", !72, i64 0}
// CHECK:STDOUT: !77 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !78 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !79 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !80 = !{!"int", !72, i64 0}
// CHECK:STDOUT: !81 = !{!"__libc_errno", !80, i64 0}
// CHECK:STDOUT: !82 = !{!81, !80, i64 0}
// CHECK:STDOUT: !137 = !DIFile(filename: "thunks_for_anonymous_namespace_collision.carbon", directory: "")
// CHECK:STDOUT: !138 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !137, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !137, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !151)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !36)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 2, scope: !139, type: !36)
// CHECK:STDOUT: !144 = !DILocation(line: 12, column: 3, scope: !139)
// CHECK:STDOUT: !145 = !DILocation(line: 12, column: 32, scope: !139)
// CHECK:STDOUT: !147 = !DILocation(line: 13, column: 3, scope: !139)
// CHECK:STDOUT: !148 = !DILocation(line: 13, column: 36, scope: !139)
// CHECK:STDOUT: !150 = !DILocation(line: 11, column: 1, scope: !139)
// CHECK:STDOUT: !151 = !{!140, !141}
// CHECK:STDOUT: !152 = !DIFile(filename: "./anonymous_namespace_collision.h", directory: "")
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.9e522bb4dbf82e5c.Core:Collide.Cpp", scope: null, file: !152, line: 17, type: !42, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !156)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !36)
// CHECK:STDOUT: !155 = !DILocation(line: 17, column: 3, scope: !153)
// CHECK:STDOUT: !156 = !{!154}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.d1fe27f3241437fa.Core:Collide..Cpp", scope: null, file: !152, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !160)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !36)
// CHECK:STDOUT: !159 = !DILocation(line: 8, column: 3, scope: !157)
// CHECK:STDOUT: !160 = !{!158}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.ca1c08705916a9b7.Core:DestroyImpl.Core", scope: null, file: !137, type: !38, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !164)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !36)
// CHECK:STDOUT: !163 = !DILocation(line: 0, scope: !161)
// CHECK:STDOUT: !164 = !{!162}
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.80ba2e027e5b3e23.Core:DestroyImpl.Core", scope: null, file: !137, type: !38, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !168)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 1, scope: !165, type: !36)
// CHECK:STDOUT: !167 = !DILocation(line: 0, scope: !165)
// CHECK:STDOUT: !168 = !{!166}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.59ab7877a93f58b8", scope: null, file: !28, line: 26, type: !38, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !175)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !36)
// CHECK:STDOUT: !172 = !DILocation(line: 27, column: 5, scope: !169)
// CHECK:STDOUT: !173 = !DILocation(line: 28, column: 5, scope: !169)
// CHECK:STDOUT: !174 = !DILocation(line: 26, column: 1, scope: !169)
// CHECK:STDOUT: !175 = !{!170}
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bcd607a73108c5ec", scope: null, file: !28, line: 26, type: !38, spFlags: DISPFlagDefinition, unit: !138, retainedNodes: !182)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 1, scope: !176, type: !36)
// CHECK:STDOUT: !179 = !DILocation(line: 27, column: 5, scope: !176)
// CHECK:STDOUT: !180 = !DILocation(line: 28, column: 5, scope: !176)
// CHECK:STDOUT: !181 = !DILocation(line: 26, column: 1, scope: !176)
// CHECK:STDOUT: !182 = !{!177}
// CHECK:STDOUT: !183 = !{!"p1 _ZTS7Collide", !73, i64 0}
// CHECK:STDOUT: !184 = !{!183, !183, i64 0}
// CHECK:STDOUT: !186 = !{!"p1 _ZTSN12_GLOBAL__N_17CollideE", !73, i64 0}
// CHECK:STDOUT: !187 = !{!186, !186, i64 0}
// CHECK:STDOUT:
