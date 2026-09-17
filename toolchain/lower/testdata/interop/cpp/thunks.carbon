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
// CHECK:STDOUT: define void @_CCall.Main(ptr %n) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n2.var = alloca [1 x i8], align 1, !dbg !44
// CHECK:STDOUT:   %.loc9_34.1.temp = alloca [1 x i8], align 1, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n2.var), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_34.1.temp), !dbg !45
// CHECK:STDOUT:   call void @_ZN9NeedThunkC1ERKS_.carbon_thunk._(ptr %n, ptr %n2.var), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0f0801deab98555e(ptr %n2.var), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9NeedThunkC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !69
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !69
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !69
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !69
// CHECK:STDOUT:   call void @_ZN9NeedThunkC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.60e3c6e2415a2ad6.Core:NeedThunk.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9NeedThunkC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy:thunk:Destroy.092ad14c7174f642.Core:NeedThunk.Cpp"(ptr %self) #2 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0f0801deab98555e(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy:thunk:Destroy.092ad14c7174f642.Core:NeedThunk.Cpp"(ptr %self), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!71, !72, !73, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!76}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !32 = !DIFile(filename: "multiple_thunks_in_one_class.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !39 = !{!34, !34}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !32, line: 8, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !48)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !34)
// CHECK:STDOUT: !44 = !DILocation(line: 9, column: 3, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 9, column: 34, scope: !41)
// CHECK:STDOUT: !47 = !DILocation(line: 8, column: 1, scope: !41)
// CHECK:STDOUT: !48 = !{!42}
// CHECK:STDOUT: !49 = !DIFile(filename: "./need_thunk.h", directory: "")
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.60e3c6e2415a2ad6.Core:NeedThunk.Cpp", scope: null, file: !49, line: 4, type: !40, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !34)
// CHECK:STDOUT: !52 = !DILocation(line: 4, column: 3, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy:thunk:Destroy.092ad14c7174f642.Core:NeedThunk.Cpp", scope: null, file: !49, line: 5, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !34)
// CHECK:STDOUT: !56 = !DILocation(line: 5, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0f0801deab98555e", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !34)
// CHECK:STDOUT: !62 = !DILocation(line: 26, column: 5, scope: !58)
// CHECK:STDOUT: !63 = !DILocation(line: 24, column: 3, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT: !65 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !66 = !{!"omnipotent char", !65, i64 0}
// CHECK:STDOUT: !67 = !{!"any pointer", !66, i64 0}
// CHECK:STDOUT: !68 = !{!"p1 _ZTS9NeedThunk", !67, i64 0}
// CHECK:STDOUT: !69 = !{!68, !68, i64 0}
// CHECK:STDOUT: !71 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !72 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !73 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !74 = !{!"int", !66, i64 0}
// CHECK:STDOUT: !75 = !{!"__libc_errno", !74, i64 0}
// CHECK:STDOUT: !76 = !{!75, !74, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunks_for_distinct_specializations.carbon'
// CHECK:STDOUT: source_filename = "thunks_for_distinct_specializations.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %a, ptr %b) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a2.var = alloca [1 x i8], align 1, !dbg !84
// CHECK:STDOUT:   %.loc12_35.1.temp = alloca [1 x i8], align 1, !dbg !85
// CHECK:STDOUT:   %b2.var = alloca [1 x i8], align 1, !dbg !87
// CHECK:STDOUT:   %.loc13_35.1.temp = alloca [1 x i8], align 1, !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a2.var), !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_35.1.temp), !dbg !85
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr %a, ptr %a2.var), !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_35.1.temp), !dbg !88
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr %b, ptr %b2.var), !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.9785c839355474a9(ptr %b2.var), !dbg !87
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.81eeecf785fc9a5b(ptr %a2.var), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !116
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !116
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !116
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !116
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.c0735c73762b4d1c.Core:Box.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr %self, ptr %return), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !120
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !120
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.2c1dd62357f2b914.Core:Box.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr %self, ptr %return), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !99
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.9785c839355474a9(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT:   call void @_ZN3BoxI1BED1Ev(ptr %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.81eeecf785fc9a5b(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT:   call void @_ZN3BoxI1AED1Ev(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !113
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!78}
// CHECK:STDOUT: !llvm.module.flags = !{!71, !72, !73, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!76}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !34, !34}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{!34, !34}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !65 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !66 = !{!"omnipotent char", !65, i64 0}
// CHECK:STDOUT: !67 = !{!"any pointer", !66, i64 0}
// CHECK:STDOUT: !71 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !72 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !73 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !74 = !{!"int", !66, i64 0}
// CHECK:STDOUT: !75 = !{!"__libc_errno", !74, i64 0}
// CHECK:STDOUT: !76 = !{!75, !74, i64 0}
// CHECK:STDOUT: !77 = !DIFile(filename: "thunks_for_distinct_specializations.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !77, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !77, line: 11, type: !38, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !91)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !34)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 2, scope: !79, type: !34)
// CHECK:STDOUT: !84 = !DILocation(line: 12, column: 3, scope: !79)
// CHECK:STDOUT: !85 = !DILocation(line: 12, column: 35, scope: !79)
// CHECK:STDOUT: !87 = !DILocation(line: 13, column: 3, scope: !79)
// CHECK:STDOUT: !88 = !DILocation(line: 13, column: 35, scope: !79)
// CHECK:STDOUT: !90 = !DILocation(line: 11, column: 1, scope: !79)
// CHECK:STDOUT: !91 = !{!80, !81}
// CHECK:STDOUT: !92 = !DIFile(filename: "./box.h", directory: "")
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.c0735c73762b4d1c.Core:Box.Cpp", scope: null, file: !92, line: 8, type: !40, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !34)
// CHECK:STDOUT: !95 = !DILocation(line: 8, column: 3, scope: !93)
// CHECK:STDOUT: !96 = !{!94}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.2c1dd62357f2b914.Core:Box.Cpp", scope: null, file: !92, line: 8, type: !40, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !100)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !34)
// CHECK:STDOUT: !99 = !DILocation(line: 8, column: 3, scope: !97)
// CHECK:STDOUT: !100 = !{!98}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.9785c839355474a9", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !107)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !34)
// CHECK:STDOUT: !105 = !DILocation(line: 26, column: 5, scope: !101)
// CHECK:STDOUT: !106 = !DILocation(line: 24, column: 3, scope: !101)
// CHECK:STDOUT: !107 = !{!102}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.81eeecf785fc9a5b", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !114)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !34)
// CHECK:STDOUT: !112 = !DILocation(line: 26, column: 5, scope: !108)
// CHECK:STDOUT: !113 = !DILocation(line: 24, column: 3, scope: !108)
// CHECK:STDOUT: !114 = !{!109}
// CHECK:STDOUT: !115 = !{!"p1 _ZTS3BoxI1AE", !67, i64 0}
// CHECK:STDOUT: !116 = !{!115, !115, i64 0}
// CHECK:STDOUT: !119 = !{!"p1 _ZTS3BoxI1BE", !67, i64 0}
// CHECK:STDOUT: !120 = !{!119, !119, i64 0}
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
// CHECK:STDOUT: define void @_CCall.Main(ptr %a, ptr %b) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a2.var = alloca [1 x i8], align 1, !dbg !130
// CHECK:STDOUT:   %.loc12_32.1.temp = alloca [1 x i8], align 1, !dbg !131
// CHECK:STDOUT:   %b2.var = alloca [1 x i8], align 1, !dbg !133
// CHECK:STDOUT:   %.loc13_36.1.temp = alloca [1 x i8], align 1, !dbg !134
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a2.var), !dbg !130
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_32.1.temp), !dbg !131
// CHECK:STDOUT:   call void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr %a, ptr %a2.var), !dbg !131
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !133
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_36.1.temp), !dbg !134
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr %b, ptr %b2.var), !dbg !134
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.195b7b5116a47bde(ptr %b2.var), !dbg !133
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.d458bc775710478b(ptr %a2.var), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !162
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !162
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !162
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !162
// CHECK:STDOUT:   call void @_ZN7CollideC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.9e522bb4dbf82e5c.Core:Collide.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !165
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !165
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !165
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !165
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC2ERKS0_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.d1fe27f3241437fa.Core:Collide..Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr %self, ptr %return), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !165
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7CollideD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !162
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.195b7b5116a47bde(ptr %self) #0 !dbg !147 {
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideD2Ev(ptr %self), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.d458bc775710478b(ptr %self) #0 !dbg !154 {
// CHECK:STDOUT:   call void @_ZN7CollideD2Ev(ptr %self), !dbg !158
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7CollideC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !162
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !162
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideC2ERKS0_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !165
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !165
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!124}
// CHECK:STDOUT: !llvm.module.flags = !{!71, !72, !73, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!76}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !28 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{null, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = !{null, !34, !34}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{!34, !34}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !65 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !66 = !{!"omnipotent char", !65, i64 0}
// CHECK:STDOUT: !67 = !{!"any pointer", !66, i64 0}
// CHECK:STDOUT: !71 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !72 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !73 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !74 = !{!"int", !66, i64 0}
// CHECK:STDOUT: !75 = !{!"__libc_errno", !74, i64 0}
// CHECK:STDOUT: !76 = !{!75, !74, i64 0}
// CHECK:STDOUT: !123 = !DIFile(filename: "thunks_for_anonymous_namespace_collision.carbon", directory: "")
// CHECK:STDOUT: !124 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !123, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !123, line: 11, type: !38, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !137)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !34)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 2, scope: !125, type: !34)
// CHECK:STDOUT: !130 = !DILocation(line: 12, column: 3, scope: !125)
// CHECK:STDOUT: !131 = !DILocation(line: 12, column: 32, scope: !125)
// CHECK:STDOUT: !133 = !DILocation(line: 13, column: 3, scope: !125)
// CHECK:STDOUT: !134 = !DILocation(line: 13, column: 36, scope: !125)
// CHECK:STDOUT: !136 = !DILocation(line: 11, column: 1, scope: !125)
// CHECK:STDOUT: !137 = !{!126, !127}
// CHECK:STDOUT: !138 = !DIFile(filename: "./anonymous_namespace_collision.h", directory: "")
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.9e522bb4dbf82e5c.Core:Collide.Cpp", scope: null, file: !138, line: 17, type: !40, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !142)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !34)
// CHECK:STDOUT: !141 = !DILocation(line: 17, column: 3, scope: !139)
// CHECK:STDOUT: !142 = !{!140}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.d1fe27f3241437fa.Core:Collide..Cpp", scope: null, file: !138, line: 8, type: !40, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !146)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !34)
// CHECK:STDOUT: !145 = !DILocation(line: 8, column: 3, scope: !143)
// CHECK:STDOUT: !146 = !{!144}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.195b7b5116a47bde", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !153)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !34)
// CHECK:STDOUT: !151 = !DILocation(line: 26, column: 5, scope: !147)
// CHECK:STDOUT: !152 = !DILocation(line: 24, column: 3, scope: !147)
// CHECK:STDOUT: !153 = !{!148}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.d458bc775710478b", scope: null, file: !28, line: 24, type: !36, spFlags: DISPFlagDefinition, unit: !124, retainedNodes: !160)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !34)
// CHECK:STDOUT: !158 = !DILocation(line: 26, column: 5, scope: !154)
// CHECK:STDOUT: !159 = !DILocation(line: 24, column: 3, scope: !154)
// CHECK:STDOUT: !160 = !{!155}
// CHECK:STDOUT: !161 = !{!"p1 _ZTS7Collide", !67, i64 0}
// CHECK:STDOUT: !162 = !{!161, !161, i64 0}
// CHECK:STDOUT: !164 = !{!"p1 _ZTSN12_GLOBAL__N_17CollideE", !67, i64 0}
// CHECK:STDOUT: !165 = !{!164, !164, i64 0}
// CHECK:STDOUT:
