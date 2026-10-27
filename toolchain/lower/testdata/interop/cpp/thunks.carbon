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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c717f465f400f4c6(ptr %n2.var), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9NeedThunkC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !67
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !67
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
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c717f465f400f4c6(ptr %t) #0 !dbg !56 {
// CHECK:STDOUT:   ret void, !dbg !61
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
// CHECK:STDOUT: !llvm.module.flags = !{!69, !70, !71, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!74}
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
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c717f465f400f4c6", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !62)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !36)
// CHECK:STDOUT: !61 = !DILocation(line: 32, column: 1, scope: !56)
// CHECK:STDOUT: !62 = !{!57}
// CHECK:STDOUT: !63 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !64 = !{!"omnipotent char", !63, i64 0}
// CHECK:STDOUT: !65 = !{!"any pointer", !64, i64 0}
// CHECK:STDOUT: !66 = !{!"p1 _ZTS9NeedThunk", !65, i64 0}
// CHECK:STDOUT: !67 = !{!66, !66, i64 0}
// CHECK:STDOUT: !69 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !70 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !71 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !72 = !{!"int", !64, i64 0}
// CHECK:STDOUT: !73 = !{!"__libc_errno", !72, i64 0}
// CHECK:STDOUT: !74 = !{!73, !72, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'thunks_for_distinct_specializations.carbon'
// CHECK:STDOUT: source_filename = "thunks_for_distinct_specializations.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main(ptr %a, ptr %b) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a2.var = alloca [1 x i8], align 1, !dbg !82
// CHECK:STDOUT:   %.loc12_35.1.temp = alloca [1 x i8], align 1, !dbg !83
// CHECK:STDOUT:   %b2.var = alloca [1 x i8], align 1, !dbg !85
// CHECK:STDOUT:   %.loc13_35.1.temp = alloca [1 x i8], align 1, !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a2.var), !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_35.1.temp), !dbg !83
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr %a, ptr %a2.var), !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_35.1.temp), !dbg !86
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr %b, ptr %b2.var), !dbg !86
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d1e44ef097900801(ptr %b2.var), !dbg !85
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5fbb58cefed751bc(ptr %a2.var), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !114
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !114
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !114
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !114
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.c0735c73762b4d1c.Core:Box.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN3BoxI1AEC1ERKS1_.carbon_thunk._(ptr %self, ptr %return), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !118
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !118
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.2c1dd62357f2b914.Core:Box.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN3BoxI1BEC1ERKS1_.carbon_thunk._(ptr %self, ptr %return), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d1e44ef097900801(ptr %t) #0 !dbg !99 {
// CHECK:STDOUT:   call void @_ZN3BoxI1BED1Ev(ptr %t), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5fbb58cefed751bc(ptr %t) #0 !dbg !106 {
// CHECK:STDOUT:   call void @_ZN3BoxI1AED1Ev(ptr %t), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!76}
// CHECK:STDOUT: !llvm.module.flags = !{!69, !70, !71, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!74}
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
// CHECK:STDOUT: !63 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !64 = !{!"omnipotent char", !63, i64 0}
// CHECK:STDOUT: !65 = !{!"any pointer", !64, i64 0}
// CHECK:STDOUT: !69 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !70 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !71 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !72 = !{!"int", !64, i64 0}
// CHECK:STDOUT: !73 = !{!"__libc_errno", !72, i64 0}
// CHECK:STDOUT: !74 = !{!73, !72, i64 0}
// CHECK:STDOUT: !75 = !DIFile(filename: "thunks_for_distinct_specializations.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !75, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !75, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !89)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !36)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 2, scope: !77, type: !36)
// CHECK:STDOUT: !82 = !DILocation(line: 12, column: 3, scope: !77)
// CHECK:STDOUT: !83 = !DILocation(line: 12, column: 35, scope: !77)
// CHECK:STDOUT: !85 = !DILocation(line: 13, column: 3, scope: !77)
// CHECK:STDOUT: !86 = !DILocation(line: 13, column: 35, scope: !77)
// CHECK:STDOUT: !88 = !DILocation(line: 11, column: 1, scope: !77)
// CHECK:STDOUT: !89 = !{!78, !79}
// CHECK:STDOUT: !90 = !DIFile(filename: "./box.h", directory: "")
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.c0735c73762b4d1c.Core:Box.Cpp", scope: null, file: !90, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !36)
// CHECK:STDOUT: !93 = !DILocation(line: 8, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.2c1dd62357f2b914.Core:Box.Cpp", scope: null, file: !90, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !36)
// CHECK:STDOUT: !97 = !DILocation(line: 8, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d1e44ef097900801", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !105)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !36)
// CHECK:STDOUT: !103 = !DILocation(line: 34, column: 5, scope: !99)
// CHECK:STDOUT: !104 = !DILocation(line: 32, column: 1, scope: !99)
// CHECK:STDOUT: !105 = !{!100}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5fbb58cefed751bc", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !112)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !36)
// CHECK:STDOUT: !110 = !DILocation(line: 34, column: 5, scope: !106)
// CHECK:STDOUT: !111 = !DILocation(line: 32, column: 1, scope: !106)
// CHECK:STDOUT: !112 = !{!107}
// CHECK:STDOUT: !113 = !{!"p1 _ZTS3BoxI1AE", !65, i64 0}
// CHECK:STDOUT: !114 = !{!113, !113, i64 0}
// CHECK:STDOUT: !117 = !{!"p1 _ZTS3BoxI1BE", !65, i64 0}
// CHECK:STDOUT: !118 = !{!117, !117, i64 0}
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
// CHECK:STDOUT: define void @_CCall.Main(ptr %a, ptr %b) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a2.var = alloca [1 x i8], align 1, !dbg !128
// CHECK:STDOUT:   %.loc12_32.1.temp = alloca [1 x i8], align 1, !dbg !129
// CHECK:STDOUT:   %b2.var = alloca [1 x i8], align 1, !dbg !131
// CHECK:STDOUT:   %.loc13_36.1.temp = alloca [1 x i8], align 1, !dbg !132
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a2.var), !dbg !128
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_32.1.temp), !dbg !129
// CHECK:STDOUT:   call void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr %a, ptr %a2.var), !dbg !129
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !131
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_36.1.temp), !dbg !132
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr %b, ptr %b2.var), !dbg !132
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c89f00da2163eb95(ptr %b2.var), !dbg !131
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0b81af486af557dc(ptr %a2.var), !dbg !128
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !160
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !160
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !160
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !160
// CHECK:STDOUT:   call void @_ZN7CollideC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.9e522bb4dbf82e5c.Core:Collide.Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN7CollideC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !139
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !163
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !163
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !163
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !163
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC2ERKS0_(ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.d1fe27f3241437fa.Core:Collide..Cpp"(ptr sret([1 x i8]) %return, ptr %self) #2 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideC1ERKS0_.carbon_thunk._(ptr %self, ptr %return), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !163
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7CollideD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !160
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c89f00da2163eb95(ptr %t) #0 !dbg !145 {
// CHECK:STDOUT:   call void @_ZN12_GLOBAL__N_17CollideD2Ev(ptr %t), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0b81af486af557dc(ptr %t) #0 !dbg !152 {
// CHECK:STDOUT:   call void @_ZN7CollideD2Ev(ptr %t), !dbg !156
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7CollideC2ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !160
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !160
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN12_GLOBAL__N_17CollideC2ERKS0_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !163
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !163
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!122}
// CHECK:STDOUT: !llvm.module.flags = !{!69, !70, !71, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!74}
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
// CHECK:STDOUT: !63 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !64 = !{!"omnipotent char", !63, i64 0}
// CHECK:STDOUT: !65 = !{!"any pointer", !64, i64 0}
// CHECK:STDOUT: !69 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !70 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !71 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !72 = !{!"int", !64, i64 0}
// CHECK:STDOUT: !73 = !{!"__libc_errno", !72, i64 0}
// CHECK:STDOUT: !74 = !{!73, !72, i64 0}
// CHECK:STDOUT: !121 = !DIFile(filename: "thunks_for_anonymous_namespace_collision.carbon", directory: "")
// CHECK:STDOUT: !122 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !121, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !121, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !135)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !36)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 2, scope: !123, type: !36)
// CHECK:STDOUT: !128 = !DILocation(line: 12, column: 3, scope: !123)
// CHECK:STDOUT: !129 = !DILocation(line: 12, column: 32, scope: !123)
// CHECK:STDOUT: !131 = !DILocation(line: 13, column: 3, scope: !123)
// CHECK:STDOUT: !132 = !DILocation(line: 13, column: 36, scope: !123)
// CHECK:STDOUT: !134 = !DILocation(line: 11, column: 1, scope: !123)
// CHECK:STDOUT: !135 = !{!124, !125}
// CHECK:STDOUT: !136 = !DIFile(filename: "./anonymous_namespace_collision.h", directory: "")
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.9e522bb4dbf82e5c.Core:Collide.Cpp", scope: null, file: !136, line: 17, type: !42, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !140)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !36)
// CHECK:STDOUT: !139 = !DILocation(line: 17, column: 3, scope: !137)
// CHECK:STDOUT: !140 = !{!138}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.d1fe27f3241437fa.Core:Collide..Cpp", scope: null, file: !136, line: 8, type: !42, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !144)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !36)
// CHECK:STDOUT: !143 = !DILocation(line: 8, column: 3, scope: !141)
// CHECK:STDOUT: !144 = !{!142}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c89f00da2163eb95", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !151)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !36)
// CHECK:STDOUT: !149 = !DILocation(line: 34, column: 5, scope: !145)
// CHECK:STDOUT: !150 = !DILocation(line: 32, column: 1, scope: !145)
// CHECK:STDOUT: !151 = !{!146}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0b81af486af557dc", scope: null, file: !26, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !158)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !36)
// CHECK:STDOUT: !156 = !DILocation(line: 34, column: 5, scope: !152)
// CHECK:STDOUT: !157 = !DILocation(line: 32, column: 1, scope: !152)
// CHECK:STDOUT: !158 = !{!153}
// CHECK:STDOUT: !159 = !{!"p1 _ZTS7Collide", !65, i64 0}
// CHECK:STDOUT: !160 = !{!159, !159, i64 0}
// CHECK:STDOUT: !162 = !{!"p1 _ZTSN12_GLOBAL__N_17CollideE", !65, i64 0}
// CHECK:STDOUT: !163 = !{!162, !162, i64 0}
// CHECK:STDOUT:
