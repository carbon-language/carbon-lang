// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --share-cpp-ast
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/share_ast.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/share_ast.carbon

// --- a.h
#ifndef A_H
#define A_H
struct A {
  int a, b;
};
inline A make_a() { return {1, 2}; }
#endif

// --- b.h
#ifndef B_H
#define B_H
struct B {
  int x, y;
};
inline B make_b() { return {3, 4}; }
#endif

// --- use_a.carbon
library "[[@TEST_NAME]]";
import Cpp library "a.h";

fn Test() -> i32 {
  var a: Cpp.A = Cpp.make_a();
  return a.b;
}

// --- use_b.carbon
library "[[@TEST_NAME]]";
import Cpp library "b.h";

fn Test() -> i32 {
  var b: Cpp.B = Cpp.make_b();
  return b.x;
}

// --- use_both.carbon
library "[[@TEST_NAME]]";
import Cpp library "a.h";
import Cpp library "b.h";

fn TestA() -> i32 {
  var a: Cpp.A = Cpp.make_a();
  return a.b;
}

fn TestB() -> i32 {
  var b: Cpp.B = Cpp.make_b();
  return b.x;
}

// --- generic_lib.carbon
library "[[@TEST_NAME]]";
import Cpp library "a.h";

fn MakeA[T: type](unused x: T) -> i32 {
  var a: Cpp.A = Cpp.make_a();
  return a.b;
}

// --- no_cpp_imports.carbon
library "[[@TEST_NAME]]";
import library "generic_lib";

fn CallGeneric() -> i32 {
  return MakeA(0);
}


// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_a.carbon'
// CHECK:STDOUT: source_filename = "use_a.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_av = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !53
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !54
// CHECK:STDOUT:   %.loc6_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !56
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.b, align 4, !dbg !56
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %a.var), !dbg !53
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %t) #0 !dbg !62 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %t), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !73
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !74
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !10, !11}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!9}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !6 = !{!"omnipotent char", !5, i64 0}
// CHECK:STDOUT: !7 = !{!"int", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !7, i64 0}
// CHECK:STDOUT: !9 = !{!8, !7, i64 0}
// CHECK:STDOUT: !10 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !11 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !42 = !DIFile(filename: "use_a.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !45 = !{!44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !48 = !{null, !47}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !42, line: 4, type: !46, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !53 = !DILocation(line: 5, column: 3, scope: !50)
// CHECK:STDOUT: !54 = !DILocation(line: 5, column: 18, scope: !50)
// CHECK:STDOUT: !56 = !DILocation(line: 6, column: 10, scope: !50)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 3, scope: !50)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core", scope: null, file: !42, type: !49, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !47)
// CHECK:STDOUT: !60 = !DILocation(line: 0, scope: !58)
// CHECK:STDOUT: !61 = !{!59}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6191dcb71c708f97", scope: null, file: !36, line: 26, type: !49, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !47)
// CHECK:STDOUT: !65 = !DILocation(line: 27, column: 5, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 26, column: 1, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !70 = !{!"p1 _ZTS1A", !69, i64 0}
// CHECK:STDOUT: !71 = !{!70, !70, i64 0}
// CHECK:STDOUT: !72 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !73 = !{!72, !7, i64 0}
// CHECK:STDOUT: !74 = !{!72, !7, i64 4}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_b.carbon'
// CHECK:STDOUT: source_filename = "use_b.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.B = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_bv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !80
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !81
// CHECK:STDOUT:   %.loc6_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !83
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.x, align 4, !dbg !83
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a4f045cd46b2d152(ptr %b.var), !dbg !80
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !97
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !97
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.75c23c1f781e9fb6.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a4f045cd46b2d152(ptr %t) #0 !dbg !89 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.75c23c1f781e9fb6.Core:DestroyImpl.Core"(ptr %t), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !99
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !100
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!76}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !10, !11}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!9}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !6 = !{!"omnipotent char", !5, i64 0}
// CHECK:STDOUT: !7 = !{!"int", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !7, i64 0}
// CHECK:STDOUT: !9 = !{!8, !7, i64 0}
// CHECK:STDOUT: !10 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !11 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !44 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !45 = !{!44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !48 = !{null, !47}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !69 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !75 = !DIFile(filename: "use_b.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !75, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !75, line: 4, type: !46, spFlags: DISPFlagDefinition, unit: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 5, column: 3, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 5, column: 18, scope: !77)
// CHECK:STDOUT: !83 = !DILocation(line: 6, column: 10, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 6, column: 3, scope: !77)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.75c23c1f781e9fb6.Core:DestroyImpl.Core", scope: null, file: !75, type: !49, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !88)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !47)
// CHECK:STDOUT: !87 = !DILocation(line: 0, scope: !85)
// CHECK:STDOUT: !88 = !{!86}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a4f045cd46b2d152", scope: null, file: !36, line: 26, type: !49, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !95)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !47)
// CHECK:STDOUT: !92 = !DILocation(line: 27, column: 5, scope: !89)
// CHECK:STDOUT: !94 = !DILocation(line: 26, column: 1, scope: !89)
// CHECK:STDOUT: !95 = !{!90}
// CHECK:STDOUT: !96 = !{!"p1 _ZTS1B", !69, i64 0}
// CHECK:STDOUT: !97 = !{!96, !96, i64 0}
// CHECK:STDOUT: !98 = !{!"_ZTS1B", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !99 = !{!98, !7, i64 0}
// CHECK:STDOUT: !100 = !{!98, !7, i64 4}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_both.carbon'
// CHECK:STDOUT: source_filename = "use_both.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A.34 = type { i32, i32 }
// CHECK:STDOUT: %struct.B.35 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_av = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_bv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestA.Main() #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !106
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !106
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !107
// CHECK:STDOUT:   %.loc7_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !109
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.b, align 4, !dbg !109
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %a.var), !dbg !106
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestB.Main() #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !118
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !118
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !119
// CHECK:STDOUT:   %.loc12_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !121
// CHECK:STDOUT:   %.loc12_11.2 = load i32, ptr %.loc12_11.1.x, align 4, !dbg !121
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a4f045cd46b2d152(ptr %b.var), !dbg !118
// CHECK:STDOUT:   ret i32 %.loc12_11.2, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !97
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !97
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.75c23c1f781e9fb6.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %t) #0 !dbg !127 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %t), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a4f045cd46b2d152(ptr %t) #0 !dbg !134 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.75c23c1f781e9fb6.Core:DestroyImpl.Core"(ptr %t), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.34, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.34, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !73
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.34, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !74
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B.35, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B.35, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !99
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B.35, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !100
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!102}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !10, !11}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!9}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !6 = !{!"omnipotent char", !5, i64 0}
// CHECK:STDOUT: !7 = !{!"int", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !7, i64 0}
// CHECK:STDOUT: !9 = !{!8, !7, i64 0}
// CHECK:STDOUT: !10 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !11 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !44 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !45 = !{!44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !48 = !{null, !47}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !69 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !70 = !{!"p1 _ZTS1A", !69, i64 0}
// CHECK:STDOUT: !71 = !{!70, !70, i64 0}
// CHECK:STDOUT: !72 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !73 = !{!72, !7, i64 0}
// CHECK:STDOUT: !74 = !{!72, !7, i64 4}
// CHECK:STDOUT: !96 = !{!"p1 _ZTS1B", !69, i64 0}
// CHECK:STDOUT: !97 = !{!96, !96, i64 0}
// CHECK:STDOUT: !98 = !{!"_ZTS1B", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !99 = !{!98, !7, i64 0}
// CHECK:STDOUT: !100 = !{!98, !7, i64 4}
// CHECK:STDOUT: !101 = !DIFile(filename: "use_both.carbon", directory: "")
// CHECK:STDOUT: !102 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !101, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "TestA", linkageName: "_CTestA.Main", scope: null, file: !101, line: 5, type: !46, spFlags: DISPFlagDefinition, unit: !102)
// CHECK:STDOUT: !106 = !DILocation(line: 6, column: 3, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 6, column: 18, scope: !103)
// CHECK:STDOUT: !109 = !DILocation(line: 7, column: 10, scope: !103)
// CHECK:STDOUT: !110 = !DILocation(line: 7, column: 3, scope: !103)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core", scope: null, file: !101, type: !49, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !47)
// CHECK:STDOUT: !113 = !DILocation(line: 0, scope: !111)
// CHECK:STDOUT: !114 = !{!112}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "TestB", linkageName: "_CTestB.Main", scope: null, file: !101, line: 10, type: !46, spFlags: DISPFlagDefinition, unit: !102)
// CHECK:STDOUT: !118 = !DILocation(line: 11, column: 3, scope: !115)
// CHECK:STDOUT: !119 = !DILocation(line: 11, column: 18, scope: !115)
// CHECK:STDOUT: !121 = !DILocation(line: 12, column: 10, scope: !115)
// CHECK:STDOUT: !122 = !DILocation(line: 12, column: 3, scope: !115)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.75c23c1f781e9fb6.Core:DestroyImpl.Core", scope: null, file: !101, type: !49, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !126)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !47)
// CHECK:STDOUT: !125 = !DILocation(line: 0, scope: !123)
// CHECK:STDOUT: !126 = !{!124}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6191dcb71c708f97", scope: null, file: !36, line: 26, type: !49, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !133)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !47)
// CHECK:STDOUT: !130 = !DILocation(line: 27, column: 5, scope: !127)
// CHECK:STDOUT: !132 = !DILocation(line: 26, column: 1, scope: !127)
// CHECK:STDOUT: !133 = !{!128}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a4f045cd46b2d152", scope: null, file: !36, line: 26, type: !49, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !140)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !47)
// CHECK:STDOUT: !137 = !DILocation(line: 27, column: 5, scope: !134)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 1, scope: !134)
// CHECK:STDOUT: !140 = !{!135}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_lib.carbon'
// CHECK:STDOUT: source_filename = "generic_lib.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A.38 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_av = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.38, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.38, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !73
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.38, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !74
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!142}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !10, !11}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!9}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !6 = !{!"omnipotent char", !5, i64 0}
// CHECK:STDOUT: !7 = !{!"int", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !7, i64 0}
// CHECK:STDOUT: !9 = !{!8, !7, i64 0}
// CHECK:STDOUT: !10 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !11 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !69 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !70 = !{!"p1 _ZTS1A", !69, i64 0}
// CHECK:STDOUT: !71 = !{!70, !70, i64 0}
// CHECK:STDOUT: !72 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !73 = !{!72, !7, i64 0}
// CHECK:STDOUT: !74 = !{!72, !7, i64 4}
// CHECK:STDOUT: !141 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !142 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !141, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_cpp_imports.carbon'
// CHECK:STDOUT: source_filename = "no_cpp_imports.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A.41 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_av = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallGeneric.Main() #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeA.call = call i32 @_CMakeA.Main.084a2801e0f5d754({} zeroinitializer), !dbg !148
// CHECK:STDOUT:   ret i32 %MakeA.call, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CMakeA.Main.084a2801e0f5d754({} %x) #0 !dbg !156 {
// CHECK:STDOUT:   %1 = alloca [8 x i8], align 4, !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !161
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %1), !dbg !162
// CHECK:STDOUT:   %b = getelementptr inbounds nuw [8 x i8], ptr %1, i32 0, i32 4, !dbg !164
// CHECK:STDOUT:   %2 = load i32, ptr %b, align 4, !dbg !164
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3c48d1e836afefe8(ptr %1), !dbg !161
// CHECK:STDOUT:   ret i32 %2, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !71
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3c48d1e836afefe8(ptr %t) #0 !dbg !167 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %t), !dbg !170
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.41, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.41, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !73
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.41, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !74
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!144}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !10, !11}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!9}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !6 = !{!"omnipotent char", !5, i64 0}
// CHECK:STDOUT: !7 = !{!"int", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !7, i64 0}
// CHECK:STDOUT: !9 = !{!8, !7, i64 0}
// CHECK:STDOUT: !10 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !11 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !44 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !45 = !{!44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !48 = !{null, !47}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !69 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !70 = !{!"p1 _ZTS1A", !69, i64 0}
// CHECK:STDOUT: !71 = !{!70, !70, i64 0}
// CHECK:STDOUT: !72 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !73 = !{!72, !7, i64 0}
// CHECK:STDOUT: !74 = !{!72, !7, i64 4}
// CHECK:STDOUT: !141 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !143 = !DIFile(filename: "no_cpp_imports.carbon", directory: "")
// CHECK:STDOUT: !144 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !143, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main", scope: null, file: !143, line: 4, type: !46, spFlags: DISPFlagDefinition, unit: !144)
// CHECK:STDOUT: !148 = !DILocation(line: 5, column: 10, scope: !145)
// CHECK:STDOUT: !149 = !{!44, !47}
// CHECK:STDOUT: !150 = !DISubroutineType(types: !149)
// CHECK:STDOUT: !151 = !DILocation(line: 5, column: 3, scope: !145)
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core", scope: null, file: !143, type: !49, spFlags: DISPFlagDefinition, unit: !144, retainedNodes: !155)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !47)
// CHECK:STDOUT: !154 = !DILocation(line: 0, scope: !152)
// CHECK:STDOUT: !155 = !{!153}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.084a2801e0f5d754", scope: null, file: !141, line: 4, type: !150, spFlags: DISPFlagDefinition, unit: !144, retainedNodes: !166)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !47)
// CHECK:STDOUT: !161 = !DILocation(line: 5, column: 3, scope: !156)
// CHECK:STDOUT: !162 = !DILocation(line: 5, column: 18, scope: !156)
// CHECK:STDOUT: !164 = !DILocation(line: 6, column: 10, scope: !156)
// CHECK:STDOUT: !165 = !DILocation(line: 6, column: 3, scope: !156)
// CHECK:STDOUT: !166 = !{!157}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3c48d1e836afefe8", scope: null, file: !36, line: 26, type: !49, spFlags: DISPFlagDefinition, unit: !144, retainedNodes: !173)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !47)
// CHECK:STDOUT: !170 = !DILocation(line: 27, column: 5, scope: !167)
// CHECK:STDOUT: !172 = !DILocation(line: 26, column: 1, scope: !167)
// CHECK:STDOUT: !173 = !{!168}
// CHECK:STDOUT:
