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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b06d72f03f121b70(ptr %a.var), !dbg !53
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b06d72f03f121b70(ptr %t) #0 !dbg !58 {
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !69
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !70
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !34 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b06d72f03f121b70", scope: null, file: !34, line: 32, type: !49, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !47)
// CHECK:STDOUT: !63 = !DILocation(line: 32, column: 1, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT: !65 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !66 = !{!"p1 _ZTS1A", !65, i64 0}
// CHECK:STDOUT: !67 = !{!66, !66, i64 0}
// CHECK:STDOUT: !68 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !69 = !{!68, !7, i64 0}
// CHECK:STDOUT: !70 = !{!68, !7, i64 4}
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
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !76
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !77
// CHECK:STDOUT:   %.loc6_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !79
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.x, align 4, !dbg !79
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6ab41dc29a50b530(ptr %b.var), !dbg !76
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6ab41dc29a50b530(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !91
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !92
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!72}
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
// CHECK:STDOUT: !34 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !44 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !45 = !{!44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !48 = !{null, !47}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !65 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !71 = !DIFile(filename: "use_b.carbon", directory: "")
// CHECK:STDOUT: !72 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !71, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !71, line: 4, type: !46, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 5, column: 3, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 5, column: 18, scope: !73)
// CHECK:STDOUT: !79 = !DILocation(line: 6, column: 10, scope: !73)
// CHECK:STDOUT: !80 = !DILocation(line: 6, column: 3, scope: !73)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6ab41dc29a50b530", scope: null, file: !34, line: 32, type: !49, spFlags: DISPFlagDefinition, unit: !72, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !47)
// CHECK:STDOUT: !86 = !DILocation(line: 32, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = !{!"p1 _ZTS1B", !65, i64 0}
// CHECK:STDOUT: !89 = !{!88, !88, i64 0}
// CHECK:STDOUT: !90 = !{!"_ZTS1B", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !91 = !{!90, !7, i64 0}
// CHECK:STDOUT: !92 = !{!90, !7, i64 4}
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
// CHECK:STDOUT: define i32 @_CTestA.Main() #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !98
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !98
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !99
// CHECK:STDOUT:   %.loc7_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !101
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.b, align 4, !dbg !101
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b06d72f03f121b70(ptr %a.var), !dbg !98
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestB.Main() #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !106
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !106
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !107
// CHECK:STDOUT:   %.loc12_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !109
// CHECK:STDOUT:   %.loc12_11.2 = load i32, ptr %.loc12_11.1.x, align 4, !dbg !109
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6ab41dc29a50b530(ptr %b.var), !dbg !106
// CHECK:STDOUT:   ret i32 %.loc12_11.2, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b06d72f03f121b70(ptr %t) #0 !dbg !111 {
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6ab41dc29a50b530(ptr %t) #0 !dbg !118 {
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.34, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.34, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !69
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.34, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !70
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B.35, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B.35, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !91
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B.35, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !92
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!94}
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
// CHECK:STDOUT: !34 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !44 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !45 = !{!44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !48 = !{null, !47}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !65 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !66 = !{!"p1 _ZTS1A", !65, i64 0}
// CHECK:STDOUT: !67 = !{!66, !66, i64 0}
// CHECK:STDOUT: !68 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !69 = !{!68, !7, i64 0}
// CHECK:STDOUT: !70 = !{!68, !7, i64 4}
// CHECK:STDOUT: !88 = !{!"p1 _ZTS1B", !65, i64 0}
// CHECK:STDOUT: !89 = !{!88, !88, i64 0}
// CHECK:STDOUT: !90 = !{!"_ZTS1B", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !91 = !{!90, !7, i64 0}
// CHECK:STDOUT: !92 = !{!90, !7, i64 4}
// CHECK:STDOUT: !93 = !DIFile(filename: "use_both.carbon", directory: "")
// CHECK:STDOUT: !94 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !93, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "TestA", linkageName: "_CTestA.Main", scope: null, file: !93, line: 5, type: !46, spFlags: DISPFlagDefinition, unit: !94)
// CHECK:STDOUT: !98 = !DILocation(line: 6, column: 3, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 6, column: 18, scope: !95)
// CHECK:STDOUT: !101 = !DILocation(line: 7, column: 10, scope: !95)
// CHECK:STDOUT: !102 = !DILocation(line: 7, column: 3, scope: !95)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "TestB", linkageName: "_CTestB.Main", scope: null, file: !93, line: 10, type: !46, spFlags: DISPFlagDefinition, unit: !94)
// CHECK:STDOUT: !106 = !DILocation(line: 11, column: 3, scope: !103)
// CHECK:STDOUT: !107 = !DILocation(line: 11, column: 18, scope: !103)
// CHECK:STDOUT: !109 = !DILocation(line: 12, column: 10, scope: !103)
// CHECK:STDOUT: !110 = !DILocation(line: 12, column: 3, scope: !103)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b06d72f03f121b70", scope: null, file: !34, line: 32, type: !49, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !117)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !47)
// CHECK:STDOUT: !116 = !DILocation(line: 32, column: 1, scope: !111)
// CHECK:STDOUT: !117 = !{!112}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6ab41dc29a50b530", scope: null, file: !34, line: 32, type: !49, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !47)
// CHECK:STDOUT: !123 = !DILocation(line: 32, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !67
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
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !69
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.38, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !70
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!126}
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
// CHECK:STDOUT: !65 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !66 = !{!"p1 _ZTS1A", !65, i64 0}
// CHECK:STDOUT: !67 = !{!66, !66, i64 0}
// CHECK:STDOUT: !68 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !69 = !{!68, !7, i64 0}
// CHECK:STDOUT: !70 = !{!68, !7, i64 4}
// CHECK:STDOUT: !125 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !126 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !125, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: define i32 @_CCallGeneric.Main() #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeA.call = call i32 @_CMakeA.Main.084a2801e0f5d754({} zeroinitializer), !dbg !132
// CHECK:STDOUT:   ret i32 %MakeA.call, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CMakeA.Main.084a2801e0f5d754({} %x) #0 !dbg !136 {
// CHECK:STDOUT:   %1 = alloca [8 x i8], align 4, !dbg !141
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !141
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %1), !dbg !142
// CHECK:STDOUT:   %b = getelementptr inbounds nuw [8 x i8], ptr %1, i32 0, i32 4, !dbg !144
// CHECK:STDOUT:   %2 = load i32, ptr %b, align 4, !dbg !144
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b06d72f03f121b70(ptr %1), !dbg !141
// CHECK:STDOUT:   ret i32 %2, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b06d72f03f121b70(ptr %t) #0 !dbg !147 {
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.41, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.41, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !69
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.41, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !70
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!128}
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
// CHECK:STDOUT: !34 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !44 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !45 = !{!44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !48 = !{null, !47}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !65 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !66 = !{!"p1 _ZTS1A", !65, i64 0}
// CHECK:STDOUT: !67 = !{!66, !66, i64 0}
// CHECK:STDOUT: !68 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !69 = !{!68, !7, i64 0}
// CHECK:STDOUT: !70 = !{!68, !7, i64 4}
// CHECK:STDOUT: !125 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !127 = !DIFile(filename: "no_cpp_imports.carbon", directory: "")
// CHECK:STDOUT: !128 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !127, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main", scope: null, file: !127, line: 4, type: !46, spFlags: DISPFlagDefinition, unit: !128)
// CHECK:STDOUT: !132 = !DILocation(line: 5, column: 10, scope: !129)
// CHECK:STDOUT: !133 = !{!44, !47}
// CHECK:STDOUT: !134 = !DISubroutineType(types: !133)
// CHECK:STDOUT: !135 = !DILocation(line: 5, column: 3, scope: !129)
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.084a2801e0f5d754", scope: null, file: !125, line: 4, type: !134, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !146)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !47)
// CHECK:STDOUT: !141 = !DILocation(line: 5, column: 3, scope: !136)
// CHECK:STDOUT: !142 = !DILocation(line: 5, column: 18, scope: !136)
// CHECK:STDOUT: !144 = !DILocation(line: 6, column: 10, scope: !136)
// CHECK:STDOUT: !145 = !DILocation(line: 6, column: 3, scope: !136)
// CHECK:STDOUT: !146 = !{!137}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b06d72f03f121b70", scope: null, file: !34, line: 32, type: !49, spFlags: DISPFlagDefinition, unit: !128, retainedNodes: !153)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !47)
// CHECK:STDOUT: !152 = !DILocation(line: 32, column: 1, scope: !147)
// CHECK:STDOUT: !153 = !{!148}
// CHECK:STDOUT:
