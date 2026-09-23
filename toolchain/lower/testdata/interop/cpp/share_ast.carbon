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
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !51
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !52
// CHECK:STDOUT:   %.loc6_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !54
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.b, align 4, !dbg !54
// CHECK:STDOUT:   call void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %a.var), !dbg !51
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !64
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !65
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
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
// CHECK:STDOUT: !40 = !DIFile(filename: "use_a.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{!42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{null, !45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !40, line: 4, type: !44, spFlags: DISPFlagDefinition, unit: !41)
// CHECK:STDOUT: !51 = !DILocation(line: 5, column: 3, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 5, column: 18, scope: !48)
// CHECK:STDOUT: !54 = !DILocation(line: 6, column: 10, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 6, column: 3, scope: !48)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core", scope: null, file: !40, line: 5, type: !47, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !59)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !45)
// CHECK:STDOUT: !58 = !DILocation(line: 5, column: 3, scope: !56)
// CHECK:STDOUT: !59 = !{!57}
// CHECK:STDOUT: !60 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !61 = !{!"p1 _ZTS1A", !60, i64 0}
// CHECK:STDOUT: !62 = !{!61, !61, i64 0}
// CHECK:STDOUT: !63 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !64 = !{!63, !7, i64 0}
// CHECK:STDOUT: !65 = !{!63, !7, i64 4}
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
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !71
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !71
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !72
// CHECK:STDOUT:   %.loc6_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !74
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.x, align 4, !dbg !74
// CHECK:STDOUT:   call void @"_CSelfDestruct.d4378aa76ca99c88:core.Destroy.Core"(ptr %b.var), !dbg !71
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !81
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !81
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d4378aa76ca99c88:core.Destroy.Core"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !83
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !84
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!67}
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
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{!42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{null, !45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !60 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !66 = !DIFile(filename: "use_b.carbon", directory: "")
// CHECK:STDOUT: !67 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !66, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !66, line: 4, type: !44, spFlags: DISPFlagDefinition, unit: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 5, column: 3, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 5, column: 18, scope: !68)
// CHECK:STDOUT: !74 = !DILocation(line: 6, column: 10, scope: !68)
// CHECK:STDOUT: !75 = !DILocation(line: 6, column: 3, scope: !68)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d4378aa76ca99c88:core.Destroy.Core", scope: null, file: !66, line: 5, type: !47, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !79)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !45)
// CHECK:STDOUT: !78 = !DILocation(line: 5, column: 3, scope: !76)
// CHECK:STDOUT: !79 = !{!77}
// CHECK:STDOUT: !80 = !{!"p1 _ZTS1B", !60, i64 0}
// CHECK:STDOUT: !81 = !{!80, !80, i64 0}
// CHECK:STDOUT: !82 = !{!"_ZTS1B", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !83 = !{!82, !7, i64 0}
// CHECK:STDOUT: !84 = !{!82, !7, i64 4}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_both.carbon'
// CHECK:STDOUT: source_filename = "use_both.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A.32 = type { i32, i32 }
// CHECK:STDOUT: %struct.B.33 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_av = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_bv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestA.Main() #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !90
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !91
// CHECK:STDOUT:   %.loc7_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !93
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.b, align 4, !dbg !93
// CHECK:STDOUT:   call void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %a.var), !dbg !90
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestB.Main() #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !102
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !102
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !103
// CHECK:STDOUT:   %.loc12_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !105
// CHECK:STDOUT:   %.loc12_11.2 = load i32, ptr %.loc12_11.1.x, align 4, !dbg !105
// CHECK:STDOUT:   call void @"_CSelfDestruct.d4378aa76ca99c88:core.Destroy.Core"(ptr %b.var), !dbg !102
// CHECK:STDOUT:   ret i32 %.loc12_11.2, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !81
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !81
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d4378aa76ca99c88:core.Destroy.Core"(ptr %self) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.32, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.32, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !64
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.32, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !65
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B.33, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B.33, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !83
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B.33, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !84
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!86}
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
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{!42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{null, !45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !60 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !61 = !{!"p1 _ZTS1A", !60, i64 0}
// CHECK:STDOUT: !62 = !{!61, !61, i64 0}
// CHECK:STDOUT: !63 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !64 = !{!63, !7, i64 0}
// CHECK:STDOUT: !65 = !{!63, !7, i64 4}
// CHECK:STDOUT: !80 = !{!"p1 _ZTS1B", !60, i64 0}
// CHECK:STDOUT: !81 = !{!80, !80, i64 0}
// CHECK:STDOUT: !82 = !{!"_ZTS1B", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !83 = !{!82, !7, i64 0}
// CHECK:STDOUT: !84 = !{!82, !7, i64 4}
// CHECK:STDOUT: !85 = !DIFile(filename: "use_both.carbon", directory: "")
// CHECK:STDOUT: !86 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !85, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "TestA", linkageName: "_CTestA.Main", scope: null, file: !85, line: 5, type: !44, spFlags: DISPFlagDefinition, unit: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 6, column: 3, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 6, column: 18, scope: !87)
// CHECK:STDOUT: !93 = !DILocation(line: 7, column: 10, scope: !87)
// CHECK:STDOUT: !94 = !DILocation(line: 7, column: 3, scope: !87)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core", scope: null, file: !85, line: 6, type: !47, spFlags: DISPFlagDefinition, unit: !86, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !45)
// CHECK:STDOUT: !97 = !DILocation(line: 6, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "TestB", linkageName: "_CTestB.Main", scope: null, file: !85, line: 10, type: !44, spFlags: DISPFlagDefinition, unit: !86)
// CHECK:STDOUT: !102 = !DILocation(line: 11, column: 3, scope: !99)
// CHECK:STDOUT: !103 = !DILocation(line: 11, column: 18, scope: !99)
// CHECK:STDOUT: !105 = !DILocation(line: 12, column: 10, scope: !99)
// CHECK:STDOUT: !106 = !DILocation(line: 12, column: 3, scope: !99)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d4378aa76ca99c88:core.Destroy.Core", scope: null, file: !85, line: 11, type: !47, spFlags: DISPFlagDefinition, unit: !86, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !45)
// CHECK:STDOUT: !109 = !DILocation(line: 11, column: 3, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_lib.carbon'
// CHECK:STDOUT: source_filename = "generic_lib.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A.36 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_av = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %self) #1 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #2 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.36, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.36, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !64
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.36, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !65
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!112}
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
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{null, !45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !60 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !61 = !{!"p1 _ZTS1A", !60, i64 0}
// CHECK:STDOUT: !62 = !{!61, !61, i64 0}
// CHECK:STDOUT: !63 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !64 = !{!63, !7, i64 0}
// CHECK:STDOUT: !65 = !{!63, !7, i64 4}
// CHECK:STDOUT: !111 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !112 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !111, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core", scope: null, file: !111, line: 5, type: !47, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !116)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !45)
// CHECK:STDOUT: !115 = !DILocation(line: 5, column: 3, scope: !113)
// CHECK:STDOUT: !116 = !{!114}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_cpp_imports.carbon'
// CHECK:STDOUT: source_filename = "no_cpp_imports.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A.39 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z6make_av = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCallGeneric.Main() #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeA.call = call i32 @_CMakeA.Main.084a2801e0f5d754({} zeroinitializer), !dbg !122
// CHECK:STDOUT:   ret i32 %MakeA.call, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CMakeA.Main.084a2801e0f5d754({} %x) #0 !dbg !126 {
// CHECK:STDOUT:   %1 = alloca [8 x i8], align 4, !dbg !131
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !131
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %1), !dbg !132
// CHECK:STDOUT:   %b = getelementptr inbounds nuw [8 x i8], ptr %1, i32 0, i32 4, !dbg !134
// CHECK:STDOUT:   %2 = load i32, ptr %b, align 4, !dbg !134
// CHECK:STDOUT:   call void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %1), !dbg !131
// CHECK:STDOUT:   ret i32 %2, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.39, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.39, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !64
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.39, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !65
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!118}
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
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{!42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !60 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !61 = !{!"p1 _ZTS1A", !60, i64 0}
// CHECK:STDOUT: !62 = !{!61, !61, i64 0}
// CHECK:STDOUT: !63 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !64 = !{!63, !7, i64 0}
// CHECK:STDOUT: !65 = !{!63, !7, i64 4}
// CHECK:STDOUT: !111 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !117 = !DIFile(filename: "no_cpp_imports.carbon", directory: "")
// CHECK:STDOUT: !118 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !117, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main", scope: null, file: !117, line: 4, type: !44, spFlags: DISPFlagDefinition, unit: !118)
// CHECK:STDOUT: !122 = !DILocation(line: 5, column: 10, scope: !119)
// CHECK:STDOUT: !123 = !{!42, !45}
// CHECK:STDOUT: !124 = !DISubroutineType(types: !123)
// CHECK:STDOUT: !125 = !DILocation(line: 5, column: 3, scope: !119)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.084a2801e0f5d754", scope: null, file: !111, line: 4, type: !124, spFlags: DISPFlagDefinition, unit: !118, retainedNodes: !136)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !45)
// CHECK:STDOUT: !131 = !DILocation(line: 5, column: 3, scope: !126)
// CHECK:STDOUT: !132 = !DILocation(line: 5, column: 18, scope: !126)
// CHECK:STDOUT: !134 = !DILocation(line: 6, column: 10, scope: !126)
// CHECK:STDOUT: !135 = !DILocation(line: 6, column: 3, scope: !126)
// CHECK:STDOUT: !136 = !{!127}
// CHECK:STDOUT:
