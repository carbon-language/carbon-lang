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
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !54
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !55
// CHECK:STDOUT:   %.loc6_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !57
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.b, align 4, !dbg !57
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %a.var), !dbg !54
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %t) #0 !dbg !59 {
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !70
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !71
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!44}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !11, !12}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!10}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !7 = !{!"omnipotent char", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !7, i64 0}
// CHECK:STDOUT: !9 = !{!"__libc_errno", !8, i64 0}
// CHECK:STDOUT: !10 = !{!9, !8, i64 0}
// CHECK:STDOUT: !11 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !12 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !43 = !DIFile(filename: "use_a.carbon", directory: "")
// CHECK:STDOUT: !44 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !43, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !49 = !{null, !48}
// CHECK:STDOUT: !50 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !43, line: 4, type: !47, spFlags: DISPFlagDefinition, unit: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 5, column: 3, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 5, column: 18, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 10, scope: !51)
// CHECK:STDOUT: !58 = !DILocation(line: 6, column: 3, scope: !51)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e885b4b34ded440e", scope: null, file: !35, line: 32, type: !50, spFlags: DISPFlagDefinition, unit: !44, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !48)
// CHECK:STDOUT: !64 = !DILocation(line: 32, column: 1, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT: !66 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !67 = !{!"p1 _ZTS1A", !66, i64 0}
// CHECK:STDOUT: !68 = !{!67, !67, i64 0}
// CHECK:STDOUT: !69 = !{!"_ZTS1A", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !70 = !{!69, !8, i64 0}
// CHECK:STDOUT: !71 = !{!69, !8, i64 4}
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
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !77
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !78
// CHECK:STDOUT:   %.loc6_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !80
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.x, align 4, !dbg !80
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b993e18582fc56e0(ptr %b.var), !dbg !77
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !90
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !90
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b993e18582fc56e0(ptr %t) #0 !dbg !82 {
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !92
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !93
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!73}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !11, !12}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!10}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !7 = !{!"omnipotent char", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !7, i64 0}
// CHECK:STDOUT: !9 = !{!"__libc_errno", !8, i64 0}
// CHECK:STDOUT: !10 = !{!9, !8, i64 0}
// CHECK:STDOUT: !11 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !12 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !49 = !{null, !48}
// CHECK:STDOUT: !50 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !66 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !72 = !DIFile(filename: "use_b.carbon", directory: "")
// CHECK:STDOUT: !73 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !72, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !72, line: 4, type: !47, spFlags: DISPFlagDefinition, unit: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 5, column: 3, scope: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 5, column: 18, scope: !74)
// CHECK:STDOUT: !80 = !DILocation(line: 6, column: 10, scope: !74)
// CHECK:STDOUT: !81 = !DILocation(line: 6, column: 3, scope: !74)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b993e18582fc56e0", scope: null, file: !35, line: 32, type: !50, spFlags: DISPFlagDefinition, unit: !73, retainedNodes: !88)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !48)
// CHECK:STDOUT: !87 = !DILocation(line: 32, column: 1, scope: !82)
// CHECK:STDOUT: !88 = !{!83}
// CHECK:STDOUT: !89 = !{!"p1 _ZTS1B", !66, i64 0}
// CHECK:STDOUT: !90 = !{!89, !89, i64 0}
// CHECK:STDOUT: !91 = !{!"_ZTS1B", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !92 = !{!91, !8, i64 0}
// CHECK:STDOUT: !93 = !{!91, !8, i64 4}
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
// CHECK:STDOUT: define i32 @_CTestA.Main() #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !99
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !99
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !100
// CHECK:STDOUT:   %.loc7_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !102
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.b, align 4, !dbg !102
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %a.var), !dbg !99
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestB.Main() #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !107
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !107
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !108
// CHECK:STDOUT:   %.loc12_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !110
// CHECK:STDOUT:   %.loc12_11.2 = load i32, ptr %.loc12_11.1.x, align 4, !dbg !110
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b993e18582fc56e0(ptr %b.var), !dbg !107
// CHECK:STDOUT:   ret i32 %.loc12_11.2, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !90
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !90
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %t) #0 !dbg !112 {
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b993e18582fc56e0(ptr %t) #0 !dbg !119 {
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.34, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.34, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !70
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.34, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !71
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B.35, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B.35, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !92
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B.35, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !93
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!95}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !11, !12}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!10}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !7 = !{!"omnipotent char", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !7, i64 0}
// CHECK:STDOUT: !9 = !{!"__libc_errno", !8, i64 0}
// CHECK:STDOUT: !10 = !{!9, !8, i64 0}
// CHECK:STDOUT: !11 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !12 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !49 = !{null, !48}
// CHECK:STDOUT: !50 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !66 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !67 = !{!"p1 _ZTS1A", !66, i64 0}
// CHECK:STDOUT: !68 = !{!67, !67, i64 0}
// CHECK:STDOUT: !69 = !{!"_ZTS1A", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !70 = !{!69, !8, i64 0}
// CHECK:STDOUT: !71 = !{!69, !8, i64 4}
// CHECK:STDOUT: !89 = !{!"p1 _ZTS1B", !66, i64 0}
// CHECK:STDOUT: !90 = !{!89, !89, i64 0}
// CHECK:STDOUT: !91 = !{!"_ZTS1B", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !92 = !{!91, !8, i64 0}
// CHECK:STDOUT: !93 = !{!91, !8, i64 4}
// CHECK:STDOUT: !94 = !DIFile(filename: "use_both.carbon", directory: "")
// CHECK:STDOUT: !95 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !94, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "TestA", linkageName: "_CTestA.Main", scope: null, file: !94, line: 5, type: !47, spFlags: DISPFlagDefinition, unit: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 6, column: 3, scope: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 6, column: 18, scope: !96)
// CHECK:STDOUT: !102 = !DILocation(line: 7, column: 10, scope: !96)
// CHECK:STDOUT: !103 = !DILocation(line: 7, column: 3, scope: !96)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "TestB", linkageName: "_CTestB.Main", scope: null, file: !94, line: 10, type: !47, spFlags: DISPFlagDefinition, unit: !95)
// CHECK:STDOUT: !107 = !DILocation(line: 11, column: 3, scope: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 11, column: 18, scope: !104)
// CHECK:STDOUT: !110 = !DILocation(line: 12, column: 10, scope: !104)
// CHECK:STDOUT: !111 = !DILocation(line: 12, column: 3, scope: !104)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e885b4b34ded440e", scope: null, file: !35, line: 32, type: !50, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !118)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !48)
// CHECK:STDOUT: !117 = !DILocation(line: 32, column: 1, scope: !112)
// CHECK:STDOUT: !118 = !{!113}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b993e18582fc56e0", scope: null, file: !35, line: 32, type: !50, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !125)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !48)
// CHECK:STDOUT: !124 = !DILocation(line: 32, column: 1, scope: !119)
// CHECK:STDOUT: !125 = !{!120}
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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
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
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !70
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.38, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !71
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!127}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !11, !12}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!10}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !7 = !{!"omnipotent char", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !7, i64 0}
// CHECK:STDOUT: !9 = !{!"__libc_errno", !8, i64 0}
// CHECK:STDOUT: !10 = !{!9, !8, i64 0}
// CHECK:STDOUT: !11 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !12 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !66 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !67 = !{!"p1 _ZTS1A", !66, i64 0}
// CHECK:STDOUT: !68 = !{!67, !67, i64 0}
// CHECK:STDOUT: !69 = !{!"_ZTS1A", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !70 = !{!69, !8, i64 0}
// CHECK:STDOUT: !71 = !{!69, !8, i64 4}
// CHECK:STDOUT: !126 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !127 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !126, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: define i32 @_CCallGeneric.Main() #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeA.call = call i32 @_CMakeA.Main.084a2801e0f5d754({} zeroinitializer), !dbg !133
// CHECK:STDOUT:   ret i32 %MakeA.call, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CMakeA.Main.084a2801e0f5d754({} %x) #0 !dbg !137 {
// CHECK:STDOUT:   %1 = alloca [8 x i8], align 4, !dbg !142
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !142
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %1), !dbg !143
// CHECK:STDOUT:   %b = getelementptr inbounds nuw [8 x i8], ptr %1, i32 0, i32 4, !dbg !145
// CHECK:STDOUT:   %2 = load i32, ptr %b, align 4, !dbg !145
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %1), !dbg !142
// CHECK:STDOUT:   ret i32 %2, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %t) #0 !dbg !148 {
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.41, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.41, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !70
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.41, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !71
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!129}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !11, !12}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!10}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !7 = !{!"omnipotent char", !6, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !7, i64 0}
// CHECK:STDOUT: !9 = !{!"__libc_errno", !8, i64 0}
// CHECK:STDOUT: !10 = !{!9, !8, i64 0}
// CHECK:STDOUT: !11 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !12 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !49 = !{null, !48}
// CHECK:STDOUT: !50 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !66 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !67 = !{!"p1 _ZTS1A", !66, i64 0}
// CHECK:STDOUT: !68 = !{!67, !67, i64 0}
// CHECK:STDOUT: !69 = !{!"_ZTS1A", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !70 = !{!69, !8, i64 0}
// CHECK:STDOUT: !71 = !{!69, !8, i64 4}
// CHECK:STDOUT: !126 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !128 = !DIFile(filename: "no_cpp_imports.carbon", directory: "")
// CHECK:STDOUT: !129 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !128, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main", scope: null, file: !128, line: 4, type: !47, spFlags: DISPFlagDefinition, unit: !129)
// CHECK:STDOUT: !133 = !DILocation(line: 5, column: 10, scope: !130)
// CHECK:STDOUT: !134 = !{!45, !48}
// CHECK:STDOUT: !135 = !DISubroutineType(types: !134)
// CHECK:STDOUT: !136 = !DILocation(line: 5, column: 3, scope: !130)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.084a2801e0f5d754", scope: null, file: !126, line: 4, type: !135, spFlags: DISPFlagDefinition, unit: !129, retainedNodes: !147)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !48)
// CHECK:STDOUT: !142 = !DILocation(line: 5, column: 3, scope: !137)
// CHECK:STDOUT: !143 = !DILocation(line: 5, column: 18, scope: !137)
// CHECK:STDOUT: !145 = !DILocation(line: 6, column: 10, scope: !137)
// CHECK:STDOUT: !146 = !DILocation(line: 6, column: 3, scope: !137)
// CHECK:STDOUT: !147 = !{!138}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e885b4b34ded440e", scope: null, file: !35, line: 32, type: !50, spFlags: DISPFlagDefinition, unit: !129, retainedNodes: !154)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !48)
// CHECK:STDOUT: !153 = !DILocation(line: 32, column: 1, scope: !148)
// CHECK:STDOUT: !154 = !{!149}
// CHECK:STDOUT:
