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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %a.var), !dbg !51
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !67
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !68
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
// CHECK:STDOUT: !36 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ad394d08bf7f4546", scope: null, file: !36, line: 24, type: !47, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !62)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !45)
// CHECK:STDOUT: !61 = !DILocation(line: 24, column: 3, scope: !56)
// CHECK:STDOUT: !62 = !{!57}
// CHECK:STDOUT: !63 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !64 = !{!"p1 _ZTS1A", !63, i64 0}
// CHECK:STDOUT: !65 = !{!64, !64, i64 0}
// CHECK:STDOUT: !66 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !67 = !{!66, !7, i64 0}
// CHECK:STDOUT: !68 = !{!66, !7, i64 4}
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
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !74
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !75
// CHECK:STDOUT:   %.loc6_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !77
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.x, align 4, !dbg !77
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0705a7ee2577eb1e(ptr %b.var), !dbg !74
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0705a7ee2577eb1e(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !89
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !90
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
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
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{!42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{null, !45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !63 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !69 = !DIFile(filename: "use_b.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !69, line: 4, type: !44, spFlags: DISPFlagDefinition, unit: !70)
// CHECK:STDOUT: !74 = !DILocation(line: 5, column: 3, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 5, column: 18, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 6, column: 10, scope: !71)
// CHECK:STDOUT: !78 = !DILocation(line: 6, column: 3, scope: !71)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0705a7ee2577eb1e", scope: null, file: !36, line: 24, type: !47, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !85)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !45)
// CHECK:STDOUT: !84 = !DILocation(line: 24, column: 3, scope: !79)
// CHECK:STDOUT: !85 = !{!80}
// CHECK:STDOUT: !86 = !{!"p1 _ZTS1B", !63, i64 0}
// CHECK:STDOUT: !87 = !{!86, !86, i64 0}
// CHECK:STDOUT: !88 = !{!"_ZTS1B", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !89 = !{!88, !7, i64 0}
// CHECK:STDOUT: !90 = !{!88, !7, i64 4}
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
// CHECK:STDOUT: define i32 @_CTestA.Main() #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !96
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !97
// CHECK:STDOUT:   %.loc7_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !99
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.b, align 4, !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %a.var), !dbg !96
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestB.Main() #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !104
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !104
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !105
// CHECK:STDOUT:   %.loc12_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !107
// CHECK:STDOUT:   %.loc12_11.2 = load i32, ptr %.loc12_11.1.x, align 4, !dbg !107
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.0705a7ee2577eb1e(ptr %b.var), !dbg !104
// CHECK:STDOUT:   ret i32 %.loc12_11.2, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.0705a7ee2577eb1e(ptr %self) #0 !dbg !116 {
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.32, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.32, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !67
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.32, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !68
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B.33, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B.33, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !89
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B.33, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !90
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!92}
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
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{!42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{null, !45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !63 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !64 = !{!"p1 _ZTS1A", !63, i64 0}
// CHECK:STDOUT: !65 = !{!64, !64, i64 0}
// CHECK:STDOUT: !66 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !67 = !{!66, !7, i64 0}
// CHECK:STDOUT: !68 = !{!66, !7, i64 4}
// CHECK:STDOUT: !86 = !{!"p1 _ZTS1B", !63, i64 0}
// CHECK:STDOUT: !87 = !{!86, !86, i64 0}
// CHECK:STDOUT: !88 = !{!"_ZTS1B", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !89 = !{!88, !7, i64 0}
// CHECK:STDOUT: !90 = !{!88, !7, i64 4}
// CHECK:STDOUT: !91 = !DIFile(filename: "use_both.carbon", directory: "")
// CHECK:STDOUT: !92 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !91, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "TestA", linkageName: "_CTestA.Main", scope: null, file: !91, line: 5, type: !44, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 6, column: 3, scope: !93)
// CHECK:STDOUT: !97 = !DILocation(line: 6, column: 18, scope: !93)
// CHECK:STDOUT: !99 = !DILocation(line: 7, column: 10, scope: !93)
// CHECK:STDOUT: !100 = !DILocation(line: 7, column: 3, scope: !93)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "TestB", linkageName: "_CTestB.Main", scope: null, file: !91, line: 10, type: !44, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !104 = !DILocation(line: 11, column: 3, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 11, column: 18, scope: !101)
// CHECK:STDOUT: !107 = !DILocation(line: 12, column: 10, scope: !101)
// CHECK:STDOUT: !108 = !DILocation(line: 12, column: 3, scope: !101)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ad394d08bf7f4546", scope: null, file: !36, line: 24, type: !47, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !115)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !45)
// CHECK:STDOUT: !114 = !DILocation(line: 24, column: 3, scope: !109)
// CHECK:STDOUT: !115 = !{!110}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.0705a7ee2577eb1e", scope: null, file: !36, line: 24, type: !47, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !122)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !45)
// CHECK:STDOUT: !121 = !DILocation(line: 24, column: 3, scope: !116)
// CHECK:STDOUT: !122 = !{!117}
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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.36, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.36, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !67
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.36, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !68
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!124}
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
// CHECK:STDOUT: !63 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !64 = !{!"p1 _ZTS1A", !63, i64 0}
// CHECK:STDOUT: !65 = !{!64, !64, i64 0}
// CHECK:STDOUT: !66 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !67 = !{!66, !7, i64 0}
// CHECK:STDOUT: !68 = !{!66, !7, i64 4}
// CHECK:STDOUT: !123 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !124 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !123, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: define i32 @_CCallGeneric.Main() #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeA.call = call i32 @_CMakeA.Main.084a2801e0f5d754({} zeroinitializer), !dbg !130
// CHECK:STDOUT:   ret i32 %MakeA.call, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CMakeA.Main.084a2801e0f5d754({} %x) #0 !dbg !134 {
// CHECK:STDOUT:   %1 = alloca [8 x i8], align 4, !dbg !139
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !139
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %1), !dbg !140
// CHECK:STDOUT:   %b = getelementptr inbounds nuw [8 x i8], ptr %1, i32 0, i32 4, !dbg !142
// CHECK:STDOUT:   %2 = load i32, ptr %b, align 4, !dbg !142
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %1), !dbg !139
// CHECK:STDOUT:   ret i32 %2, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ad394d08bf7f4546(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.39, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.39, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !67
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.39, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !68
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !36 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !43 = !{!42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !46 = !{null, !45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !63 = !{!"any pointer", !6, i64 0}
// CHECK:STDOUT: !64 = !{!"p1 _ZTS1A", !63, i64 0}
// CHECK:STDOUT: !65 = !{!64, !64, i64 0}
// CHECK:STDOUT: !66 = !{!"_ZTS1A", !7, i64 0, !7, i64 4}
// CHECK:STDOUT: !67 = !{!66, !7, i64 0}
// CHECK:STDOUT: !68 = !{!66, !7, i64 4}
// CHECK:STDOUT: !123 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !125 = !DIFile(filename: "no_cpp_imports.carbon", directory: "")
// CHECK:STDOUT: !126 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !125, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main", scope: null, file: !125, line: 4, type: !44, spFlags: DISPFlagDefinition, unit: !126)
// CHECK:STDOUT: !130 = !DILocation(line: 5, column: 10, scope: !127)
// CHECK:STDOUT: !131 = !{!42, !45}
// CHECK:STDOUT: !132 = !DISubroutineType(types: !131)
// CHECK:STDOUT: !133 = !DILocation(line: 5, column: 3, scope: !127)
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.084a2801e0f5d754", scope: null, file: !123, line: 4, type: !132, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !144)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !45)
// CHECK:STDOUT: !139 = !DILocation(line: 5, column: 3, scope: !134)
// CHECK:STDOUT: !140 = !DILocation(line: 5, column: 18, scope: !134)
// CHECK:STDOUT: !142 = !DILocation(line: 6, column: 10, scope: !134)
// CHECK:STDOUT: !143 = !DILocation(line: 6, column: 3, scope: !134)
// CHECK:STDOUT: !144 = !{!135}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ad394d08bf7f4546", scope: null, file: !36, line: 24, type: !47, spFlags: DISPFlagDefinition, unit: !126, retainedNodes: !151)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !45)
// CHECK:STDOUT: !150 = !DILocation(line: 24, column: 3, scope: !145)
// CHECK:STDOUT: !151 = !{!146}
// CHECK:STDOUT:
