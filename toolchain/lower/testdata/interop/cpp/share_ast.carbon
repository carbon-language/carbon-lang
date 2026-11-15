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
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
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
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !63
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !64
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
// CHECK:STDOUT: !43 = !DIFile(filename: "use_a.carbon", directory: "")
// CHECK:STDOUT: !44 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !43, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !43, line: 4, type: !47, spFlags: DISPFlagDefinition, unit: !44)
// CHECK:STDOUT: !54 = !DILocation(line: 5, column: 3, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 5, column: 18, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 10, scope: !51)
// CHECK:STDOUT: !58 = !DILocation(line: 6, column: 3, scope: !51)
// CHECK:STDOUT: !59 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !60 = !{!"p1 _ZTS1A", !59, i64 0}
// CHECK:STDOUT: !61 = !{!60, !60, i64 0}
// CHECK:STDOUT: !62 = !{!"_ZTS1A", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !63 = !{!62, !8, i64 0}
// CHECK:STDOUT: !64 = !{!62, !8, i64 4}
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
// CHECK:STDOUT: define i32 @_CTest.Main() #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !70
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !71
// CHECK:STDOUT:   %.loc6_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !73
// CHECK:STDOUT:   %.loc6_11.2 = load i32, ptr %.loc6_11.1.x, align 4, !dbg !73
// CHECK:STDOUT:   ret i32 %.loc6_11.2, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
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
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !78
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !79
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!66}
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
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !59 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !65 = !DIFile(filename: "use_b.carbon", directory: "")
// CHECK:STDOUT: !66 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !65, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Test", linkageName: "_CTest.Main", scope: null, file: !65, line: 4, type: !47, spFlags: DISPFlagDefinition, unit: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 5, column: 3, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 5, column: 18, scope: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 6, column: 10, scope: !67)
// CHECK:STDOUT: !74 = !DILocation(line: 6, column: 3, scope: !67)
// CHECK:STDOUT: !75 = !{!"p1 _ZTS1B", !59, i64 0}
// CHECK:STDOUT: !76 = !{!75, !75, i64 0}
// CHECK:STDOUT: !77 = !{!"_ZTS1B", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !78 = !{!77, !8, i64 0}
// CHECK:STDOUT: !79 = !{!77, !8, i64 4}
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
// CHECK:STDOUT: define i32 @_CTestA.Main() #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [8 x i8], align 4, !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !85
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %a.var), !dbg !86
// CHECK:STDOUT:   %.loc7_11.1.b = getelementptr inbounds nuw [8 x i8], ptr %a.var, i32 0, i32 4, !dbg !88
// CHECK:STDOUT:   %.loc7_11.2 = load i32, ptr %.loc7_11.1.b, align 4, !dbg !88
// CHECK:STDOUT:   ret i32 %.loc7_11.2, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestB.Main() #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.var = alloca [8 x i8], align 4, !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !93
// CHECK:STDOUT:   call void @_Z6make_bv.carbon_thunk.(ptr %b.var), !dbg !94
// CHECK:STDOUT:   %.loc12_11.1.x = getelementptr inbounds nuw [8 x i8], ptr %b.var, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   %.loc12_11.2 = load i32, ptr %.loc12_11.1.x, align 4, !dbg !96
// CHECK:STDOUT:   ret i32 %.loc12_11.2, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_bv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %call = call i64 @_Z6make_bv()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.34, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.34, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !63
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.34, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !64
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_bv() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.B.35, align 4
// CHECK:STDOUT:   %x = getelementptr inbounds nuw %struct.B.35, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 3, ptr %x, align 4, !tbaa !78
// CHECK:STDOUT:   %y = getelementptr inbounds nuw %struct.B.35, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 4, ptr %y, align 4, !tbaa !79
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!81}
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
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !59 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !60 = !{!"p1 _ZTS1A", !59, i64 0}
// CHECK:STDOUT: !61 = !{!60, !60, i64 0}
// CHECK:STDOUT: !62 = !{!"_ZTS1A", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !63 = !{!62, !8, i64 0}
// CHECK:STDOUT: !64 = !{!62, !8, i64 4}
// CHECK:STDOUT: !75 = !{!"p1 _ZTS1B", !59, i64 0}
// CHECK:STDOUT: !76 = !{!75, !75, i64 0}
// CHECK:STDOUT: !77 = !{!"_ZTS1B", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !78 = !{!77, !8, i64 0}
// CHECK:STDOUT: !79 = !{!77, !8, i64 4}
// CHECK:STDOUT: !80 = !DIFile(filename: "use_both.carbon", directory: "")
// CHECK:STDOUT: !81 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !80, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "TestA", linkageName: "_CTestA.Main", scope: null, file: !80, line: 5, type: !47, spFlags: DISPFlagDefinition, unit: !81)
// CHECK:STDOUT: !85 = !DILocation(line: 6, column: 3, scope: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 6, column: 18, scope: !82)
// CHECK:STDOUT: !88 = !DILocation(line: 7, column: 10, scope: !82)
// CHECK:STDOUT: !89 = !DILocation(line: 7, column: 3, scope: !82)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "TestB", linkageName: "_CTestB.Main", scope: null, file: !80, line: 10, type: !47, spFlags: DISPFlagDefinition, unit: !81)
// CHECK:STDOUT: !93 = !DILocation(line: 11, column: 3, scope: !90)
// CHECK:STDOUT: !94 = !DILocation(line: 11, column: 18, scope: !90)
// CHECK:STDOUT: !96 = !DILocation(line: 12, column: 10, scope: !90)
// CHECK:STDOUT: !97 = !DILocation(line: 12, column: 3, scope: !90)
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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !61
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
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !63
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.38, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !64
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!99}
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
// CHECK:STDOUT: !59 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !60 = !{!"p1 _ZTS1A", !59, i64 0}
// CHECK:STDOUT: !61 = !{!60, !60, i64 0}
// CHECK:STDOUT: !62 = !{!"_ZTS1A", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !63 = !{!62, !8, i64 0}
// CHECK:STDOUT: !64 = !{!62, !8, i64 4}
// CHECK:STDOUT: !98 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !99 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !98, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: define i32 @_CCallGeneric.Main() #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeA.call = call i32 @_CMakeA.Main.084a2801e0f5d754({} zeroinitializer), !dbg !105
// CHECK:STDOUT:   ret i32 %MakeA.call, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CMakeA.Main.084a2801e0f5d754({} %x) #0 !dbg !109 {
// CHECK:STDOUT:   %1 = alloca [8 x i8], align 4, !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !114
// CHECK:STDOUT:   call void @_Z6make_av.carbon_thunk.(ptr %1), !dbg !115
// CHECK:STDOUT:   %b = getelementptr inbounds nuw [8 x i8], ptr %1, i32 0, i32 4, !dbg !117
// CHECK:STDOUT:   %2 = load i32, ptr %b, align 4, !dbg !117
// CHECK:STDOUT:   ret i32 %2, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z6make_av.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %call = call i64 @_Z6make_av()
// CHECK:STDOUT:   store i64 %call, ptr %0, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i64 @_Z6make_av() #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.A.41, align 4
// CHECK:STDOUT:   %a = getelementptr inbounds nuw %struct.A.41, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4, !tbaa !63
// CHECK:STDOUT:   %b = getelementptr inbounds nuw %struct.A.41, ptr %retval, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !tbaa !64
// CHECK:STDOUT:   %0 = load i64, ptr %retval, align 4
// CHECK:STDOUT:   ret i64 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!101}
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
// CHECK:STDOUT: !45 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !46 = !{!45}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !59 = !{!"any pointer", !7, i64 0}
// CHECK:STDOUT: !60 = !{!"p1 _ZTS1A", !59, i64 0}
// CHECK:STDOUT: !61 = !{!60, !60, i64 0}
// CHECK:STDOUT: !62 = !{!"_ZTS1A", !8, i64 0, !8, i64 4}
// CHECK:STDOUT: !63 = !{!62, !8, i64 0}
// CHECK:STDOUT: !64 = !{!62, !8, i64 4}
// CHECK:STDOUT: !98 = !DIFile(filename: "generic_lib.carbon", directory: "")
// CHECK:STDOUT: !100 = !DIFile(filename: "no_cpp_imports.carbon", directory: "")
// CHECK:STDOUT: !101 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !100, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "CallGeneric", linkageName: "_CCallGeneric.Main", scope: null, file: !100, line: 4, type: !47, spFlags: DISPFlagDefinition, unit: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 5, column: 10, scope: !102)
// CHECK:STDOUT: !106 = !{!45, !48}
// CHECK:STDOUT: !107 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !108 = !DILocation(line: 5, column: 3, scope: !102)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "MakeA", linkageName: "_CMakeA.Main.084a2801e0f5d754", scope: null, file: !98, line: 4, type: !107, spFlags: DISPFlagDefinition, unit: !101, retainedNodes: !119)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !48)
// CHECK:STDOUT: !114 = !DILocation(line: 5, column: 3, scope: !109)
// CHECK:STDOUT: !115 = !DILocation(line: 5, column: 18, scope: !109)
// CHECK:STDOUT: !117 = !DILocation(line: 6, column: 10, scope: !109)
// CHECK:STDOUT: !118 = !DILocation(line: 6, column: 3, scope: !109)
// CHECK:STDOUT: !119 = !{!110}
// CHECK:STDOUT:
