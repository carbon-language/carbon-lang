// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/template.carbon

// --- function_template.h

template<typename T> T identity(T x) { return x; }

class Class {};

// --- call.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_template.h";

fn PassI32(a: i32) -> i32 {
  return Cpp.identity(a);
}

fn PassClass(a: Cpp.Class) -> Cpp.Class {
  return Cpp.identity(a);
}

// --- variadic.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
template<typename ...T>
auto foo(int, T...) -> void;
''';

fn Call1() {
  //@dump-sem-ir-begin
  Cpp.foo(1);
  //@dump-sem-ir-end
}

fn Call2() {
  //@dump-sem-ir-begin
  Cpp.foo(1, 2);
  //@dump-sem-ir-end
}

fn Call3() {
  //@dump-sem-ir-begin
  Cpp.foo(1, 2, 3);
  //@dump-sem-ir-end
}

// --- variadic_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class X {};
template<typename ...T>
auto foo(X, T...) -> void;
''';

fn Call1() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.X);
  //@dump-sem-ir-end
}

fn Call2() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.X, 2);
  //@dump-sem-ir-end
}

fn Call3() {
  //@dump-sem-ir-begin
  Cpp.foo({} as Cpp.X, 2, 3);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.Class = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z8identityI5ClassET_S1_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z8identityIiET_S0_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CPassI32.Main(i32 %a) #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %identity.call = call i32 @_Z8identityIiET_S0_(i32 %a), !dbg !10
// CHECK:STDOUT:   ret i32 %identity.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassClass.Main(ptr sret({}) %return, ptr %a) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8identityI5ClassET_S1_.carbon_thunk(ptr %a, ptr %return), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z8identityI5ClassET_S1_.carbon_thunk(ptr %x, ptr %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.Class, align 1
// CHECK:STDOUT:   %undef.agg.tmp = alloca %class.Class, align 1
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8
// CHECK:STDOUT:   call void @_Z8identityI5ClassET_S1_()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z8identityI5ClassET_S1_() #2 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %class.Class, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local i32 @_Z8identityIiET_S0_(i32 %x) #2 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %x, ptr %x.addr, align 4
// CHECK:STDOUT:   %0 = load i32, ptr %x.addr, align 4
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassI32", linkageName: "_CPassI32.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 10, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "PassClass", linkageName: "_CPassClass.Main", scope: null, file: !6, line: 10, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 11, column: 10, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: ; ModuleID = 'variadic.carbon'
// CHECK:STDOUT: source_filename = "variadic.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall1.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3fooIJEEviDpT_(i32 1), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJEEviDpT_(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall2.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3fooIJiEEviDpT_(i32 1, i32 2), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJiEEviDpT_(i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall3.Main() #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3fooIJiiEEviDpT_(i32 1, i32 2, i32 3), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJiiEEviDpT_(i32, i32, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "variadic.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Call1", linkageName: "_CCall1.Main", scope: null, file: !6, line: 9, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 11, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 1, scope: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Call2", linkageName: "_CCall2.Main", scope: null, file: !6, line: 15, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 17, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 1, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Call3", linkageName: "_CCall3.Main", scope: null, file: !6, line: 21, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DILocation(line: 23, column: 3, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 21, column: 1, scope: !15)
// CHECK:STDOUT: ; ModuleID = 'variadic_thunk.carbon'
// CHECK:STDOUT: source_filename = "variadic_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.X = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @X.val.loc12_12.3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall1.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_12.2.temp = alloca {}, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_12.2.temp), !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc12_12.2.temp, ptr align 1 @X.val.loc12_12.3, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   call void @_Z3fooIJEEv1XDpT_.carbon_thunk(ptr %.loc12_12.2.temp), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall2.Main() #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_12.2.temp = alloca {}, align 8, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_12.2.temp), !dbg !14
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc18_12.2.temp, ptr align 1 @X.val.loc12_12.3, i64 0, i1 false), !dbg !14
// CHECK:STDOUT:   call void @_Z3fooIJiEEv1XDpT_.carbon_thunk(ptr %.loc18_12.2.temp, i32 2), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall3.Main() #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_12.2.temp = alloca {}, align 8, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_12.2.temp), !dbg !18
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc24_12.2.temp, ptr align 1 @X.val.loc12_12.3, i64 0, i1 false), !dbg !18
// CHECK:STDOUT:   call void @_Z3fooIJiiEEv1XDpT_.carbon_thunk(ptr %.loc24_12.2.temp, i32 2, i32 3), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z3fooIJEEv1XDpT_.carbon_thunk(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.X, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z3fooIJEEv1XDpT_()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJEEv1XDpT_() #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z3fooIJiEEv1XDpT_.carbon_thunk(ptr %0, i32 %1) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %class.X, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   %3 = load i32, ptr %.addr1, align 4
// CHECK:STDOUT:   call void @_Z3fooIJiEEv1XDpT_(i32 %3)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJiEEv1XDpT_(i32) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z3fooIJiiEEv1XDpT_.carbon_thunk(ptr %0, i32 %1, i32 %2) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %class.X, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4
// CHECK:STDOUT:   %3 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4
// CHECK:STDOUT:   %5 = load i32, ptr %.addr2, align 4
// CHECK:STDOUT:   call void @_Z3fooIJiiEEv1XDpT_(i32 %4, i32 %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJiiEEv1XDpT_(i32, i32) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "variadic_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Call1", linkageName: "_CCall1.Main", scope: null, file: !6, line: 10, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 11, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 12, column: 3, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 10, column: 1, scope: !7)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Call2", linkageName: "_CCall2.Main", scope: null, file: !6, line: 16, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 11, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 18, column: 3, scope: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 16, column: 1, scope: !13)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Call3", linkageName: "_CCall3.Main", scope: null, file: !6, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 24, column: 11, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 24, column: 3, scope: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 22, column: 1, scope: !17)
