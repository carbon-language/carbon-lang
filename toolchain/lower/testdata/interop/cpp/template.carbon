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

fn PassI32Explicitly(a: i32) -> i32 {
  return Cpp.identity(i32, a);
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call.carbon'
// CHECK:STDOUT: source_filename = "call.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.Class = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z8identityIiET_S0_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z8identityI5ClassET_S1_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CPassI32.Main(i32 %a) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %identity.call = call i32 @_Z8identityIiET_S0_(i32 %a), !dbg !36
// CHECK:STDOUT:   ret i32 %identity.call, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef i32 @_Z8identityIiET_S0_(i32 noundef %x) #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %x, ptr %x.addr, align 4, !tbaa !60
// CHECK:STDOUT:   %0 = load i32, ptr %x.addr, align 4, !tbaa !60
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CPassI32Explicitly.Main(i32 %a) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %identity.call = call i32 @_Z8identityIiET_S0_(i32 %a), !dbg !45
// CHECK:STDOUT:   ret i32 %identity.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassClass.Main(ptr sret({}) %return, ptr %a) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z8identityI5ClassET_S1_.carbon_thunk._(ptr %a, ptr %return), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8identityI5ClassET_S1_.carbon_thunk._(ptr noundef %x, ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.Class, align 1
// CHECK:STDOUT:   %undef.agg.tmp = alloca %class.Class, align 1
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !63
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !63
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !63
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !63
// CHECK:STDOUT:   call void @_Z8identityI5ClassET_S1_()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z8identityI5ClassET_S1_() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %class.Class, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!65, !66, !67, !68, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "call.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!25, !25}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "PassI32", linkageName: "_CPassI32.Main", scope: null, file: !20, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !38)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 7, column: 10, scope: !30)
// CHECK:STDOUT: !37 = !DILocation(line: 7, column: 3, scope: !30)
// CHECK:STDOUT: !38 = !{!31}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "PassI32Explicitly", linkageName: "_CPassI32Explicitly.Main", scope: null, file: !20, line: 10, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !47)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !22)
// CHECK:STDOUT: !45 = !DILocation(line: 11, column: 10, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 11, column: 3, scope: !39)
// CHECK:STDOUT: !47 = !{!40}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "PassClass", linkageName: "_CPassClass.Main", scope: null, file: !20, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !56)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !25)
// CHECK:STDOUT: !54 = !DILocation(line: 15, column: 10, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 15, column: 3, scope: !48)
// CHECK:STDOUT: !56 = !{!49}
// CHECK:STDOUT: !57 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !58 = !{!"omnipotent char", !57, i64 0}
// CHECK:STDOUT: !59 = !{!"int", !58, i64 0}
// CHECK:STDOUT: !60 = !{!59, !59, i64 0}
// CHECK:STDOUT: !61 = !{!"any pointer", !58, i64 0}
// CHECK:STDOUT: !62 = !{!"p1 _ZTS5Class", !61, i64 0}
// CHECK:STDOUT: !63 = !{!62, !62, i64 0}
// CHECK:STDOUT: !65 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !66 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !67 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !68 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !69 = !{!"__libc_errno", !59, i64 0}
// CHECK:STDOUT: !70 = !{!69, !59, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'variadic.carbon'
// CHECK:STDOUT: source_filename = "variadic.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall1.Main() #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3fooIJEEviDpT_(i32 1), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJEEviDpT_(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall2.Main() #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3fooIJiEEviDpT_(i32 1, i32 2), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJiEEviDpT_(i32 noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall3.Main() #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z3fooIJiiEEviDpT_(i32 1, i32 2, i32 3), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJiiEEviDpT_(i32 noundef, i32 noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!72}
// CHECK:STDOUT: !llvm.module.flags = !{!65, !66, !67, !68, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !57 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !58 = !{!"omnipotent char", !57, i64 0}
// CHECK:STDOUT: !59 = !{!"int", !58, i64 0}
// CHECK:STDOUT: !65 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !66 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !67 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !68 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !69 = !{!"__libc_errno", !59, i64 0}
// CHECK:STDOUT: !70 = !{!69, !59, i64 0}
// CHECK:STDOUT: !71 = !DIFile(filename: "variadic.carbon", directory: "")
// CHECK:STDOUT: !72 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !71, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !73 = !{null}
// CHECK:STDOUT: !74 = !DISubroutineType(types: !73)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Call1", linkageName: "_CCall1.Main", scope: null, file: !71, line: 9, type: !74, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !82 = !DILocation(line: 11, column: 3, scope: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 9, column: 1, scope: !81)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "Call2", linkageName: "_CCall2.Main", scope: null, file: !71, line: 15, type: !74, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !85 = !DILocation(line: 17, column: 3, scope: !84)
// CHECK:STDOUT: !86 = !DILocation(line: 15, column: 1, scope: !84)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Call3", linkageName: "_CCall3.Main", scope: null, file: !71, line: 21, type: !74, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !88 = !DILocation(line: 23, column: 3, scope: !87)
// CHECK:STDOUT: !89 = !DILocation(line: 21, column: 1, scope: !87)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'variadic_thunk.carbon'
// CHECK:STDOUT: source_filename = "variadic_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.X = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @X.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @X.val.loc12_14.3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall1.Main() #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_12.2.temp = alloca {}, align 1, !dbg !99
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_12.2.temp), !dbg !99
// CHECK:STDOUT:   call void @_Z3fooIJEEv1XDpT_.carbon_thunk._(ptr @X.val.loc12_14.3), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z3fooIJEEv1XDpT_.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.X, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !111
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !111
// CHECK:STDOUT:   call void @_Z3fooIJEEv1XDpT_()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall2.Main() #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_12.2.temp = alloca {}, align 1, !dbg !103
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_12.2.temp), !dbg !103
// CHECK:STDOUT:   call void @_Z3fooIJiEEv1XDpT_.carbon_thunk.__(ptr @X.val.loc12_14.3, i32 2), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z3fooIJiEEv1XDpT_.carbon_thunk.__(ptr noundef %0, i32 noundef %1) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %class.X, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !111
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !60
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !111
// CHECK:STDOUT:   %3 = load i32, ptr %.addr1, align 4, !tbaa !60
// CHECK:STDOUT:   call void @_Z3fooIJiEEv1XDpT_(i32 noundef %3)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall3.Main() #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_12.2.temp = alloca {}, align 1, !dbg !107
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_12.2.temp), !dbg !107
// CHECK:STDOUT:   call void @_Z3fooIJiiEEv1XDpT_.carbon_thunk.___(ptr @X.val.loc12_14.3, i32 2, i32 3), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z3fooIJiiEEv1XDpT_.carbon_thunk.___(ptr noundef %0, i32 noundef %1, i32 noundef %2) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %class.X, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !111
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !60
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !60
// CHECK:STDOUT:   %3 = load ptr, ptr %.addr, align 8, !tbaa !111
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !60
// CHECK:STDOUT:   %5 = load i32, ptr %.addr2, align 4, !tbaa !60
// CHECK:STDOUT:   call void @_Z3fooIJiiEEv1XDpT_(i32 noundef %4, i32 noundef %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJEEv1XDpT_() #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJiEEv1XDpT_(i32 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3fooIJiiEEv1XDpT_(i32 noundef, i32 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!91}
// CHECK:STDOUT: !llvm.module.flags = !{!65, !66, !67, !68, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !57 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !58 = !{!"omnipotent char", !57, i64 0}
// CHECK:STDOUT: !59 = !{!"int", !58, i64 0}
// CHECK:STDOUT: !60 = !{!59, !59, i64 0}
// CHECK:STDOUT: !61 = !{!"any pointer", !58, i64 0}
// CHECK:STDOUT: !65 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !66 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !67 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !68 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !69 = !{!"__libc_errno", !59, i64 0}
// CHECK:STDOUT: !70 = !{!69, !59, i64 0}
// CHECK:STDOUT: !73 = !{null}
// CHECK:STDOUT: !74 = !DISubroutineType(types: !73)
// CHECK:STDOUT: !90 = !DIFile(filename: "variadic_thunk.carbon", directory: "")
// CHECK:STDOUT: !91 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !90, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Call1", linkageName: "_CCall1.Main", scope: null, file: !90, line: 10, type: !74, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !99 = !DILocation(line: 12, column: 11, scope: !98)
// CHECK:STDOUT: !100 = !DILocation(line: 12, column: 3, scope: !98)
// CHECK:STDOUT: !101 = !DILocation(line: 10, column: 1, scope: !98)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Call2", linkageName: "_CCall2.Main", scope: null, file: !90, line: 16, type: !74, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !103 = !DILocation(line: 18, column: 11, scope: !102)
// CHECK:STDOUT: !104 = !DILocation(line: 18, column: 3, scope: !102)
// CHECK:STDOUT: !105 = !DILocation(line: 16, column: 1, scope: !102)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Call3", linkageName: "_CCall3.Main", scope: null, file: !90, line: 22, type: !74, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !107 = !DILocation(line: 24, column: 11, scope: !106)
// CHECK:STDOUT: !108 = !DILocation(line: 24, column: 3, scope: !106)
// CHECK:STDOUT: !109 = !DILocation(line: 22, column: 1, scope: !106)
// CHECK:STDOUT: !110 = !{!"p1 _ZTS1X", !61, i64 0}
// CHECK:STDOUT: !111 = !{!110, !110, i64 0}
// CHECK:STDOUT:
