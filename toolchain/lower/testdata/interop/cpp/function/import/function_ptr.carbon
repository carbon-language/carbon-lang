// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/function_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/function_ptr.carbon

// --- simple_callback.h

using CallbackType = int (*)(float);

void TakeCallback(CallbackType callback);

auto GetCallback() -> CallbackType;

// --- relay_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

fn F() {
  Cpp.TakeCallback(Cpp.GetCallback());
}

// --- convert_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

fn MyCallback(arg: f32) -> i32;

fn F() {
  var unused fn_ptr: Cpp.CallbackType = MyCallback;
}

// --- pass_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

fn MyCallback(arg: f32) -> i32;

fn F() {
  Cpp.TakeCallback(MyCallback);
}

// --- call_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

fn F() {
  var unused x: i32 = Cpp.GetCallback()(1.0);
}

// --- complex_callback.h

class C {
 public:
  C();
  C(const C&);
  C& operator=(const C&);

 private:
  int x;
};

using CallbackType = C (*)(C);

void TakeComplexCallback(CallbackType callback);

auto GetComplexCallback() -> CallbackType;

// --- relay_complex_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "complex_callback.h";

fn F() {
  Cpp.TakeComplexCallback(Cpp.GetComplexCallback());
}

// --- convert_complex_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "complex_callback.h";

fn MyCallback(arg: Cpp.C) -> Cpp.C;

fn F() {
  var unused fn_ptr: Cpp.CallbackType = MyCallback;
}

// --- pass_complex_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "complex_callback.h";

fn MyCallback(arg: Cpp.C) -> Cpp.C;

fn F() {
  Cpp.TakeComplexCallback(MyCallback);
}

// --- call_complex_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "complex_callback.h";

fn F() {
  var unused x: Cpp.C = Cpp.GetComplexCallback()(Cpp.C.C());
}

// --- forward_decl.h

class C;

using CallbackType = C (*)();

auto ReturnCallback() -> CallbackType;

// --- mention_forward_decl.carbon
library "[[@TEST_NAME]]";

import Cpp library "forward_decl.h";

fn F() {
  var unused callback: Cpp.CallbackType = Cpp.ReturnCallback();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'relay_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "relay_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GetCallback.call = call ptr @_Z11GetCallbackv(), !dbg !14
// CHECK:STDOUT:   call void @_Z12TakeCallbackPFifE(ptr %GetCallback.call), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z11GetCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z12TakeCallbackPFifE(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "relay_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !4, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !14 = !DILocation(line: 6, column: 20, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 6, column: 3, scope: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 5, column: 1, scope: !13)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "convert_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMyCallback.Main(float)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn_ptr.var = alloca ptr, align 8, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %fn_ptr.var), !dbg !34
// CHECK:STDOUT:   store ptr @_CMyCallback.Main, ptr %fn_ptr.var, align 8, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMyCallback__carbon_thunk.Main(ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #1 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8 = load float, ptr %_, align 4, !dbg !40
// CHECK:STDOUT:   %MyCallback.call = call i32 @_CMyCallback.Main(float %.loc8), !dbg !41
// CHECK:STDOUT:   store i32 %MyCallback.call, ptr %_1, align 4, !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!27}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT: !26 = !DIFile(filename: "convert_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !27 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !26, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{null, !8, !28}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !26, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !27)
// CHECK:STDOUT: !34 = !DILocation(line: 8, column: 3, scope: !33)
// CHECK:STDOUT: !36 = !DILocation(line: 7, column: 1, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !26, line: 8, type: !32, spFlags: DISPFlagDefinition, unit: !27, retainedNodes: !42)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !8)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !37, type: !28)
// CHECK:STDOUT: !40 = !DILocation(line: 8, column: 41, scope: !37)
// CHECK:STDOUT: !41 = !DILocation(line: 5, column: 1, scope: !37)
// CHECK:STDOUT: !42 = !{!38, !39}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pass_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "pass_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMyCallback.Main(float)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z12TakeCallbackPFifE(ptr @_CMyCallback.Main), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMyCallback__carbon_thunk.Main(ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #1 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5 = load float, ptr %_, align 4, !dbg !51
// CHECK:STDOUT:   %MyCallback.call = call i32 @_CMyCallback.Main(float %.loc5), !dbg !51
// CHECK:STDOUT:   store i32 %MyCallback.call, ptr %_1, align 4, !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z12TakeCallbackPFifE(ptr noundef) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!44}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT: !28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{null, !8, !28}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !43 = !DIFile(filename: "pass_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !44 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !43, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !43, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 8, column: 3, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 7, column: 1, scope: !45)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !43, line: 5, type: !32, spFlags: DISPFlagDefinition, unit: !44, retainedNodes: !52)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !8)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !48, type: !28)
// CHECK:STDOUT: !51 = !DILocation(line: 5, column: 1, scope: !48)
// CHECK:STDOUT: !52 = !{!49, !50}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "call_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @float.239 = internal constant float 1.000000e+00
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !58
// CHECK:STDOUT:   %.loc6_41.3.temp = alloca float, align 4, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !58
// CHECK:STDOUT:   %GetCallback.call = call ptr @_Z11GetCallbackv(), !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_41.3.temp), !dbg !60
// CHECK:STDOUT:   %__invoke__carbon_thunk.call = call i32 @.carbon_thunk.__(ptr %GetCallback.call, ptr @float.239), !dbg !59
// CHECK:STDOUT:   store i32 %__invoke__carbon_thunk.call, ptr %x.var, align 4, !dbg !58
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z11GetCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @.carbon_thunk.__(ptr noundef %this, ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !64
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !64
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !67
// CHECK:STDOUT:   %3 = load float, ptr %2, align 4, !tbaa !68
// CHECK:STDOUT:   %call = call noundef i32 %1(float noundef %3)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!54}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT: !53 = !DIFile(filename: "call_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !54 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !53, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !53, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 6, column: 3, scope: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 6, column: 23, scope: !57)
// CHECK:STDOUT: !60 = !DILocation(line: 6, column: 41, scope: !57)
// CHECK:STDOUT: !62 = !DILocation(line: 5, column: 1, scope: !57)
// CHECK:STDOUT: !63 = !{!"any pointer", !22, i64 0}
// CHECK:STDOUT: !64 = !{!63, !63, i64 0}
// CHECK:STDOUT: !65 = !{!"float", !22, i64 0}
// CHECK:STDOUT: !66 = !{!"p1 float", !63, i64 0}
// CHECK:STDOUT: !67 = !{!66, !66, i64 0}
// CHECK:STDOUT: !68 = !{!65, !65, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'relay_complex_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "relay_complex_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GetComplexCallback.call = call ptr @_Z18GetComplexCallbackv(), !dbg !72
// CHECK:STDOUT:   call void @_Z19TakeComplexCallbackPF1CS_E(ptr %GetComplexCallback.call), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18GetComplexCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19TakeComplexCallbackPF1CS_E(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT: !69 = !DIFile(filename: "relay_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !69, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !70)
// CHECK:STDOUT: !72 = !DILocation(line: 6, column: 27, scope: !71)
// CHECK:STDOUT: !73 = !DILocation(line: 6, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !DILocation(line: 5, column: 1, scope: !71)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert_complex_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "convert_complex_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMyCallback.Main(ptr sret([4 x i8]), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn_ptr.var = alloca ptr, align 8, !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %fn_ptr.var), !dbg !82
// CHECK:STDOUT:   store ptr @_CMyCallback.Main, ptr %fn_ptr.var, align 8, !dbg !82
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMyCallback__carbon_thunk.Main(ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #1 !dbg !85 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMyCallback.Main(ptr %_1, ptr %_), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!76}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT: !75 = !DIFile(filename: "convert_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !76 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !75, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !79 = !{null, !8, !8}
// CHECK:STDOUT: !80 = !DISubroutineType(types: !79)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !75, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !76)
// CHECK:STDOUT: !82 = !DILocation(line: 8, column: 3, scope: !81)
// CHECK:STDOUT: !84 = !DILocation(line: 7, column: 1, scope: !81)
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !75, line: 8, type: !80, spFlags: DISPFlagDefinition, unit: !76, retainedNodes: !90)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !8)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 2, scope: !85, type: !8)
// CHECK:STDOUT: !89 = !DILocation(line: 5, column: 1, scope: !85)
// CHECK:STDOUT: !90 = !{!86, !87}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pass_complex_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "pass_complex_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMyCallback.Main(ptr sret([4 x i8]), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z19TakeComplexCallbackPF1CS_E(ptr @_CMyCallback.Main), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMyCallback__carbon_thunk.Main(ptr noundef nonnull align 4 dereferenceable(4) %_, ptr noundef nonnull align 4 dereferenceable(4) %_1) #1 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMyCallback.Main(ptr %_1, ptr %_), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19TakeComplexCallbackPF1CS_E(ptr noundef) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!92}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT: !79 = !{null, !8, !8}
// CHECK:STDOUT: !80 = !DISubroutineType(types: !79)
// CHECK:STDOUT: !91 = !DIFile(filename: "pass_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !92 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !91, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !91, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !94 = !DILocation(line: 8, column: 3, scope: !93)
// CHECK:STDOUT: !95 = !DILocation(line: 7, column: 1, scope: !93)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !91, line: 5, type: !80, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !100)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !8)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 2, scope: !96, type: !8)
// CHECK:STDOUT: !99 = !DILocation(line: 5, column: 1, scope: !96)
// CHECK:STDOUT: !100 = !{!97, !98}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_complex_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "call_complex_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [4 x i8], align 4, !dbg !106
// CHECK:STDOUT:   %.loc6_58.1.temp = alloca [4 x i8], align 4, !dbg !108
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !106
// CHECK:STDOUT:   %GetComplexCallback.call = call ptr @_Z18GetComplexCallbackv(), !dbg !107
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_58.1.temp), !dbg !108
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc6_58.1.temp), !dbg !108
// CHECK:STDOUT:   call void @.carbon_thunk.__(ptr %GetComplexCallback.call, ptr %.loc6_58.1.temp, ptr %x.var), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18GetComplexCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !112
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !112
// CHECK:STDOUT:   call void @_ZN1CC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @.carbon_thunk.__(ptr noundef %this, ptr noundef %0, ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.C, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !64
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !112
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !112
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !112
// CHECK:STDOUT:   %2 = load ptr, ptr %this.addr, align 8, !tbaa !64
// CHECK:STDOUT:   %3 = load ptr, ptr %.addr, align 8, !tbaa !112
// CHECK:STDOUT:   call void @_ZN1CC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %agg.tmp, ptr noundef nonnull align 4 dereferenceable(4) %3)
// CHECK:STDOUT:   call void %2(ptr dead_on_unwind writable sret(%class.C) align 4 %1, ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %agg.tmp)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1CC1Ev(ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1CC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4), ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!102}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT: !63 = !{!"any pointer", !22, i64 0}
// CHECK:STDOUT: !64 = !{!63, !63, i64 0}
// CHECK:STDOUT: !101 = !DIFile(filename: "call_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !102 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !101, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !101, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !102)
// CHECK:STDOUT: !106 = !DILocation(line: 6, column: 3, scope: !105)
// CHECK:STDOUT: !107 = !DILocation(line: 6, column: 25, scope: !105)
// CHECK:STDOUT: !108 = !DILocation(line: 6, column: 50, scope: !105)
// CHECK:STDOUT: !110 = !DILocation(line: 5, column: 1, scope: !105)
// CHECK:STDOUT: !111 = !{!"p1 _ZTS1C", !63, i64 0}
// CHECK:STDOUT: !112 = !{!111, !111, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mention_forward_decl.carbon'
// CHECK:STDOUT: source_filename = "mention_forward_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %callback.var = alloca ptr, align 8, !dbg !117
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %callback.var), !dbg !117
// CHECK:STDOUT:   %ReturnCallback.call = call ptr @_Z14ReturnCallbackv(), !dbg !118
// CHECK:STDOUT:   store ptr %ReturnCallback.call, ptr %callback.var, align 8, !dbg !117
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z14ReturnCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!115}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !20, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!25}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !17 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !18 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !20 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !21 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !22 = !{!"omnipotent char", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"int", !22, i64 0}
// CHECK:STDOUT: !24 = !{!"__libc_errno", !23, i64 0}
// CHECK:STDOUT: !25 = !{!24, !23, i64 0}
// CHECK:STDOUT: !114 = !DIFile(filename: "mention_forward_decl.carbon", directory: "")
// CHECK:STDOUT: !115 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !114, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !114, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !115)
// CHECK:STDOUT: !117 = !DILocation(line: 6, column: 3, scope: !116)
// CHECK:STDOUT: !118 = !DILocation(line: 6, column: 43, scope: !116)
// CHECK:STDOUT: !120 = !DILocation(line: 5, column: 1, scope: !116)
// CHECK:STDOUT:
