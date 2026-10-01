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
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
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
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn_ptr.var = alloca ptr, align 8, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %fn_ptr.var), !dbg !33
// CHECK:STDOUT:   store ptr @_CMyCallback.Main, ptr %fn_ptr.var, align 8, !dbg !33
// CHECK:STDOUT:   call void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %fn_ptr.var), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyCallback__carbon_thunk.Main(ptr %_, ptr %_1) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8 = load float, ptr %_, align 4, !dbg !39
// CHECK:STDOUT:   %MyCallback.call = call i32 @_CMyCallback.Main(float %.loc8), !dbg !40
// CHECK:STDOUT:   store i32 %MyCallback.call, ptr %_1, align 4, !dbg !39
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!26}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !11 = !{null, !8}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
// CHECK:STDOUT: !25 = !DIFile(filename: "convert_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !26 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !25, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !30 = !{null, !8, !27}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !25, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 8, column: 3, scope: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 7, column: 1, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !25, line: 8, type: !31, spFlags: DISPFlagDefinition, unit: !26, retainedNodes: !41)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !8)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 2, scope: !36, type: !27)
// CHECK:STDOUT: !39 = !DILocation(line: 8, column: 41, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 5, column: 1, scope: !36)
// CHECK:STDOUT: !41 = !{!37, !38}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.659d177172b60c17:core.Destroy.Core", scope: null, file: !25, line: 8, type: !12, spFlags: DISPFlagDefinition, unit: !26, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !8)
// CHECK:STDOUT: !44 = !DILocation(line: 8, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z12TakeCallbackPFifE(ptr @_CMyCallback.Main), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyCallback__carbon_thunk.Main(ptr %_, ptr %_1) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5 = load float, ptr %_, align 4, !dbg !54
// CHECK:STDOUT:   %MyCallback.call = call i32 @_CMyCallback.Main(float %.loc5), !dbg !54
// CHECK:STDOUT:   store i32 %MyCallback.call, ptr %_1, align 4, !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z12TakeCallbackPFifE(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!47}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
// CHECK:STDOUT: !27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !30 = !{null, !8, !27}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !46 = !DIFile(filename: "pass_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !47 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !46, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !46, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 8, column: 3, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 7, column: 1, scope: !48)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !46, line: 5, type: !31, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !55)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !8)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 2, scope: !51, type: !27)
// CHECK:STDOUT: !54 = !DILocation(line: 5, column: 1, scope: !51)
// CHECK:STDOUT: !55 = !{!52, !53}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "call_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @float.943 = internal constant float 1.000000e+00
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !63
// CHECK:STDOUT:   %.loc6_41.3.temp = alloca float, align 4, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !63
// CHECK:STDOUT:   %GetCallback.call = call ptr @_Z11GetCallbackv(), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_41.3.temp), !dbg !65
// CHECK:STDOUT:   %__invoke__carbon_thunk.call = call i32 @.carbon_thunk.__(ptr %GetCallback.call, ptr @float.943), !dbg !64
// CHECK:STDOUT:   store i32 %__invoke__carbon_thunk.call, ptr %x.var, align 4, !dbg !63
// CHECK:STDOUT:   call void @"_CSelfDestruct.c73f6eecd880c71a:core.Destroy.Core"(ptr @float.943), !dbg !65
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %x.var), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z11GetCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @.carbon_thunk.__(ptr noundef %this, ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !77
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !77
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %3 = load float, ptr %2, align 4, !tbaa !81
// CHECK:STDOUT:   %call = call noundef i32 %1(float noundef %3)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.c73f6eecd880c71a:core.Destroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !74
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!57}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !11 = !{null, !8}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
// CHECK:STDOUT: !27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !56 = !DIFile(filename: "call_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !57 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !56, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !60 = !{null, !27}
// CHECK:STDOUT: !61 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !56, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 6, column: 3, scope: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 6, column: 23, scope: !62)
// CHECK:STDOUT: !65 = !DILocation(line: 6, column: 41, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 5, column: 1, scope: !62)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.c73f6eecd880c71a:core.Destroy.Core", scope: null, file: !56, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !57, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !8)
// CHECK:STDOUT: !70 = !DILocation(line: 6, column: 41, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !56, line: 6, type: !61, spFlags: DISPFlagDefinition, unit: !57, retainedNodes: !75)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !27)
// CHECK:STDOUT: !74 = !DILocation(line: 6, column: 3, scope: !72)
// CHECK:STDOUT: !75 = !{!73}
// CHECK:STDOUT: !76 = !{!"any pointer", !21, i64 0}
// CHECK:STDOUT: !77 = !{!76, !76, i64 0}
// CHECK:STDOUT: !78 = !{!"float", !21, i64 0}
// CHECK:STDOUT: !79 = !{!"p1 float", !76, i64 0}
// CHECK:STDOUT: !80 = !{!79, !79, i64 0}
// CHECK:STDOUT: !81 = !{!78, !78, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'relay_complex_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "relay_complex_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GetComplexCallback.call = call ptr @_Z18GetComplexCallbackv(), !dbg !85
// CHECK:STDOUT:   call void @_Z19TakeComplexCallbackPF1CS_E(ptr %GetComplexCallback.call), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18GetComplexCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19TakeComplexCallbackPF1CS_E(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!83}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
// CHECK:STDOUT: !82 = !DIFile(filename: "relay_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !82, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !82, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !83)
// CHECK:STDOUT: !85 = !DILocation(line: 6, column: 27, scope: !84)
// CHECK:STDOUT: !86 = !DILocation(line: 6, column: 3, scope: !84)
// CHECK:STDOUT: !87 = !DILocation(line: 5, column: 1, scope: !84)
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn_ptr.var = alloca ptr, align 8, !dbg !95
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %fn_ptr.var), !dbg !95
// CHECK:STDOUT:   store ptr @_CMyCallback.Main, ptr %fn_ptr.var, align 8, !dbg !95
// CHECK:STDOUT:   call void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %fn_ptr.var), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyCallback__carbon_thunk.Main(ptr %_, ptr %_1) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMyCallback.Main(ptr %_1, ptr %_), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!89}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !11 = !{null, !8}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
// CHECK:STDOUT: !88 = !DIFile(filename: "convert_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !89 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !88, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !92 = !{null, !8, !8}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !88, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !89)
// CHECK:STDOUT: !95 = !DILocation(line: 8, column: 3, scope: !94)
// CHECK:STDOUT: !97 = !DILocation(line: 7, column: 1, scope: !94)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !88, line: 8, type: !93, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !103)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !8)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !98, type: !8)
// CHECK:STDOUT: !102 = !DILocation(line: 5, column: 1, scope: !98)
// CHECK:STDOUT: !103 = !{!99, !100}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.659d177172b60c17:core.Destroy.Core", scope: null, file: !88, line: 8, type: !12, spFlags: DISPFlagDefinition, unit: !89, retainedNodes: !107)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !8)
// CHECK:STDOUT: !106 = !DILocation(line: 8, column: 3, scope: !104)
// CHECK:STDOUT: !107 = !{!105}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z19TakeComplexCallbackPF1CS_E(ptr @_CMyCallback.Main), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyCallback__carbon_thunk.Main(ptr %_, ptr %_1) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMyCallback.Main(ptr %_1, ptr %_), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19TakeComplexCallbackPF1CS_E(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!109}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
// CHECK:STDOUT: !92 = !{null, !8, !8}
// CHECK:STDOUT: !93 = !DISubroutineType(types: !92)
// CHECK:STDOUT: !108 = !DIFile(filename: "pass_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !109 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !108, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !108, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !109)
// CHECK:STDOUT: !111 = !DILocation(line: 8, column: 3, scope: !110)
// CHECK:STDOUT: !112 = !DILocation(line: 7, column: 1, scope: !110)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !108, line: 5, type: !93, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !117)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !8)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 2, scope: !113, type: !8)
// CHECK:STDOUT: !116 = !DILocation(line: 5, column: 1, scope: !113)
// CHECK:STDOUT: !117 = !{!114, !115}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [4 x i8], align 4, !dbg !123
// CHECK:STDOUT:   %.loc6_58.1.temp = alloca [4 x i8], align 4, !dbg !125
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !123
// CHECK:STDOUT:   %GetComplexCallback.call = call ptr @_Z18GetComplexCallbackv(), !dbg !124
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_58.1.temp), !dbg !125
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc6_58.1.temp), !dbg !125
// CHECK:STDOUT:   call void @.carbon_thunk.__(ptr %GetComplexCallback.call, ptr %.loc6_58.1.temp, ptr %x.var), !dbg !124
// CHECK:STDOUT:   call void @"_CSelfDestruct.417eccf5eb8addbd:core.Destroy.Core"(ptr %.loc6_58.1.temp), !dbg !125
// CHECK:STDOUT:   call void @"_CSelfDestruct.417eccf5eb8addbd:core.Destroy.Core"(ptr %x.var), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18GetComplexCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !133
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !133
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !77
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !133
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !133
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !133
// CHECK:STDOUT:   %2 = load ptr, ptr %this.addr, align 8, !tbaa !77
// CHECK:STDOUT:   %3 = load ptr, ptr %.addr, align 8, !tbaa !133
// CHECK:STDOUT:   call void @_ZN1CC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %agg.tmp, ptr noundef nonnull align 4 dereferenceable(4) %3)
// CHECK:STDOUT:   call void %2(ptr dead_on_unwind writable sret(%class.C) align 4 %1, ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %agg.tmp)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.417eccf5eb8addbd:core.Destroy.Core"(ptr %self) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !130
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
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.417eccf5eb8addbd:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!119}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !11 = !{null, !8}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
// CHECK:STDOUT: !76 = !{!"any pointer", !21, i64 0}
// CHECK:STDOUT: !77 = !{!76, !76, i64 0}
// CHECK:STDOUT: !118 = !DIFile(filename: "call_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !119 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !118, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !118, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !119)
// CHECK:STDOUT: !123 = !DILocation(line: 6, column: 3, scope: !122)
// CHECK:STDOUT: !124 = !DILocation(line: 6, column: 25, scope: !122)
// CHECK:STDOUT: !125 = !DILocation(line: 6, column: 50, scope: !122)
// CHECK:STDOUT: !127 = !DILocation(line: 5, column: 1, scope: !122)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.417eccf5eb8addbd:core.Destroy.Core", scope: null, file: !118, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !119, retainedNodes: !131)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !8)
// CHECK:STDOUT: !130 = !DILocation(line: 6, column: 50, scope: !128)
// CHECK:STDOUT: !131 = !{!129}
// CHECK:STDOUT: !132 = !{!"p1 _ZTS1C", !76, i64 0}
// CHECK:STDOUT: !133 = !{!132, !132, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mention_forward_decl.carbon'
// CHECK:STDOUT: source_filename = "mention_forward_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %callback.var = alloca ptr, align 8, !dbg !138
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %callback.var), !dbg !138
// CHECK:STDOUT:   %ReturnCallback.call = call ptr @_Z14ReturnCallbackv(), !dbg !139
// CHECK:STDOUT:   store ptr %ReturnCallback.call, ptr %callback.var, align 8, !dbg !138
// CHECK:STDOUT:   call void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %callback.var), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z14ReturnCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %self) #0 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!136}
// CHECK:STDOUT: !llvm.module.flags = !{!17, !18, !19, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!24}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !11 = !{null, !8}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !17 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !18 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !19 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !20 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !21 = !{!"omnipotent char", !20, i64 0}
// CHECK:STDOUT: !22 = !{!"int", !21, i64 0}
// CHECK:STDOUT: !23 = !{!"__libc_errno", !22, i64 0}
// CHECK:STDOUT: !24 = !{!23, !22, i64 0}
// CHECK:STDOUT: !135 = !DIFile(filename: "mention_forward_decl.carbon", directory: "")
// CHECK:STDOUT: !136 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !135, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !135, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !136)
// CHECK:STDOUT: !138 = !DILocation(line: 6, column: 3, scope: !137)
// CHECK:STDOUT: !139 = !DILocation(line: 6, column: 43, scope: !137)
// CHECK:STDOUT: !141 = !DILocation(line: 5, column: 1, scope: !137)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.659d177172b60c17:core.Destroy.Core", scope: null, file: !135, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !136, retainedNodes: !145)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !8)
// CHECK:STDOUT: !144 = !DILocation(line: 6, column: 3, scope: !142)
// CHECK:STDOUT: !145 = !{!143}
// CHECK:STDOUT:
