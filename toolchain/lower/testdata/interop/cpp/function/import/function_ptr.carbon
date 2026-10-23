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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9d0db8a57c58fabc(ptr %fn_ptr.var), !dbg !33
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
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9d0db8a57c58fabc(ptr %t) #0 !dbg !43 {
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !42 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9d0db8a57c58fabc", scope: null, file: !42, line: 35, type: !12, spFlags: DISPFlagDefinition, unit: !26, retainedNodes: !49)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !8)
// CHECK:STDOUT: !48 = !DILocation(line: 35, column: 1, scope: !43)
// CHECK:STDOUT: !49 = !{!44}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z12TakeCallbackPFifE(ptr @_CMyCallback.Main), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyCallback__carbon_thunk.Main(ptr %_, ptr %_1) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5 = load float, ptr %_, align 4, !dbg !58
// CHECK:STDOUT:   %MyCallback.call = call i32 @_CMyCallback.Main(float %.loc5), !dbg !58
// CHECK:STDOUT:   store i32 %MyCallback.call, ptr %_1, align 4, !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z12TakeCallbackPFifE(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!51}
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
// CHECK:STDOUT: !50 = !DIFile(filename: "pass_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !51 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !50, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !50, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 8, column: 3, scope: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 7, column: 1, scope: !52)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !50, line: 5, type: !31, spFlags: DISPFlagDefinition, unit: !51, retainedNodes: !59)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !8)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 2, scope: !55, type: !27)
// CHECK:STDOUT: !58 = !DILocation(line: 5, column: 1, scope: !55)
// CHECK:STDOUT: !59 = !{!56, !57}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !65
// CHECK:STDOUT:   %.loc6_41.3.temp = alloca float, align 4, !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !65
// CHECK:STDOUT:   %GetCallback.call = call ptr @_Z11GetCallbackv(), !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_41.3.temp), !dbg !67
// CHECK:STDOUT:   %__invoke__carbon_thunk.call = call i32 @.carbon_thunk.__(ptr %GetCallback.call, ptr @float.239), !dbg !66
// CHECK:STDOUT:   store i32 %__invoke__carbon_thunk.call, ptr %x.var, align 4, !dbg !65
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.37bc9e7c9d2f29bc(ptr @float.239), !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %x.var), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z11GetCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @.carbon_thunk.__(ptr noundef %this, ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !87
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !90
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !90
// CHECK:STDOUT:   %3 = load float, ptr %2, align 4, !tbaa !91
// CHECK:STDOUT:   %call = call noundef i32 %1(float noundef %3)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.37bc9e7c9d2f29bc(ptr %t) #0 !dbg !72 {
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !79 {
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!61}
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
// CHECK:STDOUT: !42 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !60 = !DIFile(filename: "call_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !60, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !60, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 6, column: 3, scope: !64)
// CHECK:STDOUT: !66 = !DILocation(line: 6, column: 23, scope: !64)
// CHECK:STDOUT: !67 = !DILocation(line: 6, column: 41, scope: !64)
// CHECK:STDOUT: !69 = !{null, !27}
// CHECK:STDOUT: !70 = !DISubroutineType(types: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 5, column: 1, scope: !64)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.37bc9e7c9d2f29bc", scope: null, file: !42, line: 35, type: !12, spFlags: DISPFlagDefinition, unit: !61, retainedNodes: !78)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !8)
// CHECK:STDOUT: !77 = !DILocation(line: 35, column: 1, scope: !72)
// CHECK:STDOUT: !78 = !{!73}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !42, line: 35, type: !70, spFlags: DISPFlagDefinition, unit: !61, retainedNodes: !85)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !27)
// CHECK:STDOUT: !84 = !DILocation(line: 35, column: 1, scope: !79)
// CHECK:STDOUT: !85 = !{!80}
// CHECK:STDOUT: !86 = !{!"any pointer", !21, i64 0}
// CHECK:STDOUT: !87 = !{!86, !86, i64 0}
// CHECK:STDOUT: !88 = !{!"float", !21, i64 0}
// CHECK:STDOUT: !89 = !{!"p1 float", !86, i64 0}
// CHECK:STDOUT: !90 = !{!89, !89, i64 0}
// CHECK:STDOUT: !91 = !{!88, !88, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'relay_complex_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "relay_complex_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GetComplexCallback.call = call ptr @_Z18GetComplexCallbackv(), !dbg !95
// CHECK:STDOUT:   call void @_Z19TakeComplexCallbackPF1CS_E(ptr %GetComplexCallback.call), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18GetComplexCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19TakeComplexCallbackPF1CS_E(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!93}
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
// CHECK:STDOUT: !92 = !DIFile(filename: "relay_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !93 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !92, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !92, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !93)
// CHECK:STDOUT: !95 = !DILocation(line: 6, column: 27, scope: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 6, column: 3, scope: !94)
// CHECK:STDOUT: !97 = !DILocation(line: 5, column: 1, scope: !94)
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn_ptr.var = alloca ptr, align 8, !dbg !105
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %fn_ptr.var), !dbg !105
// CHECK:STDOUT:   store ptr @_CMyCallback.Main, ptr %fn_ptr.var, align 8, !dbg !105
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9d0db8a57c58fabc(ptr %fn_ptr.var), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyCallback__carbon_thunk.Main(ptr %_, ptr %_1) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMyCallback.Main(ptr %_1, ptr %_), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9d0db8a57c58fabc(ptr %t) #0 !dbg !114 {
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!99}
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
// CHECK:STDOUT: !42 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !98 = !DIFile(filename: "convert_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !99 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !98, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !102 = !{null, !8, !8}
// CHECK:STDOUT: !103 = !DISubroutineType(types: !102)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !98, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !99)
// CHECK:STDOUT: !105 = !DILocation(line: 8, column: 3, scope: !104)
// CHECK:STDOUT: !107 = !DILocation(line: 7, column: 1, scope: !104)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !98, line: 8, type: !103, spFlags: DISPFlagDefinition, unit: !99, retainedNodes: !113)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !8)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !108, type: !8)
// CHECK:STDOUT: !112 = !DILocation(line: 5, column: 1, scope: !108)
// CHECK:STDOUT: !113 = !{!109, !110}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9d0db8a57c58fabc", scope: null, file: !42, line: 35, type: !12, spFlags: DISPFlagDefinition, unit: !99, retainedNodes: !120)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !8)
// CHECK:STDOUT: !119 = !DILocation(line: 35, column: 1, scope: !114)
// CHECK:STDOUT: !120 = !{!115}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z19TakeComplexCallbackPF1CS_E(ptr @_CMyCallback.Main), !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMyCallback__carbon_thunk.Main(ptr %_, ptr %_1) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMyCallback.Main(ptr %_1, ptr %_), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19TakeComplexCallbackPF1CS_E(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!122}
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
// CHECK:STDOUT: !102 = !{null, !8, !8}
// CHECK:STDOUT: !103 = !DISubroutineType(types: !102)
// CHECK:STDOUT: !121 = !DIFile(filename: "pass_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !122 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !121, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !121, line: 7, type: !7, spFlags: DISPFlagDefinition, unit: !122)
// CHECK:STDOUT: !124 = !DILocation(line: 8, column: 3, scope: !123)
// CHECK:STDOUT: !125 = !DILocation(line: 7, column: 1, scope: !123)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "MyCallback__carbon_thunk", linkageName: "_CMyCallback__carbon_thunk.Main", scope: null, file: !121, line: 5, type: !103, spFlags: DISPFlagDefinition, unit: !122, retainedNodes: !130)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !8)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 2, scope: !126, type: !8)
// CHECK:STDOUT: !129 = !DILocation(line: 5, column: 1, scope: !126)
// CHECK:STDOUT: !130 = !{!127, !128}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [4 x i8], align 4, !dbg !136
// CHECK:STDOUT:   %.loc6_58.1.temp = alloca [4 x i8], align 4, !dbg !138
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !136
// CHECK:STDOUT:   %GetComplexCallback.call = call ptr @_Z18GetComplexCallbackv(), !dbg !137
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_58.1.temp), !dbg !138
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc6_58.1.temp), !dbg !138
// CHECK:STDOUT:   call void @.carbon_thunk.__(ptr %GetComplexCallback.call, ptr %.loc6_58.1.temp, ptr %x.var), !dbg !137
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2a99575b423907d2(ptr %.loc6_58.1.temp), !dbg !138
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2a99575b423907d2(ptr %x.var), !dbg !136
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18GetComplexCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !149
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !149
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !87
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !149
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !149
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !149
// CHECK:STDOUT:   %2 = load ptr, ptr %this.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %3 = load ptr, ptr %.addr, align 8, !tbaa !149
// CHECK:STDOUT:   call void @_ZN1CC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %agg.tmp, ptr noundef nonnull align 4 dereferenceable(4) %3)
// CHECK:STDOUT:   call void %2(ptr dead_on_unwind writable sret(%class.C) align 4 %1, ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %agg.tmp)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2a99575b423907d2(ptr %t) #0 !dbg !141 {
// CHECK:STDOUT:   ret void, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1CC1Ev(ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1CC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4), ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.2a99575b423907d2, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!132}
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
// CHECK:STDOUT: !42 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !86 = !{!"any pointer", !21, i64 0}
// CHECK:STDOUT: !87 = !{!86, !86, i64 0}
// CHECK:STDOUT: !131 = !DIFile(filename: "call_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !132 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !131, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !131, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !132)
// CHECK:STDOUT: !136 = !DILocation(line: 6, column: 3, scope: !135)
// CHECK:STDOUT: !137 = !DILocation(line: 6, column: 25, scope: !135)
// CHECK:STDOUT: !138 = !DILocation(line: 6, column: 50, scope: !135)
// CHECK:STDOUT: !140 = !DILocation(line: 5, column: 1, scope: !135)
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2a99575b423907d2", scope: null, file: !42, line: 35, type: !12, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !147)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !8)
// CHECK:STDOUT: !146 = !DILocation(line: 35, column: 1, scope: !141)
// CHECK:STDOUT: !147 = !{!142}
// CHECK:STDOUT: !148 = !{!"p1 _ZTS1C", !86, i64 0}
// CHECK:STDOUT: !149 = !{!148, !148, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mention_forward_decl.carbon'
// CHECK:STDOUT: source_filename = "mention_forward_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %callback.var = alloca ptr, align 8, !dbg !154
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %callback.var), !dbg !154
// CHECK:STDOUT:   %ReturnCallback.call = call ptr @_Z14ReturnCallbackv(), !dbg !155
// CHECK:STDOUT:   store ptr %ReturnCallback.call, ptr %callback.var, align 8, !dbg !154
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9d0db8a57c58fabc(ptr %callback.var), !dbg !154
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z14ReturnCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9d0db8a57c58fabc(ptr %t) #0 !dbg !158 {
// CHECK:STDOUT:   ret void, !dbg !163
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!152}
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
// CHECK:STDOUT: !42 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !151 = !DIFile(filename: "mention_forward_decl.carbon", directory: "")
// CHECK:STDOUT: !152 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !151, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !151, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !152)
// CHECK:STDOUT: !154 = !DILocation(line: 6, column: 3, scope: !153)
// CHECK:STDOUT: !155 = !DILocation(line: 6, column: 43, scope: !153)
// CHECK:STDOUT: !157 = !DILocation(line: 5, column: 1, scope: !153)
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9d0db8a57c58fabc", scope: null, file: !42, line: 35, type: !12, spFlags: DISPFlagDefinition, unit: !152, retainedNodes: !164)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !8)
// CHECK:STDOUT: !163 = !DILocation(line: 35, column: 1, scope: !158)
// CHECK:STDOUT: !164 = !{!159}
// CHECK:STDOUT:
