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
// CHECK:STDOUT: ; ModuleID = 'call_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "call_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @float.943 = internal constant float 1.000000e+00
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca i32, align 4, !dbg !33
// CHECK:STDOUT:   %.loc6_41.3.temp = alloca float, align 4, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !33
// CHECK:STDOUT:   %GetCallback.call = call ptr @_Z11GetCallbackv(), !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_41.3.temp), !dbg !35
// CHECK:STDOUT:   %__invoke__carbon_thunk.call = call i32 @.carbon_thunk.__(ptr %GetCallback.call, ptr @float.943), !dbg !34
// CHECK:STDOUT:   store i32 %__invoke__carbon_thunk.call, ptr %x.var, align 4, !dbg !33
// CHECK:STDOUT:   call void @"_COp.c73f6eecd880c71a:core.Destroy.Core"(ptr @float.943), !dbg !35
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %x.var), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z11GetCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @.carbon_thunk.__(ptr noundef %this, ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !50
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !50
// CHECK:STDOUT:   %3 = load float, ptr %2, align 4, !tbaa !51
// CHECK:STDOUT:   %call = call noundef i32 %1(float noundef %3)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c73f6eecd880c71a:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
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
// CHECK:STDOUT: !25 = !DIFile(filename: "call_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !26 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !25, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !30 = !{null, !27}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !25, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 6, column: 3, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 6, column: 23, scope: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 6, column: 41, scope: !32)
// CHECK:STDOUT: !37 = !DILocation(line: 5, column: 1, scope: !32)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c73f6eecd880c71a:core.Destroy.Core", scope: null, file: !25, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !26, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !8)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 41, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !25, line: 6, type: !31, spFlags: DISPFlagDefinition, unit: !26, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !27)
// CHECK:STDOUT: !44 = !DILocation(line: 6, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = !{!"any pointer", !21, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !48 = !{!"float", !21, i64 0}
// CHECK:STDOUT: !49 = !{!"p1 float", !46, i64 0}
// CHECK:STDOUT: !50 = !{!49, !49, i64 0}
// CHECK:STDOUT: !51 = !{!48, !48, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'relay_complex_fn_ptr.carbon'
// CHECK:STDOUT: source_filename = "relay_complex_fn_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GetComplexCallback.call = call ptr @_Z18GetComplexCallbackv(), !dbg !55
// CHECK:STDOUT:   call void @_Z19TakeComplexCallbackPF1CS_E(ptr %GetComplexCallback.call), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18GetComplexCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z19TakeComplexCallbackPF1CS_E(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!53}
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
// CHECK:STDOUT: !52 = !DIFile(filename: "relay_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !53 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !52, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !52, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 6, column: 27, scope: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 6, column: 3, scope: !54)
// CHECK:STDOUT: !57 = !DILocation(line: 5, column: 1, scope: !54)
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca [4 x i8], align 4, !dbg !63
// CHECK:STDOUT:   %.loc6_58.1.temp = alloca [4 x i8], align 4, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !63
// CHECK:STDOUT:   %GetComplexCallback.call = call ptr @_Z18GetComplexCallbackv(), !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_58.1.temp), !dbg !65
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc6_58.1.temp), !dbg !65
// CHECK:STDOUT:   call void @.carbon_thunk.__(ptr %GetComplexCallback.call, ptr %.loc6_58.1.temp, ptr %x.var), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18GetComplexCallbackv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !69
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !69
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !69
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !69
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !69
// CHECK:STDOUT:   %2 = load ptr, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %3 = load ptr, ptr %.addr, align 8, !tbaa !69
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!59}
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
// CHECK:STDOUT: !46 = !{!"any pointer", !21, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !58 = !DIFile(filename: "call_complex_fn_ptr.carbon", directory: "")
// CHECK:STDOUT: !59 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !58, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !58, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 6, column: 3, scope: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 6, column: 25, scope: !62)
// CHECK:STDOUT: !65 = !DILocation(line: 6, column: 50, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 5, column: 1, scope: !62)
// CHECK:STDOUT: !68 = !{!"p1 _ZTS1C", !46, i64 0}
// CHECK:STDOUT: !69 = !{!68, !68, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mention_forward_decl.carbon'
// CHECK:STDOUT: source_filename = "mention_forward_decl.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %callback.var = alloca ptr, align 8, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %callback.var), !dbg !74
// CHECK:STDOUT:   %ReturnCallback.call = call ptr @_Z14ReturnCallbackv(), !dbg !75
// CHECK:STDOUT:   store ptr %ReturnCallback.call, ptr %callback.var, align 8, !dbg !74
// CHECK:STDOUT:   ret void, !dbg !77
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!72}
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
// CHECK:STDOUT: !71 = !DIFile(filename: "mention_forward_decl.carbon", directory: "")
// CHECK:STDOUT: !72 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !71, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !71, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !72)
// CHECK:STDOUT: !74 = !DILocation(line: 6, column: 3, scope: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 6, column: 43, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 5, column: 1, scope: !73)
// CHECK:STDOUT:
