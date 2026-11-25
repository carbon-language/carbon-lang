// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/method_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/method_ptr.carbon

// --- method_ptr.h

class C {
 public:
  void Method(int i) const;
};
using MethodPtrType = void(C::*)() const;

auto GetMethodPtr() -> MethodPtrType;
void TakeMethodPtr(MethodPtrType);

template <typename C>
void TakeGenericMethodPtr(void(C::* ptr)() const);

// --- relay_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

fn G() {
  Cpp.TakeMethodPtr(Cpp.GetMethodPtr());
}

// --- convert_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp;

class C {
  fn Method(self);
}

inline Cpp '''c++
using MethodPtrType = void(Carbon::C::*)() const;
''';

fn G() {
  var unused ptr: Cpp.MethodPtrType = C.Method;
}

// --- pass_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

class C {
  fn Method(self);
}

fn F() {
  Cpp.TakeGenericMethodPtr(C.Method);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'relay_method_ptr.carbon'
// CHECK:STDOUT: source_filename = "relay_method_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_38.1.temp = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   %.loc6_38.5.temp = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_38.1.temp), !dbg !12
// CHECK:STDOUT:   call void @_Z12GetMethodPtrv.carbon_thunk.(ptr %.loc6_38.1.temp), !dbg !12
// CHECK:STDOUT:   %.loc6_38.4 = load ptr, ptr %.loc6_38.1.temp, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_38.5.temp), !dbg !12
// CHECK:STDOUT:   store ptr %.loc6_38.4, ptr %.loc6_38.5.temp, align 8, !dbg !12
// CHECK:STDOUT:   call void @_Z13TakeMethodPtrM1CKFvvE.carbon_thunk._(ptr %.loc6_38.5.temp), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z12GetMethodPtrv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %call = call { i64, i64 } @_Z12GetMethodPtrv()
// CHECK:STDOUT:   store { i64, i64 } %call, ptr %0, align 8, !tbaa !19
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13TakeMethodPtrM1CKFvvE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %coerce = alloca { i64, i64 }, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %2 = load { i64, i64 }, ptr %1, align 8, !tbaa !19
// CHECK:STDOUT:   store { i64, i64 } %2, ptr %coerce, align 8, !tbaa !19
// CHECK:STDOUT:   %3 = getelementptr inbounds nuw { i64, i64 }, ptr %coerce, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i64, ptr %3, align 8
// CHECK:STDOUT:   %5 = getelementptr inbounds nuw { i64, i64 }, ptr %coerce, i32 0, i32 1
// CHECK:STDOUT:   %6 = load i64, ptr %5, align 8
// CHECK:STDOUT:   call void @_Z13TakeMethodPtrM1CKFvvE(i64 %4, i64 %6)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare { i64, i64 } @_Z12GetMethodPtrv() #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13TakeMethodPtrM1CKFvvE(i64, i64) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!20, !21, !22, !23, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!26}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "relay_method_ptr.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !4, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DILocation(line: 6, column: 21, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 6, column: 3, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 5, column: 1, scope: !11)
// CHECK:STDOUT: !15 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !16 = !{!"omnipotent char", !15, i64 0}
// CHECK:STDOUT: !17 = !{!"any pointer", !16, i64 0}
// CHECK:STDOUT: !18 = !{!17, !17, i64 0}
// CHECK:STDOUT: !19 = !{!16, !16, i64 0}
// CHECK:STDOUT: !20 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !21 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !22 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !23 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !24 = !{!"int", !16, i64 0}
// CHECK:STDOUT: !25 = !{!"__libc_errno", !24, i64 0}
// CHECK:STDOUT: !26 = !{!25, !24, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert_method_ptr.carbon'
// CHECK:STDOUT: source_filename = "convert_method_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMethod.C.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr.var = alloca ptr, align 8, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr.var), !dbg !30
// CHECK:STDOUT:   store ptr @_CMethod.C.Main, ptr %ptr.var, align 8, !dbg !30
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.C.Main"(ptr %self) #1 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMethod__carbon_thunk.C.Main(ptr noundef nonnull align 1 %self) #2 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMethod.C.Main(ptr %self), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!28}
// CHECK:STDOUT: !llvm.module.flags = !{!20, !21, !22, !23, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!26}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !15 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !16 = !{!"omnipotent char", !15, i64 0}
// CHECK:STDOUT: !20 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !21 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !22 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !23 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !24 = !{!"int", !16, i64 0}
// CHECK:STDOUT: !25 = !{!"__libc_errno", !24, i64 0}
// CHECK:STDOUT: !26 = !{!25, !24, i64 0}
// CHECK:STDOUT: !27 = !DIFile(filename: "convert_method_ptr.carbon", directory: "")
// CHECK:STDOUT: !28 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !27, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !27, line: 13, type: !7, spFlags: DISPFlagDefinition, unit: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 3, scope: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 13, column: 1, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.C.Main", scope: null, file: !27, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !36)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !8)
// CHECK:STDOUT: !35 = !DILocation(line: 5, column: 1, scope: !33)
// CHECK:STDOUT: !36 = !{!34}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Method__carbon_thunk", linkageName: "_CMethod__carbon_thunk.C.Main", scope: null, file: !27, line: 14, type: !10, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !41)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !8)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 3, scope: !37)
// CHECK:STDOUT: !41 = !{!38}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pass_method_ptr.carbon'
// CHECK:STDOUT: source_filename = "pass_method_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @.b35 = internal constant ptr @_CMethod.C.Main
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMethod.C.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_29.3.temp = alloca ptr, align 8, !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_29.3.temp), !dbg !45
// CHECK:STDOUT:   call void @_Z20TakeGenericMethodPtrIN6Carbon4Main1CEEvMT_KFvvE.carbon_thunk._(ptr @.b35), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.C.Main"(ptr %self) #1 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMethod__carbon_thunk.C.Main(ptr noundef nonnull align 1 %self) #2 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMethod.C.Main(ptr %self), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z20TakeGenericMethodPtrIN6Carbon4Main1CEEvMT_KFvvE.carbon_thunk._(ptr noundef %ptr) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr.addr = alloca ptr, align 8
// CHECK:STDOUT:   %coerce = alloca { i64, i64 }, align 8
// CHECK:STDOUT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %0 = load ptr, ptr %ptr.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %1 = load { i64, i64 }, ptr %0, align 8, !tbaa !19
// CHECK:STDOUT:   store { i64, i64 } %1, ptr %coerce, align 8, !tbaa !19
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw { i64, i64 }, ptr %coerce, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i64, ptr %2, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds nuw { i64, i64 }, ptr %coerce, i32 0, i32 1
// CHECK:STDOUT:   %5 = load i64, ptr %4, align 8
// CHECK:STDOUT:   call void @_Z20TakeGenericMethodPtrIN6Carbon4Main1CEEvMT_KFvvE(i64 %3, i64 %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z20TakeGenericMethodPtrIN6Carbon4Main1CEEvMT_KFvvE(i64, i64) #5
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!20, !21, !22, !23, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!26}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !15 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !16 = !{!"omnipotent char", !15, i64 0}
// CHECK:STDOUT: !17 = !{!"any pointer", !16, i64 0}
// CHECK:STDOUT: !18 = !{!17, !17, i64 0}
// CHECK:STDOUT: !19 = !{!16, !16, i64 0}
// CHECK:STDOUT: !20 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !21 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !22 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !23 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !24 = !{!"int", !16, i64 0}
// CHECK:STDOUT: !25 = !{!"__libc_errno", !24, i64 0}
// CHECK:STDOUT: !26 = !{!25, !24, i64 0}
// CHECK:STDOUT: !42 = !DIFile(filename: "pass_method_ptr.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !42, line: 9, type: !7, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 10, column: 28, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 10, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 9, column: 1, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.C.Main", scope: null, file: !42, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !8)
// CHECK:STDOUT: !50 = !DILocation(line: 5, column: 1, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Method__carbon_thunk", linkageName: "_CMethod__carbon_thunk.C.Main", scope: null, file: !42, line: 6, type: !10, spFlags: DISPFlagDefinition, unit: !43, retainedNodes: !55)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !8)
// CHECK:STDOUT: !54 = !DILocation(line: 6, column: 3, scope: !52)
// CHECK:STDOUT: !55 = !{!53}
// CHECK:STDOUT:
