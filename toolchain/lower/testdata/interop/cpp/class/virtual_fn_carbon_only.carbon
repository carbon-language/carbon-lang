// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --clang-arg=-fno-exceptions
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/virtual_fn_carbon_only.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/virtual_fn_carbon_only.carbon

import Cpp inline '''
struct X { virtual void f() {} };
''';

class Z {
  extend base: Cpp.X;
  override fn f(unused self) {}
}

fn F() {
  var _: Z = {.base = Cpp.X.X()};
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'virtual_fn_carbon_only.carbon'
// CHECK:STDOUT: source_filename = "virtual_fn_carbon_only.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1X1fEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1XC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTV1X = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTVN6Carbon4Main1ZE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI1X = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS1X = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTIN6Carbon4Main1ZE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTSN6Carbon4Main1ZE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV1X = linkonce_odr dso_local constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1X, ptr @_ZN1X1fEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTVN6Carbon4Main1ZE = linkonce_odr dso_local constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTIN6Carbon4Main1ZE, ptr @_ZN6Carbon4Main1Z1fEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTI1X = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1X }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS1X = linkonce_odr dso_local constant [3 x i8] c"1X\00", comdat, align 1
// CHECK:STDOUT: @_ZTIN6Carbon4Main1ZE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 16), ptr @_ZTSN6Carbon4Main1ZE, ptr @_ZTI1X }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon4Main1ZE = linkonce_odr dso_local constant [17 x i8] c"N6Carbon4Main1ZE\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1X1fEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cf.Z.Main(ptr %self) #1 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_Cf:thunk:X.Cpp:Z.Main"(ptr %self) #2 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cf.Z.Main(ptr %self), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.Z.Main"(ptr %self) #2 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff24853a526ad6af(ptr %self), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cf__carbon_thunk.Z.Main(ptr %self) #1 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cf.Z.Main(ptr %self), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #1 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { [8 x i8] }, align 8, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !38
// CHECK:STDOUT:   %.loc24_32.1.base = getelementptr inbounds nuw { [8 x i8] }, ptr %_.var, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   call void @_ZN1XC1Ev.carbon_thunk.(ptr %.loc24_32.1.base), !dbg !40
// CHECK:STDOUT:   %.loc24_32.7.base = getelementptr inbounds nuw { [8 x i8] }, ptr %_.var, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   %.loc24_32.8.vptr = getelementptr inbounds nuw [8 x i8], ptr %.loc24_32.7.base, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) (i8, ptr @_ZTVN6Carbon4Main1ZE, i64 16), ptr %.loc24_32.8.vptr, align 8, !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff24853a526ad6af(ptr %_.var), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1XC1Ev.carbon_thunk.(ptr noundef %return) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   call void @_ZN1XC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ff24853a526ad6af(ptr %t) #1 !dbg !43 {
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon4Main1Z1fEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !58
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cf__carbon_thunk.Z.Main(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1XC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #5 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) (i8, ptr @_ZTV1X, i64 16), ptr %this1, align 8, !tbaa !56
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.ff24853a526ad6af, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!13}
// CHECK:STDOUT: !llvm.module.flags = !{!61, !62, !63, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!66}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !12 = !DIFile(filename: "virtual_fn_carbon_only.carbon", directory: "")
// CHECK:STDOUT: !13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !12, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{null, !14}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = !{null}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "f", linkageName: "_Cf.Z.Main", scope: null, file: !12, line: 20, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 3, scope: !19)
// CHECK:STDOUT: !23 = !{!20}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "f", linkageName: "_Cf:thunk:X.Cpp:Z.Main", scope: null, file: !12, line: 20, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !28)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 20, column: 3, scope: !24)
// CHECK:STDOUT: !28 = !{!25}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Z.Main", scope: null, file: !12, line: 18, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !32)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 1, scope: !29)
// CHECK:STDOUT: !32 = !{!30}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "f__carbon_thunk", linkageName: "_Cf__carbon_thunk.Z.Main", scope: null, file: !12, line: 20, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !36)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !14)
// CHECK:STDOUT: !35 = !DILocation(line: 20, column: 3, scope: !33)
// CHECK:STDOUT: !36 = !{!34}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !12, line: 23, type: !18, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !38 = !DILocation(line: 24, column: 3, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 24, column: 14, scope: !37)
// CHECK:STDOUT: !40 = !DILocation(line: 24, column: 23, scope: !37)
// CHECK:STDOUT: !42 = !DILocation(line: 23, column: 1, scope: !37)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ff24853a526ad6af", scope: null, file: !4, line: 32, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !49)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !14)
// CHECK:STDOUT: !48 = !DILocation(line: 32, column: 1, scope: !43)
// CHECK:STDOUT: !49 = !{!44}
// CHECK:STDOUT: !50 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !51 = !{!"omnipotent char", !50, i64 0}
// CHECK:STDOUT: !52 = !{!"any pointer", !51, i64 0}
// CHECK:STDOUT: !53 = !{!"p1 _ZTS1X", !52, i64 0}
// CHECK:STDOUT: !54 = !{!53, !53, i64 0}
// CHECK:STDOUT: !55 = !{!"vtable pointer", !50, i64 0}
// CHECK:STDOUT: !56 = !{!55, !55, i64 0}
// CHECK:STDOUT: !57 = !{!"p1 _ZTSN6Carbon4Main1ZE", !52, i64 0}
// CHECK:STDOUT: !58 = !{!57, !57, i64 0}
// CHECK:STDOUT: !61 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !63 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !64 = !{!"int", !51, i64 0}
// CHECK:STDOUT: !65 = !{!"__libc_errno", !64, i64 0}
// CHECK:STDOUT: !66 = !{!65, !64, i64 0}
// CHECK:STDOUT:
