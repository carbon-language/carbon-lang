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
// CHECK:STDOUT: $_ZTVN6Carbon1ZE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI1X = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS1X = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTIN6Carbon1ZE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTSN6Carbon1ZE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV1X = linkonce_odr dso_local constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1X, ptr @_ZN1X1fEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTVN6Carbon1ZE = linkonce_odr dso_local constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTIN6Carbon1ZE, ptr @_ZN6Carbon1Z1fEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTI1X = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1X }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS1X = linkonce_odr dso_local constant [3 x i8] c"1X\00", comdat, align 1
// CHECK:STDOUT: @_ZTIN6Carbon1ZE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 16), ptr @_ZTSN6Carbon1ZE, ptr @_ZTI1X }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon1ZE = linkonce_odr dso_local constant [12 x i8] c"N6Carbon1ZE\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1X1fEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !73
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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7859376bb6145580(ptr %self), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.6a1352098d778c24.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.de73a19002e400c9:core.SubobjectDestroy.Core"(ptr %self) #1 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_9.1.base = getelementptr inbounds nuw { [8 x i8] }, ptr %self, i32 0, i32 0, !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b72f64bfb935bef9(ptr %.loc18_9.1.base), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.680f7fa3ef86df7a:core.SubobjectDestroy.Core"(ptr %self) #1 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_9.1.base = getelementptr inbounds nuw { [8 x i8] }, ptr %self, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b72f64bfb935bef9(ptr %.loc18_9.1.base), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cf__carbon_thunk.Z.Main(ptr %self) #1 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cf.Z.Main(ptr %self), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #1 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { [8 x i8] }, align 8, !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !50
// CHECK:STDOUT:   %.loc24_32.1.base = getelementptr inbounds nuw { [8 x i8] }, ptr %_.var, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   call void @_ZN1XC1Ev.carbon_thunk.(ptr %.loc24_32.1.base), !dbg !52
// CHECK:STDOUT:   %.loc24_32.7.base = getelementptr inbounds nuw { [8 x i8] }, ptr %_.var, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   %.loc24_32.8.vptr = getelementptr inbounds nuw [8 x i8], ptr %.loc24_32.7.base, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) (i8, ptr @_ZTVN6Carbon1ZE, i64 16), ptr %.loc24_32.8.vptr, align 8, !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7859376bb6145580(ptr %_.var), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1XC1Ev.carbon_thunk.(ptr noundef %return) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !73
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !73
// CHECK:STDOUT:   call void @_ZN1XC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7859376bb6145580(ptr %t) #1 !dbg !55 {
// CHECK:STDOUT:   call void @"_COp.680f7fa3ef86df7a:core.SubobjectDestroy.Core"(ptr %t), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b72f64bfb935bef9(ptr %t) #1 !dbg !62 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.6a1352098d778c24.Core:DestroyImpl.Core"(ptr %t), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon1Z1fEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !77
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cf__carbon_thunk.Z.Main(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1XC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #5 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !73
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) (i8, ptr @_ZTV1X, i64 16), ptr %this1, align 8, !tbaa !75
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7859376bb6145580, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.b72f64bfb935bef9, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!13}
// CHECK:STDOUT: !llvm.module.flags = !{!80, !81, !82, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!85}
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
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.6a1352098d778c24.Core:DestroyImpl.Core", scope: null, file: !12, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !36)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !14)
// CHECK:STDOUT: !35 = !DILocation(line: 0, scope: !33)
// CHECK:STDOUT: !36 = !{!34}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.de73a19002e400c9:core.SubobjectDestroy.Core", scope: null, file: !12, line: 18, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !14)
// CHECK:STDOUT: !39 = !DILocation(line: 18, column: 1, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.680f7fa3ef86df7a:core.SubobjectDestroy.Core", scope: null, file: !12, line: 18, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !14)
// CHECK:STDOUT: !43 = !DILocation(line: 18, column: 1, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "f__carbon_thunk", linkageName: "_Cf__carbon_thunk.Z.Main", scope: null, file: !12, line: 20, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !14)
// CHECK:STDOUT: !47 = !DILocation(line: 20, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !12, line: 23, type: !18, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !50 = !DILocation(line: 24, column: 3, scope: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 24, column: 14, scope: !49)
// CHECK:STDOUT: !52 = !DILocation(line: 24, column: 23, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 23, column: 1, scope: !49)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7859376bb6145580", scope: null, file: !4, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !14)
// CHECK:STDOUT: !59 = !DILocation(line: 28, column: 5, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 1, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b72f64bfb935bef9", scope: null, file: !4, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !14)
// CHECK:STDOUT: !65 = !DILocation(line: 27, column: 5, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 26, column: 1, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !70 = !{!"omnipotent char", !69, i64 0}
// CHECK:STDOUT: !71 = !{!"any pointer", !70, i64 0}
// CHECK:STDOUT: !72 = !{!"p1 _ZTS1X", !71, i64 0}
// CHECK:STDOUT: !73 = !{!72, !72, i64 0}
// CHECK:STDOUT: !74 = !{!"vtable pointer", !69, i64 0}
// CHECK:STDOUT: !75 = !{!74, !74, i64 0}
// CHECK:STDOUT: !76 = !{!"p1 _ZTSN6Carbon1ZE", !71, i64 0}
// CHECK:STDOUT: !77 = !{!76, !76, i64 0}
// CHECK:STDOUT: !80 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !81 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !82 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !83 = !{!"int", !70, i64 0}
// CHECK:STDOUT: !84 = !{!"__libc_errno", !83, i64 0}
// CHECK:STDOUT: !85 = !{!84, !83, i64 0}
// CHECK:STDOUT:
