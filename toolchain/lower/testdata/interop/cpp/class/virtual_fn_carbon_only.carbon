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
// CHECK:STDOUT: @_ZTI1X = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1X }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS1X = linkonce_odr dso_local constant [3 x i8] c"1X\00", comdat, align 1
// CHECK:STDOUT: @_ZTIN6Carbon1ZE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTSN6Carbon1ZE, ptr @_ZTI1X }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon1ZE = linkonce_odr dso_local constant [12 x i8] c"N6Carbon1ZE\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1X1fEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #0 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cf.Z.Main(ptr %self) #1 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_Cf:thunk:X.Cpp:Z.Main"(ptr %self) #2 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cf.Z.Main(ptr %self), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.Z.Main"(ptr %self) #2 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3d7b8a41b4c67e71:core.Destroy.Core"(ptr %self) #1 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.de73a19002e400c9:core.Destroy.Core"(ptr %self) #1 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.de73a19002e400c9:core.Destroy.Core"(ptr %self) #1 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.de73a19002e400c9:core.Destroy.Core"(ptr %self), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self) #1 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self) #1 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core"(ptr %self), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cf__carbon_thunk.Z.Main(ptr %self) #1 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cf.Z.Main(ptr %self), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #1 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { [8 x i8] }, align 8, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !52
// CHECK:STDOUT:   %.loc24_32.1.base = getelementptr inbounds nuw { [8 x i8] }, ptr %_.var, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   call void @_ZN1XC1Ev.carbon_thunk.(ptr %.loc24_32.1.base), !dbg !54
// CHECK:STDOUT:   %.loc24_32.7.base = getelementptr inbounds nuw { [8 x i8] }, ptr %_.var, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   %.loc24_32.8.vptr = getelementptr inbounds nuw [8 x i8], ptr %.loc24_32.7.base, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTVN6Carbon1ZE, i32 0, i32 0, i32 2), ptr %.loc24_32.8.vptr, align 8, !dbg !53
// CHECK:STDOUT:   call void @"_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core"(ptr %_.var), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1XC1Ev.carbon_thunk.(ptr noundef %return) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !61
// CHECK:STDOUT:   call void @_ZN1XC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon1Z1fEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !65
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cf__carbon_thunk.Z.Main(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1XC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #5 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !61
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1X, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !63
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!7}
// CHECK:STDOUT: !llvm.module.flags = !{!68, !69, !70, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!73}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "virtual_fn_carbon_only.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !{null}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "f", linkageName: "_Cf.Z.Main", scope: null, file: !6, line: 20, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !13, type: !8)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 3, scope: !13)
// CHECK:STDOUT: !17 = !{!14}
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "f", linkageName: "_Cf:thunk:X.Cpp:Z.Main", scope: null, file: !6, line: 20, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !18, type: !8)
// CHECK:STDOUT: !21 = !DILocation(line: 20, column: 3, scope: !18)
// CHECK:STDOUT: !22 = !{!19}
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Z.Main", scope: null, file: !6, line: 18, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !8)
// CHECK:STDOUT: !25 = !DILocation(line: 18, column: 1, scope: !23)
// CHECK:STDOUT: !26 = !{!24}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3d7b8a41b4c67e71:core.Destroy.Core", scope: null, file: !6, line: 18, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !8)
// CHECK:STDOUT: !29 = !DILocation(line: 18, column: 1, scope: !27)
// CHECK:STDOUT: !30 = !{!28}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.de73a19002e400c9:core.Destroy.Core", scope: null, file: !6, line: 18, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !34)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !8)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 1, scope: !31)
// CHECK:STDOUT: !34 = !{!32}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.de73a19002e400c9:core.Destroy.Core", scope: null, file: !6, line: 18, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !8)
// CHECK:STDOUT: !37 = !DILocation(line: 18, column: 1, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.680f7fa3ef86df7a:core.Destroy.Core", scope: null, file: !6, line: 18, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !8)
// CHECK:STDOUT: !41 = !DILocation(line: 18, column: 1, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.680f7fa3ef86df7a:core.Destroy.Core", scope: null, file: !6, line: 18, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !8)
// CHECK:STDOUT: !45 = !DILocation(line: 18, column: 1, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "f__carbon_thunk", linkageName: "_Cf__carbon_thunk.Z.Main", scope: null, file: !6, line: 20, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !8)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 3, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 23, type: !12, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !52 = !DILocation(line: 24, column: 3, scope: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 24, column: 14, scope: !51)
// CHECK:STDOUT: !54 = !DILocation(line: 24, column: 23, scope: !51)
// CHECK:STDOUT: !56 = !DILocation(line: 23, column: 1, scope: !51)
// CHECK:STDOUT: !57 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !58 = !{!"omnipotent char", !57, i64 0}
// CHECK:STDOUT: !59 = !{!"any pointer", !58, i64 0}
// CHECK:STDOUT: !60 = !{!"p1 _ZTS1X", !59, i64 0}
// CHECK:STDOUT: !61 = !{!60, !60, i64 0}
// CHECK:STDOUT: !62 = !{!"vtable pointer", !57, i64 0}
// CHECK:STDOUT: !63 = !{!62, !62, i64 0}
// CHECK:STDOUT: !64 = !{!"p1 _ZTSN6Carbon1ZE", !59, i64 0}
// CHECK:STDOUT: !65 = !{!64, !64, i64 0}
// CHECK:STDOUT: !68 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !69 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !70 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !71 = !{!"int", !58, i64 0}
// CHECK:STDOUT: !72 = !{!"__libc_errno", !71, i64 0}
// CHECK:STDOUT: !73 = !{!72, !71, i64 0}
// CHECK:STDOUT:
