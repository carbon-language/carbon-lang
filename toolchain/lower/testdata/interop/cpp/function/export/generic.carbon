// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/generic.carbon

// --- generic_type_impls_interface.carbon
library "[[@TEST_NAME]]";
import Cpp;

interface I {
  fn Doit(self);
}

class A {
  impl as I {
    fn Doit(unused self) {}
  }
}
class B {
  impl as I {
    fn Doit(unused self) {}
  }
}

fn F[T:! I](t: T) {
  t.Doit();
}

inline Cpp '''
void G() {
  Carbon::A a;
  Carbon::B b;
  Carbon::F(a);
  Carbon::F(b);
}
''';

fn Run() {
  Cpp.G();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_type_impls_interface.carbon'
// CHECK:STDOUT: source_filename = "generic_type_impls_interface.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::A" = type {}
// CHECK:STDOUT: %"class.Carbon::B" = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon1AD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon1BD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1Gv() #0 personality ptr @__gxx_personality_v0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a = alloca %"class.Carbon::A", align 1
// CHECK:STDOUT:   %b = alloca %"class.Carbon::B", align 1
// CHECK:STDOUT:   %agg.tmp = alloca %"class.Carbon::A", align 1
// CHECK:STDOUT:   %exn.slot = alloca ptr, align 8
// CHECK:STDOUT:   %ehselector.slot = alloca i32, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %"class.Carbon::B", align 1
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a) #4
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b) #4
// CHECK:STDOUT:   invoke void @_ZN6CarbonL1FENS_1AE()
// CHECK:STDOUT:           to label %invoke.cont unwind label %lpad
// CHECK:STDOUT:
// CHECK:STDOUT: invoke.cont:                                      ; preds = %entry
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %agg.tmp) #4
// CHECK:STDOUT:   invoke void @_ZN6CarbonL1FENS_1BE()
// CHECK:STDOUT:           to label %invoke.cont3 unwind label %lpad2
// CHECK:STDOUT:
// CHECK:STDOUT: invoke.cont3:                                     ; preds = %invoke.cont
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %agg.tmp1) #4
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %b) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #4
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %a) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT:
// CHECK:STDOUT: lpad:                                             ; preds = %entry
// CHECK:STDOUT:   %0 = landingpad { ptr, i32 }
// CHECK:STDOUT:           cleanup
// CHECK:STDOUT:   %1 = extractvalue { ptr, i32 } %0, 0
// CHECK:STDOUT:   store ptr %1, ptr %exn.slot, align 8
// CHECK:STDOUT:   %2 = extractvalue { ptr, i32 } %0, 1
// CHECK:STDOUT:   store i32 %2, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %agg.tmp) #4
// CHECK:STDOUT:   br label %ehcleanup
// CHECK:STDOUT:
// CHECK:STDOUT: lpad2:                                            ; preds = %invoke.cont
// CHECK:STDOUT:   %3 = landingpad { ptr, i32 }
// CHECK:STDOUT:           cleanup
// CHECK:STDOUT:   %4 = extractvalue { ptr, i32 } %3, 0
// CHECK:STDOUT:   store ptr %4, ptr %exn.slot, align 8
// CHECK:STDOUT:   %5 = extractvalue { ptr, i32 } %3, 1
// CHECK:STDOUT:   store i32 %5, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %agg.tmp1) #4
// CHECK:STDOUT:   br label %ehcleanup
// CHECK:STDOUT:
// CHECK:STDOUT: ehcleanup:                                        ; preds = %lpad2, %lpad
// CHECK:STDOUT:   call void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %b) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %b) #4
// CHECK:STDOUT:   call void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %a) #4
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #4
// CHECK:STDOUT:   br label %eh.resume
// CHECK:STDOUT:
// CHECK:STDOUT: eh.resume:                                        ; preds = %ehcleanup
// CHECK:STDOUT:   %exn = load ptr, ptr %exn.slot, align 8
// CHECK:STDOUT:   %sel = load i32, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   %lpad.val = insertvalue { ptr, i32 } poison, ptr %exn, 0
// CHECK:STDOUT:   %lpad.val7 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1
// CHECK:STDOUT:   resume { ptr, i32 } %lpad.val7
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder ptr %a, { 0, 2, 1, 3, 4 }
// CHECK:STDOUT:   uselistorder ptr %b, { 0, 2, 1, 3, 4 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL1FENS_1AE() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = alloca %"class.Carbon::A", align 1
// CHECK:STDOUT:   call void @_CF__carbon_thunkinst70000035.Main(ptr noundef nonnull align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @__gxx_personality_v0(...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon1AD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !12
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.A.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL1FENS_1BE() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = alloca %"class.Carbon::B", align 1
// CHECK:STDOUT:   call void @_CF__carbon_thunkinst70000050.Main(ptr noundef nonnull align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon1BD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !15
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.B.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CDoit.A.Main:I.Main"(ptr %self) #4 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CDoit.B.Main:I.Main"(ptr %self) #4 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_C__destroy_thunk:thunk.A.Main"(ptr %self) #5 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.68011419f123494e:core.Destroy.Core"(ptr %self), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.68011419f123494e:core.Destroy.Core"(ptr %self) #4 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_C__destroy_thunk:thunk.B.Main"(ptr %self) #5 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.6c3865e9cbd9f6de:core.Destroy.Core"(ptr %self), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6c3865e9cbd9f6de:core.Destroy.Core"(ptr %self) #4 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunkinst70000035.Main(ptr %_) #4 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.3e1f7ed08cd9ee58(ptr %_), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunkinst70000050.Main(ptr %_) #4 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.bd1b6d59f703c190(ptr %_), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #4 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z1Gv(), !dbg !56
// CHECK:STDOUT:   ret i32 0, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.3e1f7ed08cd9ee58(ptr %t) #4 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CDoit.A.Main:I.Main"(ptr %t), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.bd1b6d59f703c190(ptr %t) #4 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CDoit.B.Main:I.Main"(ptr %t), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon1AD2Ev, { 0, 2, 1, 3 }
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon1BD2Ev, { 0, 2, 1, 3 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.end.p0, { 3, 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nounwind }
// CHECK:STDOUT: attributes #5 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "generic_type_impls_interface.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = !{!13, !13, i64 0}
// CHECK:STDOUT: !13 = !{!"p1 _ZTSN6Carbon1AE", !14, i64 0}
// CHECK:STDOUT: !14 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !15 = !{!16, !16, i64 0}
// CHECK:STDOUT: !16 = !{!"p1 _ZTSN6Carbon1BE", !14, i64 0}
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Doit", linkageName: "_CDoit.A.Main:I.Main", scope: null, file: !1, line: 10, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !20}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !17, type: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 10, column: 5, scope: !17)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Doit", linkageName: "_CDoit.B.Main:I.Main", scope: null, file: !1, line: 15, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 15, column: 5, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.A.Main", scope: null, file: !1, line: 8, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !20)
// CHECK:STDOUT: !31 = !DILocation(line: 8, column: 1, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.68011419f123494e:core.Destroy.Core", scope: null, file: !1, line: 8, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !35 = !DILocation(line: 8, column: 1, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.B.Main", scope: null, file: !1, line: 13, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !20)
// CHECK:STDOUT: !39 = !DILocation(line: 13, column: 1, scope: !36)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6c3865e9cbd9f6de:core.Destroy.Core", scope: null, file: !1, line: 13, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !20)
// CHECK:STDOUT: !43 = !DILocation(line: 13, column: 1, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "F__carbon_thunkinst70000035", linkageName: "_CF__carbon_thunkinst70000035.Main", scope: null, file: !1, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !20)
// CHECK:STDOUT: !47 = !DILocation(line: 19, column: 1, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "F__carbon_thunkinst70000050", linkageName: "_CF__carbon_thunkinst70000050.Main", scope: null, file: !1, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !49)
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !48, type: !20)
// CHECK:STDOUT: !51 = !DILocation(line: 19, column: 1, scope: !48)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !1, line: 32, type: !53, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !53 = !DISubroutineType(types: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !56 = !DILocation(line: 33, column: 3, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 32, column: 1, scope: !52)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.3e1f7ed08cd9ee58", scope: null, file: !1, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !59)
// CHECK:STDOUT: !59 = !{!60}
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !58, type: !20)
// CHECK:STDOUT: !61 = !DILocation(line: 20, column: 3, scope: !58)
// CHECK:STDOUT: !62 = !DILocation(line: 19, column: 1, scope: !58)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.bd1b6d59f703c190", scope: null, file: !1, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !64)
// CHECK:STDOUT: !64 = !{!65}
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !63, type: !20)
// CHECK:STDOUT: !66 = !DILocation(line: 20, column: 3, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 19, column: 1, scope: !63)
// CHECK:STDOUT:
