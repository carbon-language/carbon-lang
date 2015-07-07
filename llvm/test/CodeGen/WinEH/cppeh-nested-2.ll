; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; class Inner {
; public:
;   Inner();
;   ~Inner();
; };
; class Outer {
; public:
;   Outer();
;   ~Outer();
; };
; void test() {
;   try {
;     Outer outer;
;     try {
;        Inner inner;
;        may_throw();
;     } catch (int i) {
;       handle_int(i);
;     }
;   } catch (float f) {
;     handle_float(f);
;   }
;   done();
; }

; ModuleID = 'nested-2.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%class.Outer = type { i8 }
%class.Inner = type { i8 }

@_ZTIf = external constant i8*
@_ZTIi = external constant i8*

; The function entry should be rewritten like this.
; CHECK: define void @_Z4testv()
; CHECK: entry:
; CHECK:   %outer = alloca %class.Outer, align 1
; CHECK:   %inner = alloca %class.Inner, align 1
; CHECK:   %i = alloca i32, align 4
; CHECK:   %f = alloca float, align 4
; CHECK:   call void (...) @llvm.localescape(float* %f, i32* %i, %class.Outer* %outer, %class.Inner* %inner)
; CHECK:   invoke void @_ZN5OuterC1Ev(%class.Outer* %outer)
; CHECK:           to label %invoke.cont unwind label %[[LPAD_LABEL:lpad[0-9]*]]

; Function Attrs: uwtable
define void @_Z4testv() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %outer = alloca %class.Outer, align 1
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %inner = alloca %class.Inner, align 1
  %i = alloca i32, align 4
  %f = alloca float, align 4
  invoke void @_ZN5OuterC1Ev(%class.Outer* %outer)
          to label %invoke.cont unwind label %lpad

; CHECK: invoke.cont:
; CHECK:   invoke void @_ZN5InnerC1Ev(%class.Inner* %inner)
; CHECK:           to label %invoke.cont2 unwind label %[[LPAD1_LABEL:lpad[0-9]*]]

invoke.cont:                                      ; preds = %entry
  invoke void @_ZN5InnerC1Ev(%class.Inner* %inner)
          to label %invoke.cont2 unwind label %lpad1

; CHECK: invoke.cont2:
; CHECK:   invoke void @_Z9may_throwv()
; CHECK:           to label %invoke.cont4 unwind label %[[LPAD3_LABEL:lpad[0-9]*]]

invoke.cont2:                                     ; preds = %invoke.cont
  invoke void @_Z9may_throwv()
          to label %invoke.cont4 unwind label %lpad3

; CHECK: invoke.cont4:
; CHECK:   invoke void @_ZN5InnerD1Ev(%class.Inner* %inner)
; CHECK:           to label %invoke.cont5 unwind label %[[LPAD1_LABEL]]

invoke.cont4:                                     ; preds = %invoke.cont2
  invoke void @_ZN5InnerD1Ev(%class.Inner* %inner)
          to label %invoke.cont5 unwind label %lpad1

; CHECK: invoke.cont5:
; CHECK:   br label %try.cont

invoke.cont5:                                     ; preds = %invoke.cont4
  br label %try.cont

; CHECK: [[LPAD_LABEL]]:
; CHECK:   landingpad { i8*, i32 }
; CHECK-NEXT:           catch i8* bitcast (i8** @_ZTIf to i8*)
; CHECK-NEXT:   [[RECOVER:\%.+]] = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (i8** @_ZTIf to i8*), i32 0, i8* (i8*, i8*)* @_Z4testv.catch)
; CHECK-NEXT:   indirectbr i8* [[RECOVER]], [label %try.cont19]

lpad:                                             ; preds = %try.cont, %entry
  %tmp = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIf to i8*)
  %tmp1 = extractvalue { i8*, i32 } %tmp, 0
  store i8* %tmp1, i8** %exn.slot
  %tmp2 = extractvalue { i8*, i32 } %tmp, 1
  store i32 %tmp2, i32* %ehselector.slot
  br label %catch.dispatch11

; CHECK: [[LPAD1_LABEL]]:
; CHECK:   landingpad { i8*, i32 }
; CHECK-NEXT:           cleanup
; CHECK-NEXT:           catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT:           catch i8* bitcast (i8** @_ZTIf to i8*)
; CHECK-NEXT:   [[RECOVER1:\%.+]] = call i8* (...) @llvm.eh.actions(
; CHECK-SAME:       i32 1, i8* bitcast (i8** @_ZTIi to i8*), i32 1, i8* (i8*, i8*)* @_Z4testv.catch.1,
; CHECK-SAME:       i32 0, void (i8*, i8*)* @_Z4testv.cleanup,
; CHECK-SAME:       i32 1, i8* bitcast (i8** @_ZTIf to i8*), i32 0, i8* (i8*, i8*)* @_Z4testv.catch)
; CHECK-NEXT:   indirectbr i8* [[RECOVER1]], [label %try.cont, label %try.cont19]

lpad1:                                            ; preds = %invoke.cont4, %invoke.cont
  %tmp3 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* bitcast (i8** @_ZTIf to i8*)
  %tmp4 = extractvalue { i8*, i32 } %tmp3, 0
  store i8* %tmp4, i8** %exn.slot
  %tmp5 = extractvalue { i8*, i32 } %tmp3, 1
  store i32 %tmp5, i32* %ehselector.slot
  br label %catch.dispatch

; CHECK: [[LPAD3_LABEL]]:
; CHECK:   landingpad { i8*, i32 }
; CHECK-NEXT:           cleanup
; CHECK-NEXT:           catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT:           catch i8* bitcast (i8** @_ZTIf to i8*)
; CHECK-NEXT:   [[RECOVER3:\%.+]] = call i8* (...) @llvm.eh.actions(
; CHECK-SAME:       i32 0, void (i8*, i8*)* @_Z4testv.cleanup.2,
; CHECK-SAME:       i32 1, i8* bitcast (i8** @_ZTIi to i8*), i32 1, i8* (i8*, i8*)* @_Z4testv.catch.1,
; CHECK-SAME:       i32 0, void (i8*, i8*)* @_Z4testv.cleanup,
; CHECK-SAME:       i32 1, i8* bitcast (i8** @_ZTIf to i8*), i32 0, i8* (i8*, i8*)* @_Z4testv.catch)
; CHECK-NEXT:   indirectbr i8* [[RECOVER3]], [label %try.cont, label %try.cont19]

lpad3:                                            ; preds = %invoke.cont2
  %tmp6 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* bitcast (i8** @_ZTIf to i8*)
  %tmp7 = extractvalue { i8*, i32 } %tmp6, 0
  store i8* %tmp7, i8** %exn.slot
  %tmp8 = extractvalue { i8*, i32 } %tmp6, 1
  store i32 %tmp8, i32* %ehselector.slot
  call void @_ZN5InnerD1Ev(%class.Inner* %inner)
  br label %catch.dispatch

; CHECK-NOT: catch.dispatch:

catch.dispatch:                                   ; preds = %lpad3, %lpad1
  %sel = load i32, i32* %ehselector.slot
  %tmp9 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #4
  %matches = icmp eq i32 %sel, %tmp9
  br i1 %matches, label %catch, label %ehcleanup

; CHECK-NOT: catch:

catch:                                            ; preds = %catch.dispatch
  %exn = load i8*, i8** %exn.slot
  %i.i8 = bitcast i32* %i to i8*
  call void @llvm.eh.begincatch(i8* %exn, i8* %i.i8) #4
  %tmp13 = load i32, i32* %i, align 4
  invoke void @_Z10handle_inti(i32 %tmp13)
          to label %invoke.cont8 unwind label %lpad7

; CHECK-NOT: invoke.cont8:

invoke.cont8:                                     ; preds = %catch
  call void @llvm.eh.endcatch() #4
  br label %try.cont

; CHECK: try.cont:
; CHECK:   invoke void @_ZN5OuterD1Ev(%class.Outer* %outer)
; CHECK:           to label %invoke.cont9 unwind label %[[LPAD_LABEL]]

try.cont:                                         ; preds = %invoke.cont8, %invoke.cont5
  invoke void @_ZN5OuterD1Ev(%class.Outer* %outer)
          to label %invoke.cont9 unwind label %lpad

invoke.cont9:                                     ; preds = %try.cont
  br label %try.cont19

; CHECK-NOT: lpad7:

lpad7:                                            ; preds = %catch
  %tmp14 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIf to i8*)
  %tmp15 = extractvalue { i8*, i32 } %tmp14, 0
  store i8* %tmp15, i8** %exn.slot
  %tmp16 = extractvalue { i8*, i32 } %tmp14, 1
  store i32 %tmp16, i32* %ehselector.slot
  call void @llvm.eh.endcatch() #4
  br label %ehcleanup

; CHECK-NOT: ehcleanup:                                        ; preds = %lpad7, %catch.dispatch

ehcleanup:                                        ; preds = %lpad7, %catch.dispatch
  call void @_ZN5OuterD1Ev(%class.Outer* %outer)
  br label %catch.dispatch11

; CHECK-NOT: catch.dispatch11:

catch.dispatch11:                                 ; preds = %ehcleanup, %lpad
  %sel12 = load i32, i32* %ehselector.slot
  %tmp17 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIf to i8*)) #4
  %matches13 = icmp eq i32 %sel12, %tmp17
  br i1 %matches13, label %catch14, label %eh.resume

; CHECK-NOT: catch14:

catch14:                                          ; preds = %catch.dispatch11
  %exn15 = load i8*, i8** %exn.slot
  %f.i8 = bitcast float* %f to i8*
  call void @llvm.eh.begincatch(i8* %exn15, i8* %f.i8) #4
  %tmp21 = load float, float* %f, align 4
  call void @_Z12handle_floatf(float %tmp21)
  call void @llvm.eh.endcatch() #4
  br label %try.cont19

try.cont19:                                       ; preds = %catch14, %invoke.cont9
  call void @_Z4donev()
  ret void

; CHECK-NOT: eh.resume:

eh.resume:                                        ; preds = %catch.dispatch11
  %exn20 = load i8*, i8** %exn.slot
  %sel21 = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn20, 0
  %lpad.val22 = insertvalue { i8*, i32 } %lpad.val, i32 %sel21, 1
  resume { i8*, i32 } %lpad.val22

; CHECK: }
}

; This catch handler should be outlined.
; CHECK: define internal i8* @_Z4testv.catch(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_F:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @_Z4testv to i8*), i8* %1, i32 0)
; CHECK:   [[F_PTR:\%.+]] = bitcast i8* [[RECOVER_F]] to float*
; CHECK:   [[TMP:\%.+]] = load float, float* [[F_PTR]], align 4
; CHECK:   call void @_Z12handle_floatf(float [[TMP]])
; CHECK:   ret i8* blockaddress(@_Z4testv, %try.cont19)
; CHECK: }

; This catch handler should be outlined.
; CHECK: define internal i8* @_Z4testv.catch.1(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_I:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @_Z4testv to i8*), i8* %1, i32 1)
; CHECK:   [[I_PTR:\%.+]] = bitcast i8* [[RECOVER_I]] to i32*
; CHECK:   [[TMP1:\%.+]] = load i32, i32* [[I_PTR]], align 4
; CHECK:   invoke void @_Z10handle_inti(i32 [[TMP1]])
; CHECK:           to label %invoke.cont8 unwind label %[[LPAD7_LABEL:lpad[0-9]*]]
;
; CHECK: invoke.cont8:                                     ; preds = %entry
; CHECK:   ret i8* blockaddress(@_Z4testv, %try.cont)
;
; CHECK: [[LPAD7_LABEL]]:{{[ ]+}}; preds = %entry
; CHECK:   [[LPAD7_VAL:\%.+]] = landingpad { i8*, i32 }
; (FIXME) The nested handler body isn't being populated yet.
; CHECK: }

; This cleanup handler should be outlined.
; CHECK: define internal void @_Z4testv.cleanup(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_OUTER:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @_Z4testv to i8*), i8* %1, i32 2)
; CHECK:   [[OUTER_PTR:\%.+]] = bitcast i8* [[RECOVER_OUTER]] to %class.Outer*
; CHECK:   call void @_ZN5OuterD1Ev(%class.Outer* [[OUTER_PTR]])
; CHECK:   ret void
; CHECK: }

; This cleanup handler should be outlined.
; CHECK: define internal void @_Z4testv.cleanup.2(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_INNER:\%.+]] = call i8* @llvm.localrecover(i8* bitcast (void ()* @_Z4testv to i8*), i8* %1, i32 3)
; CHECK:   [[INNER_PTR:\%.+]] = bitcast i8* [[RECOVER_INNER]] to %class.Inner*
; CHECK:   call void @_ZN5InnerD1Ev(%class.Inner* [[INNER_PTR]])
; CHECK:   ret void
; CHECK: }



declare void @_ZN5OuterC1Ev(%class.Outer*) #1

declare i32 @__CxxFrameHandler3(...)

declare void @_ZN5InnerC1Ev(%class.Inner*) #1

declare void @_Z9may_throwv() #1

declare void @_ZN5InnerD1Ev(%class.Inner*) #1

declare void @llvm.eh.begincatch(i8*, i8*)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #3

declare void @_Z10handle_inti(i32) #1

declare void @llvm.eh.endcatch()

declare void @_ZN5OuterD1Ev(%class.Outer*) #1

declare void @_Z12handle_floatf(float) #1

declare void @_Z4donev() #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline noreturn nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 226027)"}
