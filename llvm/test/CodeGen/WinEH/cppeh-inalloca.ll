; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is built from the following code:
; struct A {
;   A(int a);
;   A(const A &o);
;   ~A();
;   int a;
; };
;
; void may_throw();
;
; int test(A a) {
;   try {
;     may_throw();
;   }
;   catch (int e) {
;     return a.a + e;
;   }
;   return 0;
; }
;
; The test was built for a 32-bit Windows target and then the reference to
; the inalloca instruction was manually sunk into the landingpad.

; ModuleID = 'cppeh-inalloca.cpp'

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%struct.A = type { i32 }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat

; The function entry should be rewritten like this.
; CHECK: define i32 @"\01?test@@YAHUA@@@Z"(<{ %struct.A }>* inalloca)
; CHECK: entry:
; CHECK:   [[TMP_REGMEM:\%.+]] = alloca <{ %struct.A }>*
; CHECK:   store <{ %struct.A }>* %0, <{ %struct.A }>** [[TMP_REGMEM]]
; CHECK:   [[RETVAL:\%.+]] = alloca i32, align 4
; CHECK:   [[E_PTR:\%.+]] = alloca i32, align 4
; CHECK:   [[CLEANUP_SLOT:\%.+]] = alloca i32
; CHECK:   call void (...) @llvm.frameescape(i32* %e, <{ %struct.A }>** [[TMP_REGMEM]], i32* [[RETVAL]], i32* [[CLEANUP_SLOT]])
; CHECK:   invoke void @"\01?may_throw@@YAXXZ"()
; CHECK:           to label %invoke.cont unwind label %[[LPAD_LABEL:lpad[0-9]*]]

define i32 @"\01?test@@YAHUA@@@Z"(<{ %struct.A }>* inalloca) #0 {
entry:
  %retval = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %e = alloca i32, align 4
  %cleanup.dest.slot = alloca i32
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br label %try.cont

; CHECK: [[LPAD_LABEL]]:{{[ ]+}}; preds = %entry
; CHECK:   landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
; CHECK-NEXT:           cleanup
; CHECK-NEXT:           catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
; CHECK-NEXT:   [[RECOVER:\%recover.*]] = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32 0, i8* (i8*, i8*)* @"\01?test@@YAHUA@@@Z.catch", i32 0, void (i8*, i8*)* @"\01?test@@YAHUA@@@Z.cleanup")
; CHECK-NEXT:   indirectbr i8* [[RECOVER]], [label %cleanup]

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot
  br label %catch.dispatch

; CHECK-NOT: catch.dispatch:

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)) #3
  %matches = icmp eq i32 %sel, %4
  br i1 %matches, label %catch, label %ehcleanup

; CHECK-NOT: catch:

catch:                                            ; preds = %catch.dispatch
  %exn = load i8*, i8** %exn.slot
  %e.i8 = bitcast i32* %e to i8*
  call void @llvm.eh.begincatch(i8* %exn, i8* %e.i8) #3
  %a = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %0, i32 0, i32 0
  %a1 = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 0
  %tmp8 = load i32, i32* %a1, align 4
  %tmp9 = load i32, i32* %e, align 4
  %add = add nsw i32 %tmp8, %tmp9
  store i32 %add, i32* %retval
  store i32 1, i32* %cleanup.dest.slot
  call void @llvm.eh.endcatch() #3
  br label %cleanup

try.cont:                                         ; preds = %invoke.cont
  store i32 0, i32* %retval
  store i32 1, i32* %cleanup.dest.slot
  br label %cleanup

; The cleanup block should be re-written like this.
; CHECK: cleanup:{{[ ]+}}; preds = %[[LPAD_LABEL]], %try.cont
; CHECK:   %a2 = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %0, i32 0, i32 0
; CHECK:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %a2)
; CHECK:   [[TMP1:\%.+]] = load i32, i32* [[RETVAL]]
; CHECK:   ret i32 [[TMP1]]

cleanup:                                          ; preds = %try.cont, %catch
  %a2 = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %0, i32 0, i32 0
  call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %a2) #3
  %tmp10 = load i32, i32* %retval
  ret i32 %tmp10

; CHECK-NOT: ehcleanup:

ehcleanup:                                        ; preds = %catch.dispatch
  %a3 = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %0, i32 0, i32 0
  call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %a3) #3
  br label %eh.resume

; CHECK-NOT: eh.resume:

eh.resume:                                        ; preds = %ehcleanup
  %exn2 = load i8*, i8** %exn.slot
  %sel3 = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn2, 0
  %lpad.val4 = insertvalue { i8*, i32 } %lpad.val, i32 %sel3, 1
  resume { i8*, i32 } %lpad.val4

; CHECK: }
}

; The following catch handler should be outlined.
; CHECK: define internal i8* @"\01?test@@YAHUA@@@Z.catch"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_E:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (i32 (<{ %struct.A }>*)* @"\01?test@@YAHUA@@@Z" to i8*), i8* %1, i32 0)
; CHECK:   [[E_PTR:\%.+]] = bitcast i8* [[RECOVER_E]] to i32*
; CHECK:   [[RECOVER_EH_TEMP:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (i32 (<{ %struct.A }>*)* @"\01?test@@YAHUA@@@Z" to i8*), i8* %1, i32 1)
; CHECK:   [[EH_TEMP:\%.+]] = bitcast i8* [[RECOVER_EH_TEMP]] to <{ %struct.A }>**
; CHECK:   [[RECOVER_RETVAL:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (i32 (<{ %struct.A }>*)* @"\01?test@@YAHUA@@@Z" to i8*), i8* %1, i32 2)
; CHECK:   [[RETVAL1:\%.+]] = bitcast i8* [[RECOVER_RETVAL]] to i32*
; CHECK:   [[RECOVER_CLEANUPSLOT:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (i32 (<{ %struct.A }>*)* @"\01?test@@YAHUA@@@Z" to i8*), i8* %1, i32 3)
; CHECK:   [[CLEANUPSLOT1:\%.+]] = bitcast i8* [[RECOVER_CLEANUPSLOT]] to i32*
; CHECK:   [[E_I8PTR:\%.+]] = bitcast i32* [[E_PTR]] to i8*
; CHECK:   [[TMP_RELOAD:\%.+]] = load <{ %struct.A }>*, <{ %struct.A }>** [[EH_TEMP]]
; CHECK:   [[RECOVER_A:\%.+]] = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* [[TMP_RELOAD]], i32 0, i32 0
; CHECK:   [[A1:\%.+]] = getelementptr inbounds %struct.A, %struct.A* [[RECOVER_A]], i32 0, i32 0
; CHECK:   [[TMP2:\%.+]] = load i32, i32* [[A1]], align 4
; CHECK:   [[TMP3:\%.+]] = load i32, i32* [[E_PTR]], align 4
; CHECK:   [[ADD:\%.+]] = add nsw i32 [[TMP2]], [[TMP3]]
; CHECK:   store i32 [[ADD]], i32* [[RETVAL1]]
; CHECK:   store i32 1, i32* [[CLEANUPSLOT1]]
; CHECK:   ret i8* blockaddress(@"\01?test@@YAHUA@@@Z", %cleanup)
; CHECK: }

; The following cleanup handler should be outlined.
; CHECK: define internal void @"\01?test@@YAHUA@@@Z.cleanup"(i8*, i8*)
; CHECK: entry:
; CHECK:   [[RECOVER_EH_TEMP1:\%.+]] = call i8* @llvm.framerecover(i8* bitcast (i32 (<{ %struct.A }>*)* @"\01?test@@YAHUA@@@Z" to i8*), i8* %1, i32 1)
; CHECK:   [[EH_TEMP1:\%.+]] = bitcast i8* [[RECOVER_EH_TEMP]] to <{ %struct.A }>**
; CHECK:   [[TMP_RELOAD1:\%.+]] = load <{ %struct.A }>*, <{ %struct.A }>** [[EH_TEMP1]]
; CHECK:   [[A3:\%.+]] = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* [[TMP_RELOAD1]], i32 0, i32 0
; CHECK:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* [[A3]])
; CHECK:   ret void
; CHECK: }

declare void @"\01?may_throw@@YAXXZ"() #0

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare void @llvm.eh.begincatch(i8*, i8*)

declare void @llvm.eh.endcatch()

; Function Attrs: nounwind
declare x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A*) #2

attributes #0 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 228868)"}
