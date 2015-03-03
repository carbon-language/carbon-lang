; RUN: opt -mtriple=i386-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

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
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i386-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%struct.A = type { i32 }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat

; The function entry should be rewritten like this.
; CHECK: define i32 @"\01?test@@YAHUA@@@Z"(<{ %struct.A }>* inalloca) #0 {
; CHECK: entry:
; CHECK:   %frame.alloc = call i8* @llvm.frameallocate(i32 24)
; CHECK:   %eh.data = bitcast i8* %frame.alloc to %"struct.\01?test@@YAHUA@@@Z.ehdata"*
; CHECK:   %.tmp.reg2mem = getelementptr inbounds %"struct.\01?test@@YAHUA@@@Z.ehdata", %"struct.\01?test@@YAHUA@@@Z.ehdata"* %eh.data, i32 0, i32 3
; CHECK:   %.tmp = select i1 true, <{ %struct.A }>* %0, <{ %struct.A }>* undef
; CHECK:   store <{ %struct.A }>* %.tmp, <{ %struct.A }>** %.tmp.reg2mem
; CHECK-NOT:  %retval = alloca i32, align 4
; CHECK:   %retval = getelementptr inbounds %"struct.\01?test@@YAHUA@@@Z.ehdata", %"struct.\01?test@@YAHUA@@@Z.ehdata"* %eh.data, i32 0, i32 4
; CHECK:   %exn.slot = alloca i8*
; CHECK:   %ehselector.slot = alloca i32
; CHECK-NOT:  %e = alloca i32, align 4
; CHECK:   %e = getelementptr inbounds %"struct.\01?test@@YAHUA@@@Z.ehdata", %"struct.\01?test@@YAHUA@@@Z.ehdata"* %eh.data, i32 0, i32 2
; CHECK:   %cleanup.dest.slot = getelementptr inbounds %"struct.\01?test@@YAHUA@@@Z.ehdata", %"struct.\01?test@@YAHUA@@@Z.ehdata"* %eh.data, i32 0, i32 5
; CHECK:   invoke void @"\01?may_throw@@YAXXZ"()
; CHECK:           to label %invoke.cont unwind label %lpad

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

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          cleanup
          catch i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot
  br label %catch.dispatch

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*)) #3
  %matches = icmp eq i32 %sel, %4
  br i1 %matches, label %catch, label %ehcleanup

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
; CHECK: cleanup:                                          ; preds = %try.cont, %catch
; CHECK-NOT:  %a2 = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %0, i32 0, i32 0
; CHECK:   %.tmp.reload1 = load volatile <{ %struct.A }>*, <{ %struct.A }>** %.tmp.reg2mem
; CHECK:   %a2 = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %.tmp.reload1, i32 0, i32 0
; CHECK:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %a2) #2
; CHECK:   %tmp10 = load i32, i32* %retval
; CHECK:   ret i32 %tmp10

cleanup:                                          ; preds = %try.cont, %catch
  %a2 = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %0, i32 0, i32 0
  call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %a2) #3
  %tmp10 = load i32, i32* %retval
  ret i32 %tmp10

ehcleanup:                                        ; preds = %catch.dispatch
  %a3 = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %0, i32 0, i32 0
  call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %a3) #3
  br label %eh.resume

eh.resume:                                        ; preds = %ehcleanup
  %exn2 = load i8*, i8** %exn.slot
  %sel3 = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn2, 0
  %lpad.val4 = insertvalue { i8*, i32 } %lpad.val, i32 %sel3, 1
  resume { i8*, i32 } %lpad.val4
}

; The following catch handler should be outlined.
; CHECK: define internal i8* @"\01?test@@YAHUA@@@Z.catch"(i8*, i8*) {
; CHECK: entry:
; CHECK:   %eh.alloc = call i8* @llvm.framerecover(i8* bitcast (i32 (<{ %struct.A }>*)* @"\01?test@@YAHUA@@@Z" to i8*), i8* %1)
; CHECK:   %eh.data = bitcast i8* %eh.alloc to %"struct.\01?test@@YAHUA@@@Z.ehdata"*
; CHECK:   %e = getelementptr inbounds %"struct.\01?test@@YAHUA@@@Z.ehdata", %"struct.\01?test@@YAHUA@@@Z.ehdata"* %eh.data, i32 0, i32 2
; CHECK:   %eh.temp.alloca = getelementptr inbounds %"struct.\01?test@@YAHUA@@@Z.ehdata", %"struct.\01?test@@YAHUA@@@Z.ehdata"* %eh.data, i32 0, i32 3
; CHECK:   %.reload = load <{ %struct.A }>*, <{ %struct.A }>** %eh.temp.alloca
; CHECK:   %retval = getelementptr inbounds %"struct.\01?test@@YAHUA@@@Z.ehdata", %"struct.\01?test@@YAHUA@@@Z.ehdata"* %eh.data, i32 0, i32 4
; CHECK:   %cleanup.dest.slot = getelementptr inbounds %"struct.\01?test@@YAHUA@@@Z.ehdata", %"struct.\01?test@@YAHUA@@@Z.ehdata"* %eh.data, i32 0, i32 5
; CHECK:   %a = getelementptr inbounds <{ %struct.A }>, <{ %struct.A }>* %.reload, i32 0, i32 0
; CHECK:   %a1 = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 0
; CHECK:   %tmp8 = load i32, i32* %a1, align 4
; CHECK:   %tmp9 = load i32, i32* %e, align 4
; CHECK:   %add = add nsw i32 %tmp8, %tmp9
; CHECK:   store i32 %add, i32* %retval
; CHECK:   store i32 1, i32* %cleanup.dest.slot
; CHECK:   ret i8* blockaddress(@"\01?test@@YAHUA@@@Z", %cleanup)
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
