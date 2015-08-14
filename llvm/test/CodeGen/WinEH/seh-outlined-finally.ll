; RUN: opt -S -winehprepare -mtriple=x86_64-windows-msvc < %s | FileCheck %s
; RUN: opt -S -winehprepare -mtriple=x86_64-windows-coreclr < %s | FileCheck %s

; Test case based on this code:
;
; extern "C" int _abnormal_termination();
; #pragma intrinsic(_abnormal_termination)
; extern "C" int printf(const char *, ...);
; extern "C" void may_crash() {
;   *(volatile int *)0 = 42;
; }
; int main() {
;   int myres = 0;
;   __try {
;     __try {
;       may_crash();
;     } __finally {
;       printf("inner finally %d\n", _abnormal_termination());
;       may_crash();
;     }
;   } __finally {
;     printf("outer finally %d\n", _abnormal_termination());
;   }
; }
;
; Note that if the inner finally crashes, the outer finally still runs. There
; is nothing like a std::terminate call in this situation.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"

@str_outer_finally = linkonce_odr unnamed_addr constant [18 x i8] c"outer finally %d\0A\00", align 1
@str_inner_finally = linkonce_odr unnamed_addr constant [18 x i8] c"inner finally %d\0A\00", align 1

; Function Attrs: nounwind uwtable
define void @may_crash() #0 {
entry:
  store volatile i32 42, i32* null, align 4
  ret void
}

; Function Attrs: uwtable
define i32 @main() #1 personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  %myres = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 0, i32* %myres, align 4
  invoke void @may_crash() #4
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %0 = call i8* @llvm.localaddress()
  invoke void @"\01?fin$1@0@main@@"(i1 zeroext false, i8* %0) #4
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %invoke.cont
  %1 = call i8* @llvm.localaddress()
  call void @"\01?fin$0@0@main@@"(i1 zeroext false, i8* %1)
  ret i32 0

lpad:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 }
          cleanup
  %3 = extractvalue { i8*, i32 } %2, 0
  store i8* %3, i8** %exn.slot
  %4 = extractvalue { i8*, i32 } %2, 1
  store i32 %4, i32* %ehselector.slot
  %5 = call i8* @llvm.localaddress()
  invoke void @"\01?fin$1@0@main@@"(i1 zeroext true, i8* %5) #4
          to label %invoke.cont3 unwind label %lpad1

lpad1:                                            ; preds = %lpad, %invoke.cont
  %6 = landingpad { i8*, i32 }
          cleanup
  %7 = extractvalue { i8*, i32 } %6, 0
  store i8* %7, i8** %exn.slot
  %8 = extractvalue { i8*, i32 } %6, 1
  store i32 %8, i32* %ehselector.slot
  br label %ehcleanup

invoke.cont3:                                     ; preds = %lpad
  br label %ehcleanup

ehcleanup:                                        ; preds = %invoke.cont3, %lpad1
  %9 = call i8* @llvm.localaddress()
  call void @"\01?fin$0@0@main@@"(i1 zeroext true, i8* %9)
  br label %eh.resume

eh.resume:                                        ; preds = %ehcleanup
  %exn = load i8*, i8** %exn.slot
  %sel = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
  %lpad.val4 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
  resume { i8*, i32 } %lpad.val4
}

; CHECK-NOT: define internal void @

; CHECK-LABEL: define i32 @main()
; CHECK: invoke void @may_crash()
;
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 0, void (i1, i8*)* @"\01?fin$1@0@main@@", i32 0, void (i1, i8*)* @"\01?fin$0@0@main@@")
; CHECK-NEXT: indirectbr
;
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 0, void (i1, i8*)* @"\01?fin$0@0@main@@")
; CHECK-NEXT: indirectbr

; There should not be any *new* cleanup helpers, just the existing ones.
; CHECK-NOT: define internal void @
; CHECK: define internal void @"\01?fin$0@0@main@@"(i1 zeroext %abnormal_termination, i8* %frame_pointer)
; CHECK-NOT: define internal void @
; CHECK: define internal void @"\01?fin$1@0@main@@"(i1 zeroext %abnormal_termination, i8* %frame_pointer)
; CHECK-NOT: define internal void @

define internal void @"\01?fin$0@0@main@@"(i1 zeroext %abnormal_termination, i8* %frame_pointer) #2 {
entry:
  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  %frombool = zext i1 %abnormal_termination to i8
  store i8 %frombool, i8* %abnormal_termination.addr, align 1
  %0 = zext i1 %abnormal_termination to i32
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @str_outer_finally, i32 0, i32 0), i32 %0)
  ret void
}

declare i32 @printf(i8*, ...) #2

define internal void @"\01?fin$1@0@main@@"(i1 zeroext %abnormal_termination, i8* %frame_pointer) #2 {
entry:
  %frame_pointer.addr = alloca i8*, align 8
  %abnormal_termination.addr = alloca i8, align 1
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 8
  %frombool = zext i1 %abnormal_termination to i8
  store i8 %frombool, i8* %abnormal_termination.addr, align 1
  %0 = zext i1 %abnormal_termination to i32
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @str_inner_finally, i32 0, i32 0), i32 %0)
  call void @may_crash()
  ret void
}

declare i32 @__C_specific_handler(...)

; Function Attrs: nounwind readnone
declare i8* @llvm.localaddress() #3

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { noinline }
