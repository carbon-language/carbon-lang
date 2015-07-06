; RUN: opt -S -winehprepare < %s | FileCheck %s

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

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

$"\01??_C@_0BC@LHHILCPN@outer?5finally?5?$CFd?6?$AA@" = comdat any

$"\01??_C@_0BC@JELAHKN@inner?5finally?5?$CFd?6?$AA@" = comdat any

@"\01??_C@_0BC@LHHILCPN@outer?5finally?5?$CFd?6?$AA@" = linkonce_odr unnamed_addr constant [18 x i8] c"outer finally %d\0A\00", comdat, align 1
@"\01??_C@_0BC@JELAHKN@inner?5finally?5?$CFd?6?$AA@" = linkonce_odr unnamed_addr constant [18 x i8] c"inner finally %d\0A\00", comdat, align 1

; Function Attrs: nounwind
define void @may_crash() #0 {
entry:
  store volatile i32 42, i32* null, align 4
  ret void
}

; Function Attrs: nounwind
define i32 @main() #0 personality i8* bitcast (i32 (...)* @_except_handler3 to i8*) {
entry:
  %myres = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 0, i32* %myres, align 4
  invoke void @may_crash() #4
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %0 = call i8* @llvm.frameaddress(i32 0)
  invoke void @"\01?fin$1@0@main@@"(i8 zeroext 0, i8* %0) #4
          to label %invoke.cont.2 unwind label %lpad.1

invoke.cont.2:                                    ; preds = %invoke.cont
  %1 = call i8* @llvm.frameaddress(i32 0)
  call void @"\01?fin$0@0@main@@"(i8 zeroext 0, i8* %1)
  ret i32 0

lpad:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 }
          cleanup
  %3 = extractvalue { i8*, i32 } %2, 0
  store i8* %3, i8** %exn.slot
  %4 = extractvalue { i8*, i32 } %2, 1
  store i32 %4, i32* %ehselector.slot
  %5 = call i8* @llvm.frameaddress(i32 0)
  invoke void @"\01?fin$1@0@main@@"(i8 zeroext 1, i8* %5) #4
          to label %invoke.cont.3 unwind label %lpad.1

lpad.1:                                           ; preds = %lpad, %invoke.cont
  %6 = landingpad { i8*, i32 }
          cleanup
  %7 = extractvalue { i8*, i32 } %6, 0
  store i8* %7, i8** %exn.slot
  %8 = extractvalue { i8*, i32 } %6, 1
  store i32 %8, i32* %ehselector.slot
  br label %ehcleanup

invoke.cont.3:                                    ; preds = %lpad
  br label %ehcleanup

ehcleanup:                                        ; preds = %invoke.cont.3, %lpad.1
  %9 = call i8* @llvm.frameaddress(i32 0)
  call void @"\01?fin$0@0@main@@"(i8 zeroext 1, i8* %9)
  br label %eh.resume

eh.resume:                                        ; preds = %ehcleanup
  %exn = load i8*, i8** %exn.slot
  %sel = load i32, i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
  %lpad.val.4 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
  resume { i8*, i32 } %lpad.val.4
}

; CHECK-LABEL: define i32 @main()
; CHECK: invoke void @may_crash()
;
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 0, void ()* @main.cleanup)
; CHECK-NEXT: indirectbr
;
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 0, void ()* @main.cleanup.1)
; CHECK-NEXT: indirectbr

; CHECK-LABEL: define internal void @main.cleanup()
; CHECK: call i8* @llvm.frameaddress(i32 1)
; CHECK: call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 ()* @main to i8*), i8* %{{.*}})
; CHECK: call void @"\01?fin$1@0@main@@"(i8 zeroext 1, i8* %{{.*}})
; CHECK: call void @"\01?fin$0@0@main@@"(i8 zeroext 1, i8* %{{.*}})

; CHECK-LABEL: define internal void @main.cleanup.1()
; CHECK: call i8* @llvm.frameaddress(i32 1)
; CHECK: call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 ()* @main to i8*), i8* %{{.*}})
; CHECK: call void @"\01?fin$0@0@main@@"(i8 zeroext 1, i8* %{{.*}})

; Function Attrs: noinline nounwind
define internal void @"\01?fin$0@0@main@@"(i8 zeroext %abnormal_termination, i8* %frame_pointer) #1 {
entry:
  %frame_pointer.addr = alloca i8*, align 4
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.frameaddress(i32 1)
  %1 = call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 ()* @main to i8*), i8* %0)
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 4
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  %2 = zext i8 %abnormal_termination to i32
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @"\01??_C@_0BC@LHHILCPN@outer?5finally?5?$CFd?6?$AA@", i32 0, i32 0), i32 %2)
  ret void
}

; Function Attrs: nounwind readnone
declare i8* @llvm.frameaddress(i32) #2

; Function Attrs: nounwind readnone
declare i8* @llvm.x86.seh.recoverfp(i8*, i8*) #2

declare i32 @printf(i8*, ...) #3

; Function Attrs: noinline nounwind
define internal void @"\01?fin$1@0@main@@"(i8 zeroext %abnormal_termination, i8* %frame_pointer) #1 {
entry:
  %frame_pointer.addr = alloca i8*, align 4
  %abnormal_termination.addr = alloca i8, align 1
  %0 = call i8* @llvm.frameaddress(i32 1)
  %1 = call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 ()* @main to i8*), i8* %0)
  store i8* %frame_pointer, i8** %frame_pointer.addr, align 4
  store i8 %abnormal_termination, i8* %abnormal_termination.addr, align 1
  %2 = zext i8 %abnormal_termination to i32
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @"\01??_C@_0BC@JELAHKN@inner?5finally?5?$CFd?6?$AA@", i32 0, i32 0), i32 %2)
  call void @may_crash()
  ret void
}

declare i32 @_except_handler3(...)

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 "}
