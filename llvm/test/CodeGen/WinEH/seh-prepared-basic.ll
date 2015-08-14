; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-windows-coreclr < %s | FileCheck %s

; Test case based on this code:
; extern "C" unsigned long _exception_code();
; extern "C" int filt(unsigned long);
; extern "C" void g();
; extern "C" void do_except() {
;   __try {
;     g();
;   } __except(filt(_exception_code())) {
;   }
; }

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: uwtable
define void @do_except() #0 personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  call void (...) @llvm.localescape()
  invoke void @g() #5
          to label %__try.cont unwind label %lpad1

lpad1:                                            ; preds = %entry
  %ehvals = landingpad { i8*, i32 }
          catch i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@do_except@@" to i8*)
  %recover = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@do_except@@" to i8*), i32 -1, i8* blockaddress(@do_except, %__try.cont))
  indirectbr i8* %recover, [label %__try.cont]

__try.cont:                                       ; preds = %lpad1, %entry
  ret void
}

; CHECK-LABEL: do_except:
; CHECK: .seh_handler __C_specific_handler
; CHECK-NOT: jmpq *
; CHECK: .seh_handlerdata
; CHECK-NEXT: .long 1
; CHECK-NEXT: .long .Ltmp{{.*}}
; CHECK-NEXT: .long .Ltmp{{.*}}
; CHECK-NEXT: .long "?filt$0@0@do_except@@"@IMGREL
; CHECK-NEXT: .long .Ltmp{{.*}}@IMGREL

; Function Attrs: noinline nounwind
define internal i32 @"\01?filt$0@0@do_except@@"(i8* nocapture readonly %exception_pointers, i8* nocapture readnone %frame_pointer) #1 {
entry:
  %0 = bitcast i8* %exception_pointers to i32**
  %1 = load i32*, i32** %0, align 8
  %2 = load i32, i32* %1, align 4
  %call = tail call i32 @filt(i32 %2) #4
  ret i32 %call
}

declare i32 @filt(i32) #2

declare void @g() #2

declare i32 @__C_specific_handler(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #3

; Function Attrs: nounwind
declare i8* @llvm.eh.actions(...) #4

; Function Attrs: nounwind
declare void @llvm.localescape(...) #4

; Function Attrs: nounwind readnone
declare i8* @llvm.localrecover(i8*, i8*, i32) #3

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" "wineh-parent"="do_except" }
attributes #1 = { noinline nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }
attributes #5 = { noinline }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 "}
