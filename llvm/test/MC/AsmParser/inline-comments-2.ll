; RUN: llc %s -o - | sed -n -e '/@APP/,/@NO_APP/p' > %t
; RUN: sed -n -e 's/^;CHECK://p' %s > %t2
; RUN: diff %t %t2

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"


; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  call void asm sideeffect "#isolated preprocessor comment", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	@APP
;CHECK:	@isolated preprocessor comment
;CHECK:	@NO_APP
  ret void
}

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!""}
