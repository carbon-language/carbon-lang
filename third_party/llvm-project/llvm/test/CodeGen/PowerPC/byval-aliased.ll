; RUN: llc -verify-machineinstrs -mcpu=ppc64 -ppc-asm-full-reg-names < %s | FileCheck %s
target datalayout = "E-m:o-p:32:32-f64:32:64-n32"
target triple = "powerpc-unknown-linux-gnu"

%struct.sm = type { i8, i8 }

; Function Attrs: nounwind ssp
define void @foo(%struct.sm* byval(%struct.sm) %s) #0 {
entry:
  %a = getelementptr inbounds %struct.sm, %struct.sm* %s, i32 0, i32 0
  %0 = load i8, i8* %a, align 1
  %conv2 = zext i8 %0 to i32
  %add = add nuw nsw i32 %conv2, 3
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, i8* %a, align 1
  call void @bar(%struct.sm* byval(%struct.sm) %s, %struct.sm* byval(%struct.sm) %s) #1
  ret void
}

; CHECK-LABEL: @foo
; CHECK: stb {{r[0-9]+}}, [[OFF:[0-9]+]]({{r[3?1]}})
; CHECK: lhz r4, [[OFF]]({{r[3?1]}})
; CHECK: bl bar
; CHECK: blr

declare void @bar(%struct.sm* byval(%struct.sm), %struct.sm* byval(%struct.sm))

attributes #0 = { nounwind ssp }
attributes #1 = { nounwind }

