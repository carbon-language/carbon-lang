; RUN: llc -mtriple=x86_64-linux-gnu -mattr=-ermsb < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=NOFAST
; RUN: llc -mtriple=x86_64-linux-gnu -mattr=+ermsb < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=FAST
; RUN: llc -mtriple=x86_64-linux-gnu -mcpu=haswell < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=HASWELL
; RUN: llc -mtriple=x86_64-linux-gnu -mcpu=generic < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC

%struct.large = type { [4096 x i8] }

declare void @foo(%struct.large* align 8 byval) nounwind

define void @test1(%struct.large* nocapture %x) nounwind {
  call void @foo(%struct.large* align 8 byval %x)
  ret void

; ALL-LABEL: test1:
; NOFAST: rep;movsq
; GENERIC: rep;movsq
; FAST: rep;movsb
; HASWELL: rep;movsb
}

define void @test2(%struct.large* nocapture %x) nounwind minsize {
  call void @foo(%struct.large* align 8 byval %x)
  ret void

; ALL-LABEL: test2:
; NOFAST: rep;movsq
; GENERIC: rep;movsq
; FAST: rep;movsb
; HASWELL: rep;movsb
}

%struct.large_oddsize = type { [4095 x i8] }

declare void @foo_oddsize(%struct.large_oddsize* align 8 byval) nounwind

define void @test3(%struct.large_oddsize* nocapture %x) nounwind minsize {
  call void @foo_oddsize(%struct.large_oddsize* align 8 byval %x)
  ret void

; ALL-LABEL: test3:
; NOFAST: rep;movsb
; GENERIC: rep;movsb
; FAST: rep;movsb
; HASWELL: rep;movsb
}
