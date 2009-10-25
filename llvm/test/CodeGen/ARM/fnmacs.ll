; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon -arm-use-neon-fp=0 | FileCheck %s -check-prefix=NEON
; RUN: llc < %s -march=arm -mattr=+neon -arm-use-neon-fp=1 | FileCheck %s -check-prefix=NEONFP

define float @test(float %acc, float %a, float %b) {
entry:
; VFP2: fnmacs
; NEON: fnmacs

; NEONFP:     vmls
; NEONFP-NOT: fcpys
; NEONFP:     fmrs

	%0 = fmul float %a, %b
        %1 = fsub float %acc, %0
	ret float %1
}

