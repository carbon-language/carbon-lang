; This file is used by testlink1.ll, so it doesn't actually do anything itself
;
; RUN: echo

target endian = little
target pointersize = 32

  %myint = type ushort
	%struct2 = type { %struct1 }
	%struct1 = type { int, void (%struct2*)*, %myint *, int (uint *)* }

implementation   ; Functions:

internal void %f1 (%struct1* %tty) {
loopentry.preheader:

  ; <void (%struct2*, ubyte)**> [#uses=1]
  ; <void (%struct2*, ubyte)*> [#uses=0]

	%tmp.2.i.i = getelementptr %struct1* %tty, uint 0, uint 1
	%tmp.3.i.i = volatile load void (%struct2*)** %tmp.2.i.i

	ret void
}

