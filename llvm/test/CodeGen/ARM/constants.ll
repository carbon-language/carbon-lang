; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -o %t.s -f &&
; RUN: grep "mov r0, #0" %t.s     | wc -l | grep 1 &&
; RUN: grep "mov r0, #255" %t.s   | wc -l | grep 1 &&
; RUN: grep "mov r0, #256" %t.s   | wc -l | grep 1 &&
; RUN: grep "mov r0, #1" %t.s     | wc -l | grep 2 &&
; RUN: grep "orr r0, r0, #256" %t.s     | wc -l | grep 1 &&
; RUN: grep "mov r0, #-1073741761" %t.s | wc -l | grep 1 &&
; RUN: grep "mov r0, #1008" %t.s  | wc -l | grep 1 &&
; RUN: grep "cmp r0, #65536" %t.s | wc -l | grep 1 &&
; RUN: grep "\.comm.*a,4,4" %t.s  | wc -l | grep 1

%a = internal global int 0

uint %f1() {
  ret uint 0
}

uint %f2() {
  ret uint 255
}

uint %f3() {
  ret uint 256
}

uint %f4() {
  ret uint 257
}

uint %f5() {
  ret uint 3221225535
}

uint %f6() {
  ret uint 1008
}

void %f7(uint %a) {
entry:
	%b = setgt uint %a, 65536
	br bool %b, label %r, label %r

r:
	ret void
}
