; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

fastcc void %foo() {
  ret void
}

coldcc void %bar() {
  call fastcc void %foo()
  ret void
}


cc0 void %foo2() {
  ret void
}

coldcc void %bar2() {
  call fastcc void %foo()
  ret void
}

cc42 void %bar3() {
  invoke fastcc void %foo() to label %Ok unwind label %U
Ok:
  ret void
U:
  unwind
}


void %bar4() {
  call cc42 void %bar()
  invoke cc42 void %bar3() to label %Ok unwind label %U
Ok:
  ret void
U:
  unwind
}


