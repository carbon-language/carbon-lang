; RUN: llvm-as -f %s -o %t.bc
; RUN: lli %t.bc > /dev/null

; This testcase failed to work because two variable sized allocas confused the
; local register allocator.

int %main(uint %X) {
  %A = alloca uint, uint %X

  %B = alloca float, uint %X
  ret int 0
}
