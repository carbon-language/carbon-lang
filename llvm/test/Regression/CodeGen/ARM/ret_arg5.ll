; RUN: llvm-as < %s | llc -march=arm
; XFAIL: *
int %test(int %a1, int %a2, int %a3, int %a4, int %a5) {
  ret int %a5
}
