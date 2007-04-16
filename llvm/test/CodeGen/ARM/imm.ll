; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | not grep CPI

int %test1(int %A) {
  %B = add int %A, -268435441  ; 0xF000000F
  ret int %B
}

int %test2() {
  ret int 65533
}

int %test3(int %A) {
  %B = or int %A, 65533
  ret int %B
}


