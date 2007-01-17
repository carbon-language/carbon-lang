; RUN: llvm-upgrade < %s | llvm-as | llc

int %foo(int %x) {
  ret int %x
}

int %main() {  
  %r = call int %foo(int 0) 
  ret int %r
}
