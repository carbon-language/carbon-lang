; RUN: llvm-as < %s | llc

int %foo(int %x) {
  ret int 42
}

int %main() {  
  %r = call int %foo(int 15) 
  ret int %r
}
