; RUN: llvm-upgrade < %s | llvm-as | llc

void %foo() {
  ret void
}

int %main() {  
  call void ()* %foo() 
  ret int 0
}
