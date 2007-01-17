; RUN: llvm-upgrade < %s | llvm-as | llc

int %main() {  
  ret int 42
}
