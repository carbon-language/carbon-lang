; RUN: llvm-as < %s | llc

int %bar(int %x) {
  ret int 0
}

int %foo(int %x) {
  %q = call int %bar(int 1)
  ret int %q
}

int %main() {  
  %r = call int %foo(int 2) 
  ret int %r
}
