; RUN: llvm-as < %s | opt -ds-aa -load-vn -gcse -instcombine | llvm-dis | not grep sub

void %bar(int* %p) {
  ret void
}

int %foo(int* %a) {
  %b = load int* %a
  call void %bar(int* %a)
  %d = load int* %a
  %e = sub int %b, %d
  ret int %e
}
