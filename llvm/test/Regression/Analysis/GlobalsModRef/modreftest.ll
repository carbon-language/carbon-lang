; RUN: llvm-as < %s | opt -globalsmodref-aa -load-vn -gcse | llvm-dis | not grep load
%X = internal global int 4

int %test(int *%P) {
  store int 12,  int* %X
  call void %doesnotmodX()
  %V = load int* %X
  ret int %V
}

void %doesnotmodX() {
  ret void
}
