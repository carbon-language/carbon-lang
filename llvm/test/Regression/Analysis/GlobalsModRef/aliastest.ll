; RUN: llvm-as < %s | opt -globalsmodref-aa -load-vn -gcse | llvm-dis | not grep load
%X = internal global int 4

int %test(int *%P) {
  store int 7, int* %P
  store int 12,  int* %X   ;; cannot alias P, X's addr isn't taken
  %V = load int* %P
  ret int %V
}
