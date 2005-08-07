; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | %prcontext alloca 1 | grep 'int %foo'

declare void %bar(int*)
int %foo() {
  %A = alloca int             ;; Should stay in entry block because of 'tail' marker
  store int 17, int* %A
  call void %bar(int* %A)

  %X = tail call int %foo()
  ret int %X
}
