; RUN: llvm-as < %s | opt -ds-aa -load-vn -gcse | llvm-dis | grep 'load int\* %L'

%G = internal global int* null

int %caller(bool %P) {
  %L = alloca int
  call void %callee(bool %P, int* %L)

  ;; At this point, G could point to L, so we can't eliminate these operations.
  %GP = load int** %G
  store int 17, int* %L
  store int 18, int* %GP  ;; might clober L

  %A = load int* %L    ;; is not necessarily 17!
  ret int %A
}

internal void %callee(bool %Cond, int* %P) {
  br bool %Cond, label %T, label %F
T:
  store int* %P, int** %G
  ret void
F:
  ret void
}

