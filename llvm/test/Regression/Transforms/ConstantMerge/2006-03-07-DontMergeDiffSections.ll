; RUN: llvm-as < %s | opt -constmerge | llvm-dis | grep foo
; RUN: llvm-as < %s | opt -constmerge | llvm-dis | grep bar

; Don't merge constants in different sections.

%G1 = internal constant int 1, section "foo"
%G2 = internal constant int 1, section "bar"
%G3 = internal constant int 1, section "bar"

void %test(int** %P1, int **%P2, int **%P3) {
  store int* %G1, int** %P1
  store int* %G2, int** %P2
  store int* %G3, int** %P3
  ret void
}
