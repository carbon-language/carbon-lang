; This crashes raise, with an cast<> failure

; RUN: llvm-as < %s | opt -raise

implementation
sbyte* %test(int* %ptr) {
  %A = cast int* %ptr to sbyte *
  %A = cast sbyte* %A to ulong
  %B = add ulong %A, %A
  %B = cast ulong %B to sbyte* 
  ret sbyte * %B
}
