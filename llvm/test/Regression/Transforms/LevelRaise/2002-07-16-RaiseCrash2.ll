; This crashes raise, with an cast<> failure

; RUN: as < %s | opt -raise

implementation
sbyte* %test(int* %ptr) {
  %A = cast int* %ptr to sbyte *
  %B = add sbyte* %A, %A
  ret sbyte * %B
}
