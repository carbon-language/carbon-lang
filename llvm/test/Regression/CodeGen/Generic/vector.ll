; RUN: llvm-as < %s | llc
; Test that vectors are scalarized/lowered correctly.

%f1 = type <1 x float>
%f2 = type <2 x float>
%f4 = type <4 x float>
%f8 = type <8 x float>

implementation

void %test_f1(%f1 *%P, %f1* %Q, %f1 *%S) {
  %p = load %f1 *%P
  %q = load %f1* %Q
  %R = add %f1 %p, %q
  store %f1 %R, %f1 *%S
  ret void
}

void %test_f2(%f2 *%P, %f2* %Q, %f2 *%S) {
  %p = load %f2* %P
  %q = load %f2* %Q
  %R = add %f2 %p, %q
  store %f2 %R, %f2 *%S
  ret void
}

void %test_f4(%f4 *%P, %f4* %Q, %f4 *%S) {
  %p = load %f4* %P
  %q = load %f4* %Q
  %R = add %f4 %p, %q
  store %f4 %R, %f4 *%S
  ret void
}

void %test_f8(%f8 *%P, %f8* %Q, %f8 *%S) {
  %p = load %f8* %P
  %q = load %f8* %Q
  %R = add %f8 %p, %q
  store %f8 %R, %f8 *%S
  ret void
}
