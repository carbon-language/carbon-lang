; RUN: llvm-as < %s | llc
; Test that vectors are scalarized/lowered correctly.

%f1 = type <1 x float>
%f2 = type <2 x float>
%f4 = type <4 x float>
%f8 = type <8 x float>

implementation

;;; TEST HANDLING OF VARIOUS VECTOR SIZES

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

;;; TEST VECTOR CONSTRUCTS

void %test_cst(%f4 *%P, %f4 *%S) {
  %p = load %f4* %P
  %R = add %f4 %p, <float 0.1, float 1.0, float 2.0, float 4.5>
  store %f4 %R, %f4 *%S
  ret void
}

void %test_zero(%f4 *%P, %f4 *%S) {
  %p = load %f4* %P
  %R = add %f4 %p, zeroinitializer
  store %f4 %R, %f4 *%S
  ret void
}

void %test_undef(%f4 *%P, %f4 *%S) {
  %p = load %f4* %P
  %R = add %f4 %p, undef
  store %f4 %R, %f4 *%S
  ret void
}

void %test_constant_insert(%f4 *%S) {
  %R = insertelement %f4 zeroinitializer, float 10.0, uint 0
  store %f4 %R, %f4 *%S
  ret void
}

void %test_variable_buildvector(float %F, %f4 *%S) {
  %R = insertelement %f4 zeroinitializer, float %F, uint 0
  store %f4 %R, %f4 *%S
  ret void
}
