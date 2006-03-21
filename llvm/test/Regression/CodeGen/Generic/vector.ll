; Test that vectors are scalarized/lowered correctly.
; RUN: llvm-as < %s | llc && 
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g3

%f1 = type <1 x float>
%f2 = type <2 x float>
%f4 = type <4 x float>
%i4 = type <4 x int>
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

void %test_fmul(%f8 *%P, %f8* %Q, %f8 *%S) {
  %p = load %f8* %P
  %q = load %f8* %Q
  %R = mul %f8 %p, %q
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

void %test_scalar_to_vector(float %F, %f4 *%S) {
  %R = insertelement %f4 undef, float %F, uint 0   ;; R = scalar_to_vector F
  store %f4 %R, %f4 *%S
  ret void
}

;;; TEST IMPORTANT IDIOMS

void %splat(%f4* %P, %f4* %Q, float %X) {
        %tmp = insertelement %f4 undef, float %X, uint 0
        %tmp2 = insertelement %f4 %tmp, float %X, uint 1
        %tmp4 = insertelement %f4 %tmp2, float %X, uint 2
        %tmp6 = insertelement %f4 %tmp4, float %X, uint 3
	%q = load %f4* %Q
	%R = add %f4 %q, %tmp6
        store %f4 %R, %f4* %P
        ret void
}

void %splat_i4(%i4* %P, %i4* %Q, int %X) {
        %tmp = insertelement %i4 undef, int %X, uint 0
        %tmp2 = insertelement %i4 %tmp, int %X, uint 1
        %tmp4 = insertelement %i4 %tmp2, int %X, uint 2
        %tmp6 = insertelement %i4 %tmp4, int %X, uint 3
	%q = load %i4* %Q
	%R = add %i4 %q, %tmp6
        store %i4 %R, %i4* %P
        ret void
}

