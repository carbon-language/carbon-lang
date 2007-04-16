; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep fabs | wc -l | grep 2
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep fmscs | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep fcvt | wc -l | grep 2
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep fuito | wc -l | grep 2
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep fto.i | wc -l | grep 4
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep bmi | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep bgt | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep fcmpezs | wc -l | grep 1

void %test(float *%P, double* %D) {
  %A = load float* %P
  %B = load double* %D
  store float %A, float* %P
  store double %B, double* %D
  ret void
}

declare float %fabsf(float)
declare double %fabs(double)

void %test_abs(float *%P, double* %D) {
  %a = load float* %P
  %b = call float %fabsf(float %a)
  store float %b, float* %P

  %A = load double* %D
  %B = call double %fabs(double %A)
  store double %B, double* %D
  ret void
}

void %test_add(float *%P, double* %D) {
  %a = load float* %P
  %b = add float %a, %a
  store float %b, float* %P

  %A = load double* %D
  %B = add double %A, %A
  store double %B, double* %D
  ret void
}

void %test_ext_round(float *%P, double* %D) {
  %a = load float* %P
  %b = cast float %a to double

  %A = load double* %D
  %B = cast double %A to float

  store double %b, double* %D
  store float %B, float* %P
  ret void
}

void %test_fma(float *%P1, float* %P2, float *%P3) {
  %a1 = load float* %P1
  %a2 = load float* %P2
  %a3 = load float* %P3

  %X = mul float %a1, %a2
  %Y = sub float %X, %a3

  store float %Y, float* %P1
  ret void
}

int %test_ftoi(float *%P1) {
  %a1 = load float* %P1
  %b1 = cast float %a1 to int
  ret int %b1
}

uint %test_ftou(float *%P1) {
  %a1 = load float* %P1
  %b1 = cast float %a1 to uint
  ret uint %b1
}

int %test_dtoi(double *%P1) {
  %a1 = load double* %P1
  %b1 = cast double %a1 to int
  ret int %b1
}

uint %test_dtou(double *%P1) {
  %a1 = load double* %P1
  %b1 = cast double %a1 to uint
  ret uint %b1
}

void %test_utod(double *%P1, uint %X) {
  %b1 = cast uint %X to double
  store double %b1, double* %P1
  ret void
}

void %test_utod2(double *%P1, ubyte %X) {
  %b1 = cast ubyte %X to double
  store double %b1, double* %P1
  ret void
}

void %test_cmp(float* %glob, int %X) {
entry:
        %tmp = load float* %glob                ; <float> [#uses=2]
        %tmp3 = getelementptr float* %glob, int 2               ; <float*> [#uses=1]
        %tmp4 = load float* %tmp3               ; <float> [#uses=2]
        %tmp = seteq float %tmp, %tmp4          ; <bool> [#uses=1]
        %tmp5 = tail call bool %llvm.isunordered.f32( float %tmp, float %tmp4 )         ; <bool> [#uses=1]
        %tmp6 = or bool %tmp, %tmp5             ; <bool> [#uses=1]
        br bool %tmp6, label %cond_true, label %cond_false

cond_true:              ; preds = %entry
        %tmp = tail call int (...)* %bar( )             ; <int> [#uses=0]
        ret void

cond_false:             ; preds = %entry
        %tmp7 = tail call int (...)* %baz( )            ; <int> [#uses=0]
        ret void
}

declare bool %llvm.isunordered.f32(float, float)

declare int %bar(...)

declare int %baz(...)

void %test_cmpfp0(float* %glob, int %X) {
entry:
        %tmp = load float* %glob                ; <float> [#uses=1]
        %tmp = setgt float %tmp, 0.000000e+00           ; <bool> [#uses=1]
        br bool %tmp, label %cond_true, label %cond_false

cond_true:              ; preds = %entry
        %tmp = tail call int (...)* %bar( )             ; <int> [#uses=0]
        ret void

cond_false:             ; preds = %entry
        %tmp1 = tail call int (...)* %baz( )            ; <int> [#uses=0]
        ret void
}

