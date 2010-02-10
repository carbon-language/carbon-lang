; RUN: llc < %s -march=arm -mattr=+v6,+vfp2 -stats -info-output-file - | grep "Number of re-materialization"

define arm_apcscc i32 @main(i32 %argc, i8** nocapture %argv, double %d1, double %d2) nounwind {
entry:
  br i1 undef, label %smvp.exit, label %bb.i3

bb.i3:                                            ; preds = %bb.i3, %bb134
  br i1 undef, label %smvp.exit, label %bb.i3

smvp.exit:                                        ; preds = %bb.i3
  %0 = fmul double %d1, 2.400000e-03            ; <double> [#uses=2]
  br i1 undef, label %bb138.preheader, label %bb159

bb138.preheader:                                  ; preds = %smvp.exit
  br label %bb138

bb138:                                            ; preds = %bb138, %bb138.preheader
  br i1 undef, label %bb138, label %bb145.loopexit

bb142:                                            ; preds = %bb.nph218.bb.nph218.split_crit_edge, %phi0.exit
  %1 = fmul double %d1, -1.200000e-03           ; <double> [#uses=1]
  %2 = fadd double %d2, %1                      ; <double> [#uses=1]
  %3 = fmul double %2, %d2                      ; <double> [#uses=1]
  %4 = fsub double 0.000000e+00, %3               ; <double> [#uses=1]
  br i1 %14, label %phi1.exit, label %bb.i35

bb.i35:                                           ; preds = %bb142
  %5 = call arm_apcscc  double @sin(double %15) nounwind readonly ; <double> [#uses=1]
  %6 = fmul double %5, 0x4031740AFA84AD8A         ; <double> [#uses=1]
  %7 = fsub double 1.000000e+00, undef            ; <double> [#uses=1]
  %8 = fdiv double %7, 6.000000e-01               ; <double> [#uses=1]
  br label %phi1.exit

phi1.exit:                                        ; preds = %bb.i35, %bb142
  %.pn = phi double [ %6, %bb.i35 ], [ 0.000000e+00, %bb142 ] ; <double> [#uses=1]
  %9 = phi double [ %8, %bb.i35 ], [ 0.000000e+00, %bb142 ] ; <double> [#uses=1]
  %10 = fmul double %.pn, %9                      ; <double> [#uses=1]
  br i1 %14, label %phi0.exit, label %bb.i

bb.i:                                             ; preds = %phi1.exit
  unreachable

phi0.exit:                                        ; preds = %phi1.exit
  %11 = fsub double %4, %10                       ; <double> [#uses=1]
  %12 = fadd double 0.000000e+00, %11             ; <double> [#uses=1]
  store double %12, double* undef, align 4
  br label %bb142

bb145.loopexit:                                   ; preds = %bb138
  br i1 undef, label %bb.nph218.bb.nph218.split_crit_edge, label %bb159

bb.nph218.bb.nph218.split_crit_edge:              ; preds = %bb145.loopexit
  %13 = fmul double %0, 0x401921FB54442D18        ; <double> [#uses=1]
  %14 = fcmp ugt double %0, 6.000000e-01          ; <i1> [#uses=2]
  %15 = fdiv double %13, 6.000000e-01             ; <double> [#uses=1]
  br label %bb142

bb159:                                            ; preds = %bb145.loopexit, %smvp.exit, %bb134
  unreachable

bb166:                                            ; preds = %bb127
  unreachable
}

declare arm_apcscc double @sin(double) nounwind readonly
