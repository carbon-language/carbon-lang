; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.Ray.5.11.53.113.119.137.149.185.329.389.416 = type { %struct.Vec.0.6.48.108.114.132.144.180.324.384.414, %struct.Vec.0.6.48.108.114.132.144.180.324.384.414 }
%struct.Vec.0.6.48.108.114.132.144.180.324.384.414 = type { double, double, double }

; Function Attrs: ssp uwtable
define void @main() #0 {
entry:
  br i1 undef, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  unreachable

cond.end:                                         ; preds = %entry
  br label %invoke.cont

invoke.cont:                                      ; preds = %invoke.cont, %cond.end
  br i1 undef, label %arrayctor.cont, label %invoke.cont

arrayctor.cont:                                   ; preds = %invoke.cont
  %agg.tmp99208.sroa.0.0.idx = getelementptr inbounds %struct.Ray.5.11.53.113.119.137.149.185.329.389.416, %struct.Ray.5.11.53.113.119.137.149.185.329.389.416* undef, i64 0, i32 0, i32 0
  %agg.tmp99208.sroa.1.8.idx388 = getelementptr inbounds %struct.Ray.5.11.53.113.119.137.149.185.329.389.416, %struct.Ray.5.11.53.113.119.137.149.185.329.389.416* undef, i64 0, i32 0, i32 1
  %agg.tmp101211.sroa.0.0.idx = getelementptr inbounds %struct.Ray.5.11.53.113.119.137.149.185.329.389.416, %struct.Ray.5.11.53.113.119.137.149.185.329.389.416* undef, i64 0, i32 1, i32 0
  %agg.tmp101211.sroa.1.8.idx390 = getelementptr inbounds %struct.Ray.5.11.53.113.119.137.149.185.329.389.416, %struct.Ray.5.11.53.113.119.137.149.185.329.389.416* undef, i64 0, i32 1, i32 1
  br label %for.cond36.preheader

for.cond36.preheader:                             ; preds = %_Z5clampd.exit.1, %arrayctor.cont
  br i1 undef, label %for.body42.lr.ph.us, label %_Z5clampd.exit.1

cond.false51.us:                                  ; preds = %for.body42.lr.ph.us
  unreachable

cond.true48.us:                                   ; preds = %for.body42.lr.ph.us
  br i1 undef, label %cond.true63.us, label %cond.false66.us

cond.false66.us:                                  ; preds = %cond.true48.us
  %add.i276.us = fadd double 0.000000e+00, undef
  %add.i264.us = fadd double %add.i276.us, 0.000000e+00
  %add4.i267.us = fadd double undef, 0xBFA5CC2D1960285F
  %mul.i254.us = fmul double %add.i264.us, 1.400000e+02
  %mul2.i256.us = fmul double %add4.i267.us, 1.400000e+02
  %add.i243.us = fadd double %mul.i254.us, 5.000000e+01
  %add4.i246.us = fadd double %mul2.i256.us, 5.200000e+01
  %mul.i.i.us = fmul double undef, %add.i264.us
  %mul2.i.i.us = fmul double undef, %add4.i267.us
  store double %add.i243.us, double* %agg.tmp99208.sroa.0.0.idx, align 8
  store double %add4.i246.us, double* %agg.tmp99208.sroa.1.8.idx388, align 8
  store double %mul.i.i.us, double* %agg.tmp101211.sroa.0.0.idx, align 8
  store double %mul2.i.i.us, double* %agg.tmp101211.sroa.1.8.idx390, align 8
  unreachable

cond.true63.us:                                   ; preds = %cond.true48.us
  unreachable

for.body42.lr.ph.us:                              ; preds = %for.cond36.preheader
  br i1 undef, label %cond.true48.us, label %cond.false51.us

_Z5clampd.exit.1:                                 ; preds = %for.cond36.preheader
  br label %for.cond36.preheader
}


%struct.Ray.5.11.53.95.137.191.197.203.239.257.263.269.275.281.287.293.383.437.443.455.461.599.601 = type { %struct.Vec.0.6.48.90.132.186.192.198.234.252.258.264.270.276.282.288.378.432.438.450.456.594.600, %struct.Vec.0.6.48.90.132.186.192.198.234.252.258.264.270.276.282.288.378.432.438.450.456.594.600 }
%struct.Vec.0.6.48.90.132.186.192.198.234.252.258.264.270.276.282.288.378.432.438.450.456.594.600 = type { double, double, double }

define void @_Z8radianceRK3RayiPt() #0 {
entry:
  br i1 undef, label %if.then78, label %if.then38

if.then38:                                        ; preds = %entry
  %mul.i.i790 = fmul double undef, undef
  %mul3.i.i792 = fmul double undef, undef
  %mul.i764 = fmul double undef, %mul3.i.i792
  %mul4.i767 = fmul double undef, undef
  %sub.i768 = fsub double %mul.i764, %mul4.i767
  %mul6.i770 = fmul double undef, %mul.i.i790
  %mul9.i772 = fmul double undef, %mul3.i.i792
  %sub10.i773 = fsub double %mul6.i770, %mul9.i772
  %mul.i736 = fmul double undef, %sub.i768
  %mul2.i738 = fmul double undef, %sub10.i773
  %mul.i727 = fmul double undef, %mul.i736
  %mul2.i729 = fmul double undef, %mul2.i738
  %add.i716 = fadd double undef, %mul.i727
  %add4.i719 = fadd double undef, %mul2.i729
  %add.i695 = fadd double undef, %add.i716
  %add4.i698 = fadd double undef, %add4.i719
  %mul.i.i679 = fmul double undef, %add.i695
  %mul2.i.i680 = fmul double undef, %add4.i698
  %agg.tmp74663.sroa.0.0.idx = getelementptr inbounds %struct.Ray.5.11.53.95.137.191.197.203.239.257.263.269.275.281.287.293.383.437.443.455.461.599.601, %struct.Ray.5.11.53.95.137.191.197.203.239.257.263.269.275.281.287.293.383.437.443.455.461.599.601* undef, i64 0, i32 1, i32 0
  store double %mul.i.i679, double* %agg.tmp74663.sroa.0.0.idx, align 8
  %agg.tmp74663.sroa.1.8.idx943 = getelementptr inbounds %struct.Ray.5.11.53.95.137.191.197.203.239.257.263.269.275.281.287.293.383.437.443.455.461.599.601, %struct.Ray.5.11.53.95.137.191.197.203.239.257.263.269.275.281.287.293.383.437.443.455.461.599.601* undef, i64 0, i32 1, i32 1
  store double %mul2.i.i680, double* %agg.tmp74663.sroa.1.8.idx943, align 8
  br label %return

if.then78:                                        ; preds = %entry
  br label %return

return:                                           ; preds = %if.then78, %if.then38
  ret void
}

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
