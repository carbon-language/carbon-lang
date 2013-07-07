; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332 = type { float, [3 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330], [3 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330], %class.btVector4.7.32.67.92.117.142.177.187.262.282.331, %class.btVector4.7.32.67.92.117.142.177.187.262.282.331, %class.btVector3.5.30.65.90.115.140.175.185.260.280.330, %class.btVector3.5.30.65.90.115.140.175.185.260.280.330, %class.btVector3.5.30.65.90.115.140.175.185.260.280.330, %class.btVector3.5.30.65.90.115.140.175.185.260.280.330, [4 x float], float, float, [4 x float], float, float, [16 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330], [16 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330], [16 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330] }
%class.btVector3.5.30.65.90.115.140.175.185.260.280.330 = type { [4 x float] }
%class.btVector4.7.32.67.92.117.142.177.187.262.282.331 = type { %class.btVector3.5.30.65.90.115.140.175.185.260.280.330 }

define void @_ZN30GIM_TRIANGLE_CALCULATION_CACHE18triangle_collisionERK9btVector3S2_S2_fS2_S2_S2_fR25GIM_TRIANGLE_CONTACT_DATA(%class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332* %this) {
entry:
  %arrayidx26 = getelementptr inbounds %class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332* %this, i64 0, i32 2, i64 0, i32 0, i64 1
  %arrayidx36 = getelementptr inbounds %class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332* %this, i64 0, i32 2, i64 0, i32 0, i64 2
  %0 = load float* %arrayidx36, align 4
  %add587 = fadd float undef, undef
  %sub600 = fsub float %add587, undef
  store float %sub600, float* undef, align 4
  %sub613 = fsub float %add587, %sub600
  store float %sub613, float* %arrayidx26, align 4
  %add626 = fadd float %0, undef
  %sub639 = fsub float %add626, undef
  %sub652 = fsub float %add626, %sub639
  store float %sub652, float* %arrayidx36, align 4
  br i1 undef, label %if.else1609, label %if.then1595

if.then1595:                                      ; preds = %entry
  br i1 undef, label %return, label %for.body.lr.ph.i.i1702

for.body.lr.ph.i.i1702:                           ; preds = %if.then1595
  unreachable

if.else1609:                                      ; preds = %entry
  unreachable

return:                                           ; preds = %if.then1595
  ret void
}

