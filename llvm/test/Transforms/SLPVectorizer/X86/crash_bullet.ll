; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960" = type { i32, i32 }

define void @_ZN23btGeneric6DofConstraint8getInfo1EPN17btTypedConstraint17btConstraintInfo1E(%"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960"* nocapture %info) {
entry:
  br i1 undef, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  ret void

if.else:                                          ; preds = %entry
  %m_numConstraintRows4 = getelementptr inbounds %"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960", %"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960"* %info, i64 0, i32 0
  %nub5 = getelementptr inbounds %"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960", %"struct.btTypedConstraint::btConstraintInfo1.17.157.357.417.477.960"* %info, i64 0, i32 1
  br i1 undef, label %land.lhs.true.i.1, label %if.then7.1

land.lhs.true.i.1:                                ; preds = %if.else
  br i1 undef, label %for.inc.1, label %if.then7.1

if.then7.1:                                       ; preds = %land.lhs.true.i.1, %if.else
  %inc.1 = add nsw i32 0, 1
  store i32 %inc.1, i32* %m_numConstraintRows4, align 4
  %dec.1 = add nsw i32 6, -1
  store i32 %dec.1, i32* %nub5, align 4
  br label %for.inc.1

for.inc.1:                                        ; preds = %if.then7.1, %land.lhs.true.i.1
  %0 = phi i32 [ %dec.1, %if.then7.1 ], [ 6, %land.lhs.true.i.1 ]
  %1 = phi i32 [ %inc.1, %if.then7.1 ], [ 0, %land.lhs.true.i.1 ]
  %inc.2 = add nsw i32 %1, 1
  store i32 %inc.2, i32* %m_numConstraintRows4, align 4
  %dec.2 = add nsw i32 %0, -1
  store i32 %dec.2, i32* %nub5, align 4
  unreachable
}

%class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332 = type { float, [3 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330], [3 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330], %class.btVector4.7.32.67.92.117.142.177.187.262.282.331, %class.btVector4.7.32.67.92.117.142.177.187.262.282.331, %class.btVector3.5.30.65.90.115.140.175.185.260.280.330, %class.btVector3.5.30.65.90.115.140.175.185.260.280.330, %class.btVector3.5.30.65.90.115.140.175.185.260.280.330, %class.btVector3.5.30.65.90.115.140.175.185.260.280.330, [4 x float], float, float, [4 x float], float, float, [16 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330], [16 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330], [16 x %class.btVector3.5.30.65.90.115.140.175.185.260.280.330] }
%class.btVector3.5.30.65.90.115.140.175.185.260.280.330 = type { [4 x float] }
%class.btVector4.7.32.67.92.117.142.177.187.262.282.331 = type { %class.btVector3.5.30.65.90.115.140.175.185.260.280.330 }

define void @_ZN30GIM_TRIANGLE_CALCULATION_CACHE18triangle_collisionERK9btVector3S2_S2_fS2_S2_S2_fR25GIM_TRIANGLE_CONTACT_DATA(%class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332* %this) {
entry:
  %arrayidx26 = getelementptr inbounds %class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332, %class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332* %this, i64 0, i32 2, i64 0, i32 0, i64 1
  %arrayidx36 = getelementptr inbounds %class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332, %class.GIM_TRIANGLE_CALCULATION_CACHE.9.34.69.94.119.144.179.189.264.284.332* %this, i64 0, i32 2, i64 0, i32 0, i64 2
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

define void @_Z8dBoxBox2RK9btVector3PKfS1_S1_S3_S1_RS_PfPiiP12dContactGeomiRN36btDiscreteCollisionDetectorInterface6ResultE() {
entry:
  %add8.i2343 = fadd float undef, undef
  %add8.i2381 = fadd float undef, undef
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %return, label %if.end111

if.end111:                                        ; preds = %if.end
  br i1 undef, label %return, label %if.end136

if.end136:                                        ; preds = %if.end111
  br i1 undef, label %return, label %if.end162

if.end162:                                        ; preds = %if.end136
  br i1 undef, label %return, label %if.end189

if.end189:                                        ; preds = %if.end162
  br i1 undef, label %return, label %if.end216

if.end216:                                        ; preds = %if.end189
  br i1 undef, label %if.then218, label %if.end225

if.then218:                                       ; preds = %if.end216
  br label %if.end225

if.end225:                                        ; preds = %if.then218, %if.end216
  br i1 undef, label %return, label %if.end248

if.end248:                                        ; preds = %if.end225
  br i1 undef, label %return, label %if.end304

if.end304:                                        ; preds = %if.end248
  %mul341 = fmul float undef, %add8.i2343
  %mul344 = fmul float undef, %add8.i2381
  %sub345 = fsub float %mul341, %mul344
  br i1 undef, label %return, label %if.end361

if.end361:                                        ; preds = %if.end304
  %mul364 = fmul float %add8.i2381, %add8.i2381
  br i1 undef, label %if.then370, label %if.end395

if.then370:                                       ; preds = %if.end361
  br i1 undef, label %if.then374, label %if.end395

if.then374:                                       ; preds = %if.then370
  %cmp392 = fcmp olt float %sub345, 0.000000e+00
  br label %if.end395

if.end395:                                        ; preds = %if.then374, %if.then370, %if.end361
  unreachable

return:                                           ; preds = %if.end304, %if.end248, %if.end225, %if.end189, %if.end162, %if.end136, %if.end111, %if.end, %entry
  ret void
}
