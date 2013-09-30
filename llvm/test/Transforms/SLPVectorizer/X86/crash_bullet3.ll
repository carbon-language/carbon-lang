; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%class.btVector3.23.221.463.485.507.573.595.683.727.749.815.837.991.1585.1607.1629.1651.1849.2047.2069.2091.2113 = type { [4 x float] }

; Function Attrs: ssp uwtable
define void @_ZN11HullLibrary15CleanupVerticesEjPK9btVector3jRjPS0_fRS0_(%class.btVector3.23.221.463.485.507.573.595.683.727.749.815.837.991.1585.1607.1629.1651.1849.2047.2069.2091.2113* %vertices) #0 align 2 {
entry:
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %if.end22.2, %if.end
  br i1 undef, label %if.then17.1, label %if.end22.1

for.end36:                                        ; preds = %if.end22.2
  br label %for.body144

for.body144:                                      ; preds = %for.body144, %for.end36
  br i1 undef, label %for.end227, label %for.body144

for.end227:                                       ; preds = %for.body144
  br i1 undef, label %for.end271, label %for.body233

for.body233:                                      ; preds = %for.body233, %for.end227
  br i1 undef, label %for.body233, label %for.end271

for.end271:                                       ; preds = %for.body233, %for.end227
  %0 = phi float [ 0x47EFFFFFE0000000, %for.end227 ], [ undef, %for.body233 ]
  %1 = phi float [ 0x47EFFFFFE0000000, %for.end227 ], [ undef, %for.body233 ]
  %sub275 = fsub float undef, %1
  %sub279 = fsub float undef, %0
  br i1 undef, label %if.then291, label %return

if.then291:                                       ; preds = %for.end271
  %mul292 = fmul float %sub275, 5.000000e-01
  %add294 = fadd float %1, %mul292
  %mul295 = fmul float %sub279, 5.000000e-01
  %add297 = fadd float %0, %mul295
  br i1 undef, label %if.end332, label %if.else319

if.else319:                                       ; preds = %if.then291
  br i1 undef, label %if.then325, label %if.end327

if.then325:                                       ; preds = %if.else319
  br label %if.end327

if.end327:                                        ; preds = %if.then325, %if.else319
  br i1 undef, label %if.then329, label %if.end332

if.then329:                                       ; preds = %if.end327
  br label %if.end332

if.end332:                                        ; preds = %if.then329, %if.end327, %if.then291
  %dx272.1 = phi float [ %sub275, %if.then329 ], [ %sub275, %if.end327 ], [ 0x3F847AE140000000, %if.then291 ]
  %dy276.1 = phi float [ undef, %if.then329 ], [ undef, %if.end327 ], [ 0x3F847AE140000000, %if.then291 ]
  %sub334 = fsub float %add294, %dx272.1
  %sub338 = fsub float %add297, %dy276.1
  %arrayidx.i.i606 = getelementptr inbounds %class.btVector3.23.221.463.485.507.573.595.683.727.749.815.837.991.1585.1607.1629.1651.1849.2047.2069.2091.2113* %vertices, i64 0, i32 0, i64 0
  store float %sub334, float* %arrayidx.i.i606, align 4
  %arrayidx3.i607 = getelementptr inbounds %class.btVector3.23.221.463.485.507.573.595.683.727.749.815.837.991.1585.1607.1629.1651.1849.2047.2069.2091.2113* %vertices, i64 0, i32 0, i64 1
  store float %sub338, float* %arrayidx3.i607, align 4
  br label %return

return:                                           ; preds = %if.end332, %for.end271, %entry
  ret void

if.then17.1:                                      ; preds = %for.body
  br label %if.end22.1

if.end22.1:                                       ; preds = %if.then17.1, %for.body
  br i1 undef, label %if.then17.2, label %if.end22.2

if.then17.2:                                      ; preds = %if.end22.1
  br label %if.end22.2

if.end22.2:                                       ; preds = %if.then17.2, %if.end22.1
  br i1 undef, label %for.end36, label %for.body
}

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
