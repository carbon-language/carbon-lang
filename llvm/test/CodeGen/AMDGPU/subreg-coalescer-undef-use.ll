; RUN: llc -march=amdgcn -mcpu=SI -o /dev/null %s
; Don't crash when the use of an undefined value is only detected by the
; register coalescer because it is hidden with subregister insert/extract.
target triple="amdgcn--"

define void @foobar(float %a0, float %a1, float addrspace(1)* %out) nounwind {
entry:
  %v0 = insertelement <4 x float> undef, float %a0, i32 0
  br i1 undef, label %ift, label %ife

ift:
  %v1 = insertelement <4 x float> undef, float %a1, i32 0
  br label %ife

ife:
  %val = phi <4 x float> [ %v1, %ift ], [ %v0, %entry ]
  %v2 = extractelement <4 x float> %val, i32 1
  store float %v2, float addrspace(1)* %out, align 4
  ret void
}
