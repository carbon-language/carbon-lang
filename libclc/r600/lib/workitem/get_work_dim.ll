declare i32 @llvm.AMDGPU.read.workdim() nounwind readnone

define i32 @get_work_dim() nounwind readnone alwaysinline {
  %x = call i32 @llvm.AMDGPU.read.workdim() nounwind readnone , !range !0
  ret i32 %x
}

!0 = metadata !{ i8 1, i8 2, i8 3 }
