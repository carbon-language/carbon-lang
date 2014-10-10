; RUN: llc -march=r600 -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt -enable-misched < %s | FileCheck -check-prefix=SI %s


@lds = addrspace(3) global [512 x float] zeroinitializer, align 4


; SI-LABEL: @simple_write2st64_one_val_f32_0_1
; SI-DAG: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]]
; SI-DAG: V_LSHLREV_B32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}
; SI: DS_WRITE2ST64_B32 [[VPTR]], [[VAL]], [[VAL]] offset0:0 offset1:1 [M0]
; SI: S_ENDPGM
define void @simple_write2st64_one_val_f32_0_1(float addrspace(1)* %C, float addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %in.gep = getelementptr float addrspace(1)* %in, i32 %x.i
  %val = load float addrspace(1)* %in.gep, align 4
  %arrayidx0 = getelementptr inbounds [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 64
  %arrayidx1 = getelementptr inbounds [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; SI-LABEL: @simple_write2st64_two_val_f32_2_5
; SI-DAG: BUFFER_LOAD_DWORD [[VAL0:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: BUFFER_LOAD_DWORD [[VAL1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:0x4
; SI-DAG: V_LSHLREV_B32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}
; SI: DS_WRITE2ST64_B32 [[VPTR]], [[VAL0]], [[VAL1]] offset0:2 offset1:5 [M0]
; SI: S_ENDPGM
define void @simple_write2st64_two_val_f32_2_5(float addrspace(1)* %C, float addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %in.gep.0 = getelementptr float addrspace(1)* %in, i32 %x.i
  %in.gep.1 = getelementptr float addrspace(1)* %in.gep.0, i32 1
  %val0 = load float addrspace(1)* %in.gep.0, align 4
  %val1 = load float addrspace(1)* %in.gep.1, align 4
  %add.x.0 = add nsw i32 %x.i, 128
  %arrayidx0 = getelementptr inbounds [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x.0
  store float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x.1 = add nsw i32 %x.i, 320
  %arrayidx1 = getelementptr inbounds [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x.1
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; SI-LABEL: @simple_write2st64_two_val_max_offset_f32
; SI-DAG: BUFFER_LOAD_DWORD [[VAL0:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: BUFFER_LOAD_DWORD [[VAL1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:0x4
; SI-DAG: V_LSHLREV_B32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}
; SI: DS_WRITE2ST64_B32 [[VPTR]], [[VAL0]], [[VAL1]] offset0:0 offset1:255 [M0]
; SI: S_ENDPGM
define void @simple_write2st64_two_val_max_offset_f32(float addrspace(1)* %C, float addrspace(1)* %in, float addrspace(3)* %lds) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %in.gep.0 = getelementptr float addrspace(1)* %in, i32 %x.i
  %in.gep.1 = getelementptr float addrspace(1)* %in.gep.0, i32 1
  %val0 = load float addrspace(1)* %in.gep.0, align 4
  %val1 = load float addrspace(1)* %in.gep.1, align 4
  %arrayidx0 = getelementptr inbounds float addrspace(3)* %lds, i32 %x.i
  store float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 16320
  %arrayidx1 = getelementptr inbounds float addrspace(3)* %lds, i32 %add.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; SI-LABEL: @simple_write2st64_two_val_max_offset_f64
; SI-DAG: BUFFER_LOAD_DWORDX2 [[VAL0:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: BUFFER_LOAD_DWORDX2 [[VAL1:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:0x8
; SI-DAG: V_ADD_I32_e32 [[VPTR:v[0-9]+]],
; SI: DS_WRITE2ST64_B64 [[VPTR]], [[VAL0]], [[VAL1]] offset0:4 offset1:127 [M0]
; SI: S_ENDPGM
define void @simple_write2st64_two_val_max_offset_f64(double addrspace(1)* %C, double addrspace(1)* %in, double addrspace(3)* %lds) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %in.gep.0 = getelementptr double addrspace(1)* %in, i32 %x.i
  %in.gep.1 = getelementptr double addrspace(1)* %in.gep.0, i32 1
  %val0 = load double addrspace(1)* %in.gep.0, align 8
  %val1 = load double addrspace(1)* %in.gep.1, align 8
  %add.x.0 = add nsw i32 %x.i, 256
  %arrayidx0 = getelementptr inbounds double addrspace(3)* %lds, i32 %add.x.0
  store double %val0, double addrspace(3)* %arrayidx0, align 8
  %add.x.1 = add nsw i32 %x.i, 8128
  %arrayidx1 = getelementptr inbounds double addrspace(3)* %lds, i32 %add.x.1
  store double %val1, double addrspace(3)* %arrayidx1, align 8
  ret void
}

; SI-LABEL: @byte_size_only_divisible_64_write2st64_f64
; SI-NOT: DS_WRITE2ST64_B64
; SI: DS_WRITE2_B64 {{v[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}} offset0:0 offset1:8
; SI: S_ENDPGM
define void @byte_size_only_divisible_64_write2st64_f64(double addrspace(1)* %C, double addrspace(1)* %in, double addrspace(3)* %lds) #0 {
  %x.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %in.gep = getelementptr double addrspace(1)* %in, i32 %x.i
  %val = load double addrspace(1)* %in.gep, align 8
  %arrayidx0 = getelementptr inbounds double addrspace(3)* %lds, i32 %x.i
  store double %val, double addrspace(3)* %arrayidx0, align 8
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds double addrspace(3)* %lds, i32 %add.x
  store double %val, double addrspace(3)* %arrayidx1, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.y() #1

; Function Attrs: noduplicate nounwind
declare void @llvm.AMDGPU.barrier.local() #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { noduplicate nounwind }
