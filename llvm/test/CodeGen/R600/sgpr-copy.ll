; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s

; This test checks that no VGPR to SGPR copies are created by the register
; allocator.
; CHECK: @main
; CHECK: S_BUFFER_LOAD_DWORD [[DST:SGPR[0-9]]], {{[SGPR_[0-9]+}}, 0
; CHECK: V_MOV_B32_e32 VGPR{{[0-9]}}, [[DST]]

define void @main(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #0 {
main_body:
  %20 = getelementptr <16 x i8> addrspace(2)* %0, i32 0
  %21 = load <16 x i8> addrspace(2)* %20, !tbaa !0
  %22 = call float @llvm.SI.load.const(<16 x i8> %21, i32 0)
  %23 = call float @llvm.SI.load.const(<16 x i8> %21, i32 16)
  %24 = call float @llvm.SI.load.const(<16 x i8> %21, i32 32)
  %25 = fptosi float %23 to i32
  %26 = icmp ne i32 %25, 0
  br i1 %26, label %ENDIF, label %ELSE

ELSE:                                             ; preds = %main_body
  %27 = fsub float -0.000000e+00, %22
  br label %ENDIF

ENDIF:                                            ; preds = %main_body, %ELSE
  %temp.0 = phi float [ %27, %ELSE ], [ %22, %main_body ]
  %28 = fadd float %temp.0, %24
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %28, float %28, float 0.000000e+00, float 1.000000e+00)
  ret void
}

; We just want ot make sure the program doesn't crash
; CHECK: @loop

define void @loop(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #0 {
main_body:
  %20 = getelementptr <16 x i8> addrspace(2)* %0, i32 0
  %21 = load <16 x i8> addrspace(2)* %20, !tbaa !0
  %22 = call float @llvm.SI.load.const(<16 x i8> %21, i32 0)
  %23 = call float @llvm.SI.load.const(<16 x i8> %21, i32 4)
  %24 = call float @llvm.SI.load.const(<16 x i8> %21, i32 8)
  %25 = call float @llvm.SI.load.const(<16 x i8> %21, i32 12)
  %26 = fptosi float %25 to i32
  %27 = bitcast i32 %26 to float
  %28 = bitcast float %27 to i32
  br label %LOOP

LOOP:                                             ; preds = %ENDIF, %main_body
  %temp4.0 = phi float [ %22, %main_body ], [ %temp5.0, %ENDIF ]
  %temp5.0 = phi float [ %23, %main_body ], [ %temp6.0, %ENDIF ]
  %temp6.0 = phi float [ %24, %main_body ], [ %temp4.0, %ENDIF ]
  %temp8.0 = phi float [ 0.000000e+00, %main_body ], [ %37, %ENDIF ]
  %29 = bitcast float %temp8.0 to i32
  %30 = icmp sge i32 %29, %28
  %31 = sext i1 %30 to i32
  %32 = bitcast i32 %31 to float
  %33 = bitcast float %32 to i32
  %34 = icmp ne i32 %33, 0
  br i1 %34, label %IF, label %ENDIF

IF:                                               ; preds = %LOOP
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %temp4.0, float %temp5.0, float %temp6.0, float 1.000000e+00)
  ret void

ENDIF:                                            ; preds = %LOOP
  %35 = bitcast float %temp8.0 to i32
  %36 = add i32 %35, 1
  %37 = bitcast i32 %36 to float
  br label %LOOP
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

; Function Attrs: readonly
declare float @fabs(float) #2

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
attributes #2 = { readonly }

!0 = metadata !{metadata !"const", null, i32 1}

