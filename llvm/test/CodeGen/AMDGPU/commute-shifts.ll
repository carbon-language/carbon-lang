; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}main:
; SI: v_lshl_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; VI: v_lshlrev_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 1

define void @main() #0 {
main_body:
  %0 = fptosi float undef to i32
  %1 = call <4 x i32> @llvm.SI.imageload.v4i32(<4 x i32> undef, <32 x i8> undef, i32 2)
  %2 = extractelement <4 x i32> %1, i32 0
  %3 = and i32 %0, 7
  %4 = shl i32 1, %3
  %5 = and i32 %2, %4
  %6 = icmp eq i32 %5, 0
  %.10 = select i1 %6, float 0.000000e+00, float undef
  %7 = call i32 @llvm.SI.packf16(float undef, float %.10)
  %8 = bitcast i32 %7 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float undef, float %8, float undef, float %8)
  ret void
}

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.SI.imageload.v4i32(<4 x i32>, <32 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.SI.packf16(float, float) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" "enable-no-nans-fp-math"="true" }
attributes #1 = { nounwind readnone }
