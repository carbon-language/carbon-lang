; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

declare <16 x float> @llvm.x86.avx512.gather.dps.512 (<16 x float>, i8*, <16 x i32>, i16, i32)
declare void @llvm.x86.avx512.scatter.dps.512 (i8*, i16, <16 x i32>, <16 x float>, i32)
declare <8 x double> @llvm.x86.avx512.gather.dpd.512 (<8 x double>, i8*, <8 x i32>, i8, i32)
declare void @llvm.x86.avx512.scatter.dpd.512 (i8*, i8, <8 x i32>, <8 x double>, i32)

declare <8 x float> @llvm.x86.avx512.gather.qps.512 (<8 x float>, i8*, <8 x i64>, i8, i32)
declare void @llvm.x86.avx512.scatter.qps.512 (i8*, i8, <8 x i64>, <8 x float>, i32)
declare <8 x double> @llvm.x86.avx512.gather.qpd.512 (<8 x double>, i8*, <8 x i64>, i8, i32)
declare void @llvm.x86.avx512.scatter.qpd.512 (i8*, i8, <8 x i64>, <8 x double>, i32)

;CHECK-LABEL: gather_mask_dps
;CHECK: kmovw
;CHECK: vgatherdps
;CHECK: vpadd
;CHECK: vscatterdps
;CHECK: ret
define void @gather_mask_dps(<16 x i32> %ind, <16 x float> %src, i16 %mask, i8* %base, i8* %stbuf)  {
  %x = call <16 x float> @llvm.x86.avx512.gather.dps.512 (<16 x float> %src, i8* %base, <16 x i32>%ind, i16 %mask, i32 4)
  %ind2 = add <16 x i32> %ind, <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  call void @llvm.x86.avx512.scatter.dps.512 (i8* %stbuf, i16 %mask, <16 x i32>%ind2, <16 x float> %x, i32 4)
  ret void
}

;CHECK-LABEL: gather_mask_dpd
;CHECK: kmovw
;CHECK: vgatherdpd
;CHECK: vpadd
;CHECK: vscatterdpd
;CHECK: ret
define void @gather_mask_dpd(<8 x i32> %ind, <8 x double> %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = call <8 x double> @llvm.x86.avx512.gather.dpd.512 (<8 x double> %src, i8* %base, <8 x i32>%ind, i8 %mask, i32 4)
  %ind2 = add <8 x i32> %ind, <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  call void @llvm.x86.avx512.scatter.dpd.512 (i8* %stbuf, i8 %mask, <8 x i32>%ind2, <8 x double> %x, i32 4)
  ret void
}

;CHECK-LABEL: gather_mask_qps
;CHECK: kmovw
;CHECK: vgatherqps
;CHECK: vpadd
;CHECK: vscatterqps
;CHECK: ret
define void @gather_mask_qps(<8 x i64> %ind, <8 x float> %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = call <8 x float> @llvm.x86.avx512.gather.qps.512 (<8 x float> %src, i8* %base, <8 x i64>%ind, i8 %mask, i32 4)
  %ind2 = add <8 x i64> %ind, <i64 0, i64 1, i64 2, i64 3, i64 0, i64 1, i64 2, i64 3>
  call void @llvm.x86.avx512.scatter.qps.512 (i8* %stbuf, i8 %mask, <8 x i64>%ind2, <8 x float> %x, i32 4)
  ret void
}

;CHECK-LABEL: gather_mask_qpd
;CHECK: kmovw
;CHECK: vgatherqpd
;CHECK: vpadd
;CHECK: vscatterqpd
;CHECK: ret
define void @gather_mask_qpd(<8 x i64> %ind, <8 x double> %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = call <8 x double> @llvm.x86.avx512.gather.qpd.512 (<8 x double> %src, i8* %base, <8 x i64>%ind, i8 %mask, i32 4)
  %ind2 = add <8 x i64> %ind, <i64 0, i64 1, i64 2, i64 3, i64 0, i64 1, i64 2, i64 3>
  call void @llvm.x86.avx512.scatter.qpd.512 (i8* %stbuf, i8 %mask, <8 x i64>%ind2, <8 x double> %x, i32 4)
  ret void
}
;;
;; Integer Gather/Scatter
;;
declare <16 x i32> @llvm.x86.avx512.gather.dpi.512 (<16 x i32>, i8*, <16 x i32>, i16, i32)
declare void @llvm.x86.avx512.scatter.dpi.512 (i8*, i16, <16 x i32>, <16 x i32>, i32)
declare <8 x i64> @llvm.x86.avx512.gather.dpq.512 (<8 x i64>, i8*, <8 x i32>, i8, i32)
declare void @llvm.x86.avx512.scatter.dpq.512 (i8*, i8, <8 x i32>, <8 x i64>, i32)

declare <8 x i32> @llvm.x86.avx512.gather.qpi.512 (<8 x i32>, i8*, <8 x i64>, i8, i32)
declare void @llvm.x86.avx512.scatter.qpi.512 (i8*, i8, <8 x i64>, <8 x i32>, i32)
declare <8 x i64> @llvm.x86.avx512.gather.qpq.512 (<8 x i64>, i8*, <8 x i64>, i8, i32)
declare void @llvm.x86.avx512.scatter.qpq.512 (i8*, i8, <8 x i64>, <8 x i64>, i32)

;CHECK-LABEL: gather_mask_dd
;CHECK: kmovw
;CHECK: vpgatherdd
;CHECK: vpadd
;CHECK: vpscatterdd
;CHECK: ret
define void @gather_mask_dd(<16 x i32> %ind, <16 x i32> %src, i16 %mask, i8* %base, i8* %stbuf)  {
  %x = call <16 x i32> @llvm.x86.avx512.gather.dpi.512 (<16 x i32> %src, i8* %base, <16 x i32>%ind, i16 %mask, i32 4)
  %ind2 = add <16 x i32> %ind, <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  call void @llvm.x86.avx512.scatter.dpi.512 (i8* %stbuf, i16 %mask, <16 x i32>%ind2, <16 x i32> %x, i32 4)
  ret void
}

;CHECK-LABEL: gather_mask_qd
;CHECK: kmovw
;CHECK: vpgatherqd
;CHECK: vpadd
;CHECK: vpscatterqd
;CHECK: ret
define void @gather_mask_qd(<8 x i64> %ind, <8 x i32> %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = call <8 x i32> @llvm.x86.avx512.gather.qpi.512 (<8 x i32> %src, i8* %base, <8 x i64>%ind, i8 %mask, i32 4)
  %ind2 = add <8 x i64> %ind, <i64 0, i64 1, i64 2, i64 3, i64 0, i64 1, i64 2, i64 3>
  call void @llvm.x86.avx512.scatter.qpi.512 (i8* %stbuf, i8 %mask, <8 x i64>%ind2, <8 x i32> %x, i32 4)
  ret void
}

;CHECK-LABEL: gather_mask_qq
;CHECK: kmovw
;CHECK: vpgatherqq
;CHECK: vpadd
;CHECK: vpscatterqq
;CHECK: ret
define void @gather_mask_qq(<8 x i64> %ind, <8 x i64> %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = call <8 x i64> @llvm.x86.avx512.gather.qpq.512 (<8 x i64> %src, i8* %base, <8 x i64>%ind, i8 %mask, i32 4)
  %ind2 = add <8 x i64> %ind, <i64 0, i64 1, i64 2, i64 3, i64 0, i64 1, i64 2, i64 3>
  call void @llvm.x86.avx512.scatter.qpq.512 (i8* %stbuf, i8 %mask, <8 x i64>%ind2, <8 x i64> %x, i32 4)
  ret void
}

;CHECK-LABEL: gather_mask_dq
;CHECK: kmovw
;CHECK: vpgatherdq
;CHECK: vpadd
;CHECK: vpscatterdq
;CHECK: ret
define void @gather_mask_dq(<8 x i32> %ind, <8 x i64> %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = call <8 x i64> @llvm.x86.avx512.gather.dpq.512 (<8 x i64> %src, i8* %base, <8 x i32>%ind, i8 %mask, i32 4)
  %ind2 = add <8 x i32> %ind, <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  call void @llvm.x86.avx512.scatter.dpq.512 (i8* %stbuf, i8 %mask, <8 x i32>%ind2, <8 x i64> %x, i32 4)
  ret void
}


;CHECK-LABEL: gather_mask_dpd_execdomain
;CHECK: vgatherdpd
;CHECK: vmovapd
;CHECK: ret
define void @gather_mask_dpd_execdomain(<8 x i32> %ind, <8 x double> %src, i8 %mask, i8* %base, <8 x double>* %stbuf)  {
  %x = call <8 x double> @llvm.x86.avx512.gather.dpd.512 (<8 x double> %src, i8* %base, <8 x i32>%ind, i8 %mask, i32 4)
  store <8 x double> %x, <8 x double>* %stbuf
  ret void
}

;CHECK-LABEL: gather_mask_qpd_execdomain
;CHECK: vgatherqpd
;CHECK: vmovapd
;CHECK: ret
define void @gather_mask_qpd_execdomain(<8 x i64> %ind, <8 x double> %src, i8 %mask, i8* %base, <8 x double>* %stbuf)  {
  %x = call <8 x double> @llvm.x86.avx512.gather.qpd.512 (<8 x double> %src, i8* %base, <8 x i64>%ind, i8 %mask, i32 4)
  store <8 x double> %x, <8 x double>* %stbuf
  ret void
}

;CHECK-LABEL: gather_mask_dps_execdomain
;CHECK: vgatherdps
;CHECK: vmovaps 
;CHECK: ret
define <16 x float> @gather_mask_dps_execdomain(<16 x i32> %ind, <16 x float> %src, i16 %mask, i8* %base)  {
  %res = call <16 x float> @llvm.x86.avx512.gather.dps.512 (<16 x float> %src, i8* %base, <16 x i32>%ind, i16 %mask, i32 4)
  ret <16 x float> %res;
}

;CHECK-LABEL: gather_mask_qps_execdomain
;CHECK: vgatherqps
;CHECK: vmovaps
;CHECK: ret
define <8 x float> @gather_mask_qps_execdomain(<8 x i64> %ind, <8 x float> %src, i8 %mask, i8* %base)  {
  %res = call <8 x float> @llvm.x86.avx512.gather.qps.512 (<8 x float> %src, i8* %base, <8 x i64>%ind, i8 %mask, i32 4)
  ret <8 x float> %res;
}

;CHECK-LABEL: scatter_mask_dpd_execdomain
;CHECK: vmovapd
;CHECK: vscatterdpd
;CHECK: ret
define void @scatter_mask_dpd_execdomain(<8 x i32> %ind, <8 x double>* %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = load <8 x double>, <8 x double>* %src, align 64 
  call void @llvm.x86.avx512.scatter.dpd.512 (i8* %stbuf, i8 %mask, <8 x i32>%ind, <8 x double> %x, i32 4)
  ret void
}

;CHECK-LABEL: scatter_mask_qpd_execdomain
;CHECK: vmovapd
;CHECK: vscatterqpd
;CHECK: ret
define void @scatter_mask_qpd_execdomain(<8 x i64> %ind, <8 x double>* %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = load <8 x double>, <8 x double>* %src, align 64
  call void @llvm.x86.avx512.scatter.qpd.512 (i8* %stbuf, i8 %mask, <8 x i64>%ind, <8 x double> %x, i32 4)
  ret void
}

;CHECK-LABEL: scatter_mask_dps_execdomain
;CHECK: vmovaps
;CHECK: vscatterdps
;CHECK: ret
define void @scatter_mask_dps_execdomain(<16 x i32> %ind, <16 x float>* %src, i16 %mask, i8* %base, i8* %stbuf)  {
  %x = load <16 x float>, <16 x float>* %src, align 64
  call void @llvm.x86.avx512.scatter.dps.512 (i8* %stbuf, i16 %mask, <16 x i32>%ind, <16 x float> %x, i32 4)
  ret void
}

;CHECK-LABEL: scatter_mask_qps_execdomain
;CHECK: vmovaps
;CHECK: vscatterqps
;CHECK: ret
define void @scatter_mask_qps_execdomain(<8 x i64> %ind, <8 x float>* %src, i8 %mask, i8* %base, i8* %stbuf)  {
  %x = load <8 x float>, <8 x float>* %src, align 32 
  call void @llvm.x86.avx512.scatter.qps.512 (i8* %stbuf, i8 %mask, <8 x i64>%ind, <8 x float> %x, i32 4)
  ret void
}

;CHECK-LABEL: gather_qps
;CHECK: kxnorw
;CHECK: vgatherqps
;CHECK: vpadd
;CHECK: vscatterqps
;CHECK: ret
define void @gather_qps(<8 x i64> %ind, <8 x float> %src, i8* %base, i8* %stbuf)  {
  %x = call <8 x float> @llvm.x86.avx512.gather.qps.512 (<8 x float> %src, i8* %base, <8 x i64>%ind, i8 -1, i32 4)
  %ind2 = add <8 x i64> %ind, <i64 0, i64 1, i64 2, i64 3, i64 0, i64 1, i64 2, i64 3>
  call void @llvm.x86.avx512.scatter.qps.512 (i8* %stbuf, i8 -1, <8 x i64>%ind2, <8 x float> %x, i32 4)
  ret void
}

;CHECK-LABEL: prefetch
;CHECK: gatherpf0
;CHECK: gatherpf1
;CHECK: scatterpf0
;CHECK: scatterpf1
;CHECK: ret
declare  void @llvm.x86.avx512.gatherpf.qps.512(i8, <8 x i64>, i8* , i32, i32);
declare  void @llvm.x86.avx512.scatterpf.qps.512(i8, <8 x i64>, i8* , i32, i32);
define void @prefetch(<8 x i64> %ind, i8* %base) {
  call void @llvm.x86.avx512.gatherpf.qps.512(i8 -1, <8 x i64> %ind, i8* %base, i32 4, i32 0)
  call void @llvm.x86.avx512.gatherpf.qps.512(i8 -1, <8 x i64> %ind, i8* %base, i32 4, i32 1)
  call void @llvm.x86.avx512.scatterpf.qps.512(i8 -1, <8 x i64> %ind, i8* %base, i32 2, i32 0)
  call void @llvm.x86.avx512.scatterpf.qps.512(i8 -1, <8 x i64> %ind, i8* %base, i32 2, i32 1)
  ret void
}
