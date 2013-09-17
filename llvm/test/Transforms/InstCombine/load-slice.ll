; RUN: opt -default-data-layout="E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128" -instcombine -instcombine-stress-load-slicing -S < %s -o - | FileCheck %s --check-prefix=BIG
; RUN: opt -default-data-layout="e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128" -instcombine -instcombine-stress-load-slicing -S < %s -o - | FileCheck %s --check-prefix=LITTLE
;
; <rdar://problem/14477220>

%class.Complex = type { float, float }


; Check that independant slices leads to independant loads.
;
; The 64-bits should have been split in two 32-bits slices.
; The big endian layout is:
; MSB 7 6 5 4 | 3 2 1 0 LSB
;      High       Low
; The base address points to 7 and is 8-bytes aligned.
; Low slice starts at 3 (base + 4-bytes) and is 4-bytes aligned.
; High slice starts at 7 (base) and is 8-bytes aligned.
;
; The little endian layout is:
; LSB 0 1 2 3 | 4 5 6 7 MSB
;       Low      High
; The base address points to 0 and is 8-bytes aligned.
; Low slice starts at 0 (base) and is 8-bytes aligned.
; High slice starts at 4 (base + 4-bytes) and is 4-bytes aligned.
;
define void @t1(%class.Complex* nocapture %out, i64 %out_start) {
; BIG-LABEL: @t1
; Original load should have been sliced.
; BIG-NOT: load i64*
; BIG-NOT: trunc i64
; BIG-NOT: lshr i64
;
; First 32-bits slice.
; BIG: [[HIGH_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start
; BIG: [[HIGH_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[HIGH_SLICE_BASEADDR]] to i32*
; BIG: [[HIGH_SLICE:%[a-zA-Z.0-9_]+]] = load i32* [[HIGH_SLICE_ADDR]], align 8
;
; Second 32-bits slice.
; BIG: [[LOW_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start, i32 1
; BIG: [[LOW_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  float* [[LOW_SLICE_BASEADDR]] to i32*
; BIG: [[LOW_SLICE:%[a-zA-Z.0-9_]+]] = load i32* [[LOW_SLICE_ADDR]], align 4
;
; Cast to the final type.
; BIG: [[LOW_SLICE_FLOAT:%[a-zA-Z.0-9_]+]] = bitcast i32 [[LOW_SLICE]] to float
; BIG: [[HIGH_SLICE_FLOAT:%[a-zA-Z.0-9_]+]] = bitcast i32 [[HIGH_SLICE]] to float
;
; Uses of the slices.
; BIG: fadd float {{%[a-zA-Z.0-9_]+}}, [[LOW_SLICE_FLOAT]]
; BIG: fadd float {{%[a-zA-Z.0-9_]+}}, [[HIGH_SLICE_FLOAT]]
;
; LITTLE-LABEL: @t1
; Original load should have been sliced.
; LITTLE-NOT: load i64*
; LITTLE-NOT: trunc i64
; LITTLE-NOT: lshr i64
;
; LITTLE: [[BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start
;
; First 32-bits slice.
; LITTLE: [[HIGH_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start, i32 1
; LITTLE: [[HIGH_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast float* [[HIGH_SLICE_BASEADDR]] to i32*
; LITTLE: [[HIGH_SLICE:%[a-zA-Z.0-9_]+]] = load i32* [[HIGH_SLICE_ADDR]], align 4
;
; Second 32-bits slice.
; LITTLE: [[LOW_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[BASEADDR]] to i32*
; LITTLE: [[LOW_SLICE:%[a-zA-Z.0-9_]+]] = load i32* [[LOW_SLICE_ADDR]], align 8
;
; Cast to the final type.
; LITTLE: [[LOW_SLICE_FLOAT:%[a-zA-Z.0-9_]+]] = bitcast i32 [[LOW_SLICE]] to float
; LITTLE: [[HIGH_SLICE_FLOAT:%[a-zA-Z.0-9_]+]] = bitcast i32 [[HIGH_SLICE]] to float
;
; Uses of the slices.
; LITTLE: fadd float {{%[a-zA-Z.0-9_]+}}, [[LOW_SLICE_FLOAT]]
; LITTLE: fadd float {{%[a-zA-Z.0-9_]+}}, [[HIGH_SLICE_FLOAT]]
entry:
  %arrayidx = getelementptr inbounds %class.Complex* %out, i64 %out_start
  %tmp = bitcast %class.Complex* %arrayidx to i64*
  %tmp1 = load i64* %tmp, align 8
  %t0.sroa.0.0.extract.trunc = trunc i64 %tmp1 to i32
  %tmp2 = bitcast i32 %t0.sroa.0.0.extract.trunc to float
  %t0.sroa.2.0.extract.shift = lshr i64 %tmp1, 32
  %t0.sroa.2.0.extract.trunc = trunc i64 %t0.sroa.2.0.extract.shift to i32
  %tmp3 = bitcast i32 %t0.sroa.2.0.extract.trunc to float
  %add = add i64 %out_start, 8
  %arrayidx2 = getelementptr inbounds %class.Complex* %out, i64 %add
  %i.i = getelementptr inbounds %class.Complex* %arrayidx2, i64 0, i32 0
  %tmp4 = load float* %i.i, align 4
  %add.i = fadd float %tmp4, %tmp2
  %retval.sroa.0.0.vec.insert.i = insertelement <2 x float> undef, float %add.i, i32 0
  %r.i = getelementptr inbounds %class.Complex* %arrayidx2, i64 0, i32 1
  %tmp5 = load float* %r.i, align 4
  %add5.i = fadd float %tmp5, %tmp3
  %retval.sroa.0.4.vec.insert.i = insertelement <2 x float> %retval.sroa.0.0.vec.insert.i, float %add5.i, i32 1
  %ref.tmp.sroa.0.0.cast = bitcast %class.Complex* %arrayidx to <2 x float>*
  store <2 x float> %retval.sroa.0.4.vec.insert.i, <2 x float>* %ref.tmp.sroa.0.0.cast, align 4
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture)

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture)

; Check that slices not involved in arithmetic are not split in independant loads.
; BIG-LABEL: @t2
; BIG: load i16*
; BIG: trunc i16 {{%[a-zA-Z.0-9_]+}} to i8
; BIG: lshr i16 {{%[a-zA-Z.0-9_]+}}, 8
; BIG: trunc i16 {{%[a-zA-Z.0-9_]+}} to i8
;
; LITTLE-LABEL: @t2
; LITTLE: load i16*
; LITTLE: trunc i16 {{%[a-zA-Z.0-9_]+}} to i8
; LITTLE: lshr i16 {{%[a-zA-Z.0-9_]+}}, 8
; LITTLE: trunc i16 {{%[a-zA-Z.0-9_]+}} to i8
define void @t2(%class.Complex* nocapture %out, i64 %out_start) {
  %arrayidx = getelementptr inbounds %class.Complex* %out, i64 %out_start
  %bitcast = bitcast %class.Complex* %arrayidx to i16*
  %chunk16 = load i16* %bitcast, align 8
  %slice8_low = trunc i16 %chunk16 to i8
  %shift = lshr i16 %chunk16, 8
  %slice8_high = trunc i16 %shift to i8
  %vec = insertelement <2 x i8> undef, i8 %slice8_high, i32 0
  %vec1 = insertelement <2 x i8> %vec, i8 %slice8_low, i32 1
  %addr = bitcast %class.Complex* %arrayidx to <2 x i8>*
  store <2 x i8> %vec1, <2 x i8>* %addr, align 8
  ret void
}

; Check that we do not read outside of the chunk of bits of the original loads.
;
; The 64-bits should have been split in one 32-bits and one 16-bits slices.
; The 16-bits should be zero extended to match the final type.
; The big endian layout is:
; MSB 7 6 | 5 4 | 3 2 1 0 LSB
;    High           Low
; The base address points to 7 and is 8-bytes aligned.
; Low slice starts at 3 (base + 4-bytes) and is 4-bytes aligned.
; High slice starts at 7 (base) and is 8-bytes aligned.
;
; The little endian layout is:
; LSB 0 1 2 3 | 4 5 | 6 7 MSB
;      Low            High
; The base address points to 0 and is 8-bytes aligned.
; Low slice starts at 0 (base) and is 8-bytes aligned.
; High slice starts at 6 (base + 6-bytes) and is 2-bytes aligned.
;
; BIG-LABEL: @t3
; Original load should have been sliced.
; BIG-NOT: load i64*
; BIG-NOT: trunc i64
; BIG-NOT: lshr i64
;
; First 32-bits slice where only 16-bits comes from the memory.
; BIG: [[HIGH_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start
; BIG: [[HIGH_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[HIGH_SLICE_BASEADDR]] to i16*
; BIG: [[HIGH_SLICE:%[a-zA-Z.0-9_]+]] = load i16* [[HIGH_SLICE_ADDR]], align 8
; BIG: [[HIGH_SLICE_ZEXT:%[a-zA-Z.0-9_]+]] = zext i16 [[HIGH_SLICE]] to i32
;
; Second 32-bits slice.
; BIG: [[LOW_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start, i32 1
; BIG: [[LOW_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  float* [[LOW_SLICE_BASEADDR]] to i32*
; BIG: [[LOW_SLICE:%[a-zA-Z.0-9_]+]] = load i32* [[LOW_SLICE_ADDR]], align 4
;
; Use of the slices.
; BIG: add i32 [[HIGH_SLICE_ZEXT]], [[LOW_SLICE]]
;
; LITTLE-LABEL: @t3
; Original load should have been sliced.
; LITTLE-NOT: load i64*
; LITTLE-NOT: trunc i64
; LITTLE-NOT: lshr i64
;
; LITTLE: [[BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start
;
; First 32-bits slice where only 16-bits comes from the memory.
; LITTLE: [[HIGH_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[BASEADDR]] to i8*
; LITTLE: [[HIGH_SLICE_ADDR_I8:%[a-zA-Z.0-9_]+]] = getelementptr inbounds i8* [[HIGH_SLICE_ADDR]], i64 6
; LITTLE: [[HIGH_SLICE_ADDR_I16:%[a-zA-Z.0-9_]+]] = bitcast i8* [[HIGH_SLICE_ADDR_I8]] to i16*
; LITTLE: [[HIGH_SLICE:%[a-zA-Z.0-9_]+]] = load i16* [[HIGH_SLICE_ADDR_I16]], align 2
; LITTLE: [[HIGH_SLICE_ZEXT:%[a-zA-Z.0-9_]+]] = zext i16 [[HIGH_SLICE]] to i32
;
; Second 32-bits slice.
; LITTLE: [[LOW_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[BASEADDR]] to i32*
; LITTLE: [[LOW_SLICE:%[a-zA-Z.0-9_]+]] = load i32* [[LOW_SLICE_ADDR]], align 8
;
; Use of the slices.
; LITTLE: add i32 [[HIGH_SLICE_ZEXT]], [[LOW_SLICE]]
define i32 @t3(%class.Complex* nocapture %out, i64 %out_start) {
  %arrayidx = getelementptr inbounds %class.Complex* %out, i64 %out_start
  %bitcast = bitcast %class.Complex* %arrayidx to i64*
  %chunk64 = load i64* %bitcast, align 8
  %slice32_low = trunc i64 %chunk64 to i32
  %shift48 = lshr i64 %chunk64, 48
  %slice32_high = trunc i64 %shift48 to i32
  %res = add i32 %slice32_high, %slice32_low
  ret i32 %res
}

; Check that we do not optimize overlapping slices.
;
; The 64-bits should NOT have been split in as slices are overlapping.
; First slice uses bytes numbered 0 to 3.
; Second slice uses bytes numbered 6 and 7.
; Third slice uses bytes numbered 4 to 7.
; BIG-LABEL: @t4
; BIG: load i64* {{%[a-zA-Z.0-9_]+}}, align 8
; BIG: trunc i64 {{%[a-zA-Z.0-9_]+}} to i32
; BIG: lshr i64 {{%[a-zA-Z.0-9_]+}}, 48
; BIG: trunc i64 {{%[a-zA-Z.0-9_]+}} to i32
; BIG: lshr i64 {{%[a-zA-Z.0-9_]+}}, 32
; BIG: trunc i64 {{%[a-zA-Z.0-9_]+}} to i32
;
; LITTLE-LABEL: @t4
; LITTLE: load i64* {{%[a-zA-Z.0-9_]+}}, align 8
; LITTLE: trunc i64 {{%[a-zA-Z.0-9_]+}} to i32
; LITTLE: lshr i64 {{%[a-zA-Z.0-9_]+}}, 48
; LITTLE: trunc i64 {{%[a-zA-Z.0-9_]+}} to i32
; LITTLE: lshr i64 {{%[a-zA-Z.0-9_]+}}, 32
; LITTLE: trunc i64 {{%[a-zA-Z.0-9_]+}} to i32
define i32 @t4(%class.Complex* nocapture %out, i64 %out_start) {
  %arrayidx = getelementptr inbounds %class.Complex* %out, i64 %out_start
  %bitcast = bitcast %class.Complex* %arrayidx to i64*
  %chunk64 = load i64* %bitcast, align 8
  %slice32_low = trunc i64 %chunk64 to i32
  %shift48 = lshr i64 %chunk64, 48
  %slice32_high = trunc i64 %shift48 to i32
  %shift32 = lshr i64 %chunk64, 32
  %slice32_lowhigh = trunc i64 %shift32 to i32
  %tmpres = add i32 %slice32_high, %slice32_low
  %res = add i32 %slice32_lowhigh, %tmpres
  ret i32 %res
}

; Check that we optimize when 3 slices are involved.
; The 64-bits should have been split in one 32-bits and one 16-bits slices.
; The 16-bits should be zero extended to match the final type.
; The big endian layout is:
; MSB 7 6 | 5 4 | 3 2 1 0 LSB
;    High LowHigh    Low
; The base address points to 7 and is 8-bytes aligned.
; Low slice starts at 3 (base + 4-bytes) and is 4-bytes aligned.
; High slice starts at 7 (base) and is 8-bytes aligned.
; LowHigh slice starts at 5 (base + 2-bytes) and is 2-bytes aligned.
;
; The little endian layout is:
; LSB 0 1 2 3 | 4 5 | 6 7 MSB
;      Low    LowHigh High
; The base address points to 0 and is 8-bytes aligned.
; Low slice starts at 0 (base) and is 8-bytes aligned.
; High slice starts at 6 (base + 6-bytes) and is 2-bytes aligned.
; LowHigh slice starts at 4 (base + 4-bytes) and is 4-bytes aligned.
;
; Original load should have been sliced.
; BIG-LABEL: @t5
; BIG-NOT: load i64*
; BIG-NOT: trunc i64
; BIG-NOT: lshr i64
;
; LowHigh 32-bits slice where only 16-bits comes from the memory.
; BIG: [[LOWHIGH_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start
; BIG: [[LOWHIGH_SLICE_BASEADDR_I8:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[LOWHIGH_SLICE_BASEADDR]] to i8*
; BIG: [[LOWHIGH_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds i8* [[LOWHIGH_SLICE_BASEADDR_I8]], i64 2
; BIG: [[LOWHIGH_SLICE_ADDR_I16:%[a-zA-Z.0-9_]+]] = bitcast  i8* [[LOWHIGH_SLICE_ADDR]] to i16*
; BIG: [[LOWHIGH_SLICE:%[a-zA-Z.0-9_]+]] = load i16* [[LOWHIGH_SLICE_ADDR_I16]], align 2
;
; First 32-bits slice where only 16-bits comes from the memory.
; BIG: [[HIGH_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[LOWHIGH_SLICE_BASEADDR]] to i16*
; BIG: [[HIGH_SLICE:%[a-zA-Z.0-9_]+]] = load i16* [[HIGH_SLICE_ADDR]], align 8
; BIG: [[HIGH_SLICE_ZEXT:%[a-zA-Z.0-9_]+]] = zext i16 [[HIGH_SLICE]] to i32
;
; Second 32-bits slice.
; BIG: [[LOW_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start, i32 1
; BIG: [[LOW_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  float* [[LOW_SLICE_BASEADDR]] to i32*
; BIG: [[LOW_SLICE:%[a-zA-Z.0-9_]+]] = load i32* [[LOW_SLICE_ADDR]], align 4
;
; Original sext is still here.
; BIG: [[LOWHIGH_SLICE_SEXT:%[a-zA-Z.0-9_]+]] = sext i16 [[LOWHIGH_SLICE]] to i32
;
; Uses of the slices.
; BIG: [[RES:%[a-zA-Z.0-9_]+]] = add i32 [[HIGH_SLICE_ZEXT]], [[LOW_SLICE]]
; BIG: add i32 [[LOWHIGH_SLICE_SEXT]], [[RES]]
;
; LITTLE-LABEL: @t5
; LITTLE-NOT: load i64*
; LITTLE-NOT: trunc i64
; LITTLE-NOT: lshr i64
;
; LITTLE: [[BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start
;
; LowHigh 32-bits slice where only 16-bits comes from the memory.
; LITTLE: [[LOWHIGH_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = getelementptr inbounds %class.Complex* %out, i64 %out_start, i32 1
; LITTLE: [[LOWHIGH_SLICE_ADDR_I16:%[a-zA-Z.0-9_]+]] = bitcast  float* [[LOWHIGH_SLICE_BASEADDR]] to i16*
; LITTLE: [[LOWHIGH_SLICE:%[a-zA-Z.0-9_]+]] = load i16* [[LOWHIGH_SLICE_ADDR_I16]], align 4
;
; First 32-bits slice where only 16-bits comes from the memory.
; LITTLE: [[HIGH_SLICE_BASEADDR:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[BASEADDR]] to i8*
; LITTLE: [[HIGH_SLICE_ADDR_I8:%[a-zA-Z.0-9_]+]] = getelementptr inbounds i8* [[HIGH_SLICE_BASEADDR]], i64 6
; LITTLE: [[HIGH_SLICE_ADDR_I16:%[a-zA-Z.0-9_]+]] = bitcast  i8* [[HIGH_SLICE_ADDR_I8]] to i16*
; LITTLE: [[HIGH_SLICE:%[a-zA-Z.0-9_]+]] = load i16* [[HIGH_SLICE_ADDR_I16]], align 2
; LITTLE: [[HIGH_SLICE_ZEXT:%[a-zA-Z.0-9_]+]] = zext i16 [[HIGH_SLICE]] to i32
;
; Second 32-bits slice.
; LITTLE: [[LOW_SLICE_ADDR:%[a-zA-Z.0-9_]+]] = bitcast  %class.Complex* [[BASEADDR]] to i32*
; LITTLE: [[LOW_SLICE:%[a-zA-Z.0-9_]+]] = load i32* [[LOW_SLICE_ADDR]], align 8
;
; Original sext is still here.
; LITTLE: [[LOWHIGH_SLICE_SEXT:%[a-zA-Z.0-9_]+]] = sext i16 [[LOWHIGH_SLICE]] to i32
;
; Uses of the slices.
; LITTLE: [[RES:%[a-zA-Z.0-9_]+]] = add i32 [[HIGH_SLICE_ZEXT]], [[LOW_SLICE]]
; LITTLE: add i32 [[LOWHIGH_SLICE_SEXT]], [[RES]]
define i32 @t5(%class.Complex* nocapture %out, i64 %out_start) {
  %arrayidx = getelementptr inbounds %class.Complex* %out, i64 %out_start
  %bitcast = bitcast %class.Complex* %arrayidx to i64*
  %chunk64 = load i64* %bitcast, align 8
  %slice32_low = trunc i64 %chunk64 to i32
  %shift48 = lshr i64 %chunk64, 48
  %slice32_high = trunc i64 %shift48 to i32
  %shift32 = lshr i64 %chunk64, 32
  %slice16_lowhigh = trunc i64 %shift32 to i16
  %slice32_lowhigh = sext i16 %slice16_lowhigh to i32
  %tmpres = add i32 %slice32_high, %slice32_low
  %res = add i32 %slice32_lowhigh, %tmpres
  ret i32 %res
}
