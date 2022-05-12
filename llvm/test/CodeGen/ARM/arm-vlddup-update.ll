; RUN: llc < %s -mtriple=armv8-linux-gnueabi -verify-machineinstrs \
; RUN:     -asm-verbose=false | FileCheck %s

%struct.uint16x4x2_t = type { <4 x i16>, <4 x i16> }
%struct.uint16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.uint16x4x4_t = type { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }

%struct.uint32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.uint32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.uint32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }

%struct.uint64x1x2_t = type { <1 x i64>, <1 x i64> }
%struct.uint64x1x3_t = type { <1 x i64>, <1 x i64>, <1 x i64> }
%struct.uint64x1x4_t = type { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }

%struct.uint8x8x2_t = type { <8 x i8>, <8 x i8> }
%struct.uint8x8x3_t = type { <8 x i8>, <8 x i8>, <8 x i8> }
%struct.uint8x8x4_t = type { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }

%struct.uint16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.uint16x8x3_t = type { <8 x i16>, <8 x i16>, <8 x i16> }
%struct.uint16x8x4_t = type { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }

%struct.uint32x4x2_t = type { <4 x i32>, <4 x i32> }
%struct.uint32x4x3_t = type { <4 x i32>, <4 x i32>, <4 x i32> }
%struct.uint32x4x4_t = type { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }

%struct.uint8x16x2_t = type { <16 x i8>, <16 x i8> }
%struct.uint8x16x3_t = type { <16 x i8>, <16 x i8>, <16 x i8> }
%struct.uint8x16x4_t = type { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }

declare %struct.uint8x8x2_t @llvm.arm.neon.vld2dup.v8i8.p0i8(i8*, i32)
declare %struct.uint16x4x2_t @llvm.arm.neon.vld2dup.v4i16.p0i8(i8*, i32)
declare %struct.uint32x2x2_t @llvm.arm.neon.vld2dup.v2i32.p0i8(i8*, i32)
declare %struct.uint64x1x2_t @llvm.arm.neon.vld2dup.v1i64.p0i8(i8*, i32)

declare %struct.uint8x8x3_t @llvm.arm.neon.vld3dup.v8i8.p0i8(i8*, i32)
declare %struct.uint16x4x3_t @llvm.arm.neon.vld3dup.v4i16.p0i8(i8*, i32)
declare %struct.uint32x2x3_t @llvm.arm.neon.vld3dup.v2i32.p0i8(i8*, i32)
declare %struct.uint64x1x3_t @llvm.arm.neon.vld3dup.v1i64.p0i8(i8*, i32)

declare %struct.uint8x8x4_t @llvm.arm.neon.vld4dup.v8i8.p0i8(i8*, i32)
declare %struct.uint16x4x4_t @llvm.arm.neon.vld4dup.v4i16.p0i8(i8*, i32)
declare %struct.uint32x2x4_t @llvm.arm.neon.vld4dup.v2i32.p0i8(i8*, i32)
declare %struct.uint64x1x4_t @llvm.arm.neon.vld4dup.v1i64.p0i8(i8*, i32)

declare %struct.uint8x16x2_t @llvm.arm.neon.vld2dup.v16i8.p0i8(i8*, i32)
declare %struct.uint16x8x2_t @llvm.arm.neon.vld2dup.v8i16.p0i8(i8*, i32)
declare %struct.uint32x4x2_t @llvm.arm.neon.vld2dup.v4i32.p0i8(i8*, i32)

declare %struct.uint8x16x3_t @llvm.arm.neon.vld3dup.v16i8.p0i8(i8*, i32)
declare %struct.uint16x8x3_t @llvm.arm.neon.vld3dup.v8i16.p0i8(i8*, i32)
declare %struct.uint32x4x3_t @llvm.arm.neon.vld3dup.v4i32.p0i8(i8*, i32)

declare %struct.uint8x16x4_t @llvm.arm.neon.vld4dup.v16i8.p0i8(i8*, i32)
declare %struct.uint16x8x4_t @llvm.arm.neon.vld4dup.v8i16.p0i8(i8*, i32)
declare %struct.uint32x4x4_t @llvm.arm.neon.vld4dup.v4i32.p0i8(i8*, i32)

define i8* @test_vld2_dup_u16_update(%struct.uint16x4x2_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld2_dup_u16_update:
; CHECK:         vld2.16 {d16[], d17[]}, [r1]!
entry:
  %tmp = tail call %struct.uint16x4x2_t @llvm.arm.neon.vld2dup.v4i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x4x2_t %tmp, %struct.uint16x4x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 4
  ret i8* %updated_src
}

define i8* @test_vld2_dup_u16_update_reg(%struct.uint16x4x2_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld2_dup_u16_update_reg:
; CHECK:         vld2.16 {d16[], d17[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint16x4x2_t @llvm.arm.neon.vld2dup.v4i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x4x2_t %tmp, %struct.uint16x4x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld2_dup_update(%struct.uint32x2x2_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld2_dup_update:
; CHECK:         vld2.32 {d16[], d17[]}, [r1]!
entry:
  %tmp = tail call %struct.uint32x2x2_t @llvm.arm.neon.vld2dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x2_t %tmp, %struct.uint32x2x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 8
  ret i8* %updated_src
}

define i8* @test_vld2_dup_update_reg(%struct.uint32x2x2_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld2_dup_update_reg:
; CHECK:         vld2.32 {d16[], d17[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint32x2x2_t @llvm.arm.neon.vld2dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x2_t %tmp, %struct.uint32x2x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld2_dup_u64_update(%struct.uint64x1x2_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld2_dup_u64_update:
; CHECK:         vld1.64 {d16, d17}, [r1:64]!
entry:
  %tmp = tail call %struct.uint64x1x2_t @llvm.arm.neon.vld2dup.v1i64.p0i8(i8* %src, i32 8)
  store %struct.uint64x1x2_t %tmp, %struct.uint64x1x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 16
  ret i8* %updated_src
}

define i8* @test_vld2_dup_u64_update_reg(%struct.uint64x1x2_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld2_dup_u64_update_reg:
; CHECK:         vld1.64 {d16, d17}, [r1:64], r2
entry:
  %tmp = tail call %struct.uint64x1x2_t @llvm.arm.neon.vld2dup.v1i64.p0i8(i8* %src, i32 8)
  store %struct.uint64x1x2_t %tmp, %struct.uint64x1x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld2_dup_u8_update(%struct.uint8x8x2_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld2_dup_u8_update:
; CHECK:         vld2.8 {d16[], d17[]}, [r1]!
entry:
  %tmp = tail call %struct.uint8x8x2_t @llvm.arm.neon.vld2dup.v8i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x8x2_t %tmp, %struct.uint8x8x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 2
  ret i8* %updated_src
}

define i8* @test_vld2_dup_u8_update_reg(%struct.uint8x8x2_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld2_dup_u8_update_reg:
; CHECK:         vld2.8 {d16[], d17[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint8x8x2_t @llvm.arm.neon.vld2dup.v8i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x8x2_t %tmp, %struct.uint8x8x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld3_dup_u16_update(%struct.uint16x4x3_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld3_dup_u16_update:
; CHECK:         vld3.16 {d16[], d17[], d18[]}, [r1]!
entry:
  %tmp = tail call %struct.uint16x4x3_t @llvm.arm.neon.vld3dup.v4i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x4x3_t %tmp, %struct.uint16x4x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 6
  ret i8* %updated_src
}

define i8* @test_vld3_dup_u16_update_reg(%struct.uint16x4x3_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld3_dup_u16_update_reg:
; CHECK:         vld3.16 {d16[], d17[], d18[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint16x4x3_t @llvm.arm.neon.vld3dup.v4i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x4x3_t %tmp, %struct.uint16x4x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld3_dup_u32_update(%struct.uint32x2x3_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld3_dup_u32_update:
; CHECK:         vld3.32 {d16[], d17[], d18[]}, [r1]!
entry:
  %tmp = tail call %struct.uint32x2x3_t @llvm.arm.neon.vld3dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x3_t %tmp, %struct.uint32x2x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 12
  ret i8* %updated_src
}

define i8* @test_vld3_dup_u32_update_reg(%struct.uint32x2x3_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld3_dup_u32_update_reg:
; CHECK:         vld3.32 {d16[], d17[], d18[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint32x2x3_t @llvm.arm.neon.vld3dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x3_t %tmp, %struct.uint32x2x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld3_dup_u64_update(%struct.uint64x1x3_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld3_dup_u64_update:
; CHECK:         vld1.64 {d16, d17, d18}, [r1]!
entry:
  %tmp = tail call %struct.uint64x1x3_t @llvm.arm.neon.vld3dup.v1i64.p0i8(i8* %src, i32 8)
  store %struct.uint64x1x3_t %tmp, %struct.uint64x1x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 24
  ret i8* %updated_src
}

define i8* @test_vld3_dup_u64_update_reg(%struct.uint64x1x3_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld3_dup_u64_update_reg:
; CHECK:         vld1.64 {d16, d17, d18}, [r1], r2
entry:
  %tmp = tail call %struct.uint64x1x3_t @llvm.arm.neon.vld3dup.v1i64.p0i8(i8* %src, i32 8)
  store %struct.uint64x1x3_t %tmp, %struct.uint64x1x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld3_dup_u8_update(%struct.uint8x8x3_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld3_dup_u8_update:
; CHECK:         vld3.8 {d16[], d17[], d18[]}, [r1]!
entry:
  %tmp = tail call %struct.uint8x8x3_t @llvm.arm.neon.vld3dup.v8i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x8x3_t %tmp, %struct.uint8x8x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 3
  ret i8* %updated_src
}

define i8* @test_vld3_dup_u8_update_reg(%struct.uint8x8x3_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld3_dup_u8_update_reg:
; CHECK:         vld3.8 {d16[], d17[], d18[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint8x8x3_t @llvm.arm.neon.vld3dup.v8i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x8x3_t %tmp, %struct.uint8x8x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld4_dup_u16_update(%struct.uint16x4x4_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld4_dup_u16_update:
; CHECK:         vld4.16 {d16[], d17[], d18[], d19[]}, [r1]!
entry:
  %tmp = tail call %struct.uint16x4x4_t @llvm.arm.neon.vld4dup.v4i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x4x4_t %tmp, %struct.uint16x4x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 8
  ret i8* %updated_src
}

define i8* @test_vld4_dup_u16_update_reg(%struct.uint16x4x4_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld4_dup_u16_update_reg:
; CHECK:         vld4.16 {d16[], d17[], d18[], d19[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint16x4x4_t @llvm.arm.neon.vld4dup.v4i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x4x4_t %tmp, %struct.uint16x4x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld4_dup_u32_update(%struct.uint32x2x4_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld4_dup_u32_update:
; CHECK:         vld4.32 {d16[], d17[], d18[], d19[]}, [r1]!
entry:
  %tmp = tail call %struct.uint32x2x4_t @llvm.arm.neon.vld4dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x4_t %tmp, %struct.uint32x2x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 16
  ret i8* %updated_src
}

define i8* @test_vld4_dup_u32_update_reg(%struct.uint32x2x4_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld4_dup_u32_update_reg:
; CHECK:         vld4.32 {d16[], d17[], d18[], d19[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint32x2x4_t @llvm.arm.neon.vld4dup.v2i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x2x4_t %tmp, %struct.uint32x2x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld4_dup_u64_update(%struct.uint64x1x4_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld4_dup_u64_update:
; CHECK:         vld1.64 {d16, d17, d18, d19}, [r1:64]!
entry:
  %tmp = tail call %struct.uint64x1x4_t @llvm.arm.neon.vld4dup.v1i64.p0i8(i8* %src, i32 8)
  store %struct.uint64x1x4_t %tmp, %struct.uint64x1x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 32
  ret i8* %updated_src
}

define i8* @test_vld4_dup_u64_update_reg(%struct.uint64x1x4_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld4_dup_u64_update_reg:
; CHECK:         vld1.64 {d16, d17, d18, d19}, [r1:64], r2
entry:
  %tmp = tail call %struct.uint64x1x4_t @llvm.arm.neon.vld4dup.v1i64.p0i8(i8* %src, i32 8)
  store %struct.uint64x1x4_t %tmp, %struct.uint64x1x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld4_dup_u8_update(%struct.uint8x8x4_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld4_dup_u8_update:
; CHECK:         vld4.8 {d16[], d17[], d18[], d19[]}, [r1]!
entry:
  %tmp = tail call %struct.uint8x8x4_t @llvm.arm.neon.vld4dup.v8i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x8x4_t %tmp, %struct.uint8x8x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 4
  ret i8* %updated_src
}

define i8* @test_vld4_dup_u8_update_reg(%struct.uint8x8x4_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld4_dup_u8_update_reg:
; CHECK:         vld4.8 {d16[], d17[], d18[], d19[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint8x8x4_t @llvm.arm.neon.vld4dup.v8i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x8x4_t %tmp, %struct.uint8x8x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld2q_dup_u16_update(%struct.uint16x8x2_t* %dest, i8* %src, <8 x i16>* %dest0) {
; CHECK-LABEL: test_vld2q_dup_u16_update:
; CHECK:         vld2.16 {d16[], d18[]}, [r1]
; CHECK-NEXT:    vld2.16 {d17[], d19[]}, [r1]!
entry:
  %tmp = tail call %struct.uint16x8x2_t @llvm.arm.neon.vld2dup.v8i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x8x2_t %tmp, %struct.uint16x8x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 4
  ret i8* %updated_src
}

define i8* @test_vld2q_dup_u16_update_reg(%struct.uint16x8x2_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld2q_dup_u16_update_reg:
; CHECK:         vld2.16 {d16[], d18[]}, [r1]
; CHECK-NEXT:    vld2.16 {d17[], d19[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint16x8x2_t @llvm.arm.neon.vld2dup.v8i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x8x2_t %tmp, %struct.uint16x8x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld2q_dup_u32_update(%struct.uint32x4x2_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld2q_dup_u32_update:
; CHECK:         vld2.32 {d16[], d18[]}, [r1]
; CHECK-NEXT:    vld2.32 {d17[], d19[]}, [r1]!
entry:
  %tmp = tail call %struct.uint32x4x2_t @llvm.arm.neon.vld2dup.v4i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x4x2_t %tmp, %struct.uint32x4x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 8
  ret i8* %updated_src
}

define i8* @test_vld2q_dup_u32_update_reg(%struct.uint32x4x2_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld2q_dup_u32_update_reg:
; CHECK:         vld2.32 {d16[], d18[]}, [r1]
; CHECK-NEXT:    vld2.32 {d17[], d19[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint32x4x2_t @llvm.arm.neon.vld2dup.v4i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x4x2_t %tmp, %struct.uint32x4x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld2q_dup_u8_update(%struct.uint8x16x2_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld2q_dup_u8_update:
; CHECK:         vld2.8 {d16[], d18[]}, [r1]
; CHECK-NEXT:    vld2.8 {d17[], d19[]}, [r1]!
entry:
  %tmp = tail call %struct.uint8x16x2_t @llvm.arm.neon.vld2dup.v16i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x16x2_t %tmp, %struct.uint8x16x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 2
  ret i8* %updated_src
}

define i8* @test_vld2q_dup_u8_update_reg(%struct.uint8x16x2_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld2q_dup_u8_update_reg:
; CHECK:         vld2.8 {d16[], d18[]}, [r1]
; CHECK-NEXT:    vld2.8 {d17[], d19[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint8x16x2_t @llvm.arm.neon.vld2dup.v16i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x16x2_t %tmp, %struct.uint8x16x2_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld3q_dup_u16_update(%struct.uint16x8x3_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld3q_dup_u16_update:
; CHECK:         vld3.16 {d16[], d18[], d20[]}, [r1]
; CHECK:         vld3.16 {d17[], d19[], d21[]}, [r1]!
entry:
  %tmp = tail call %struct.uint16x8x3_t @llvm.arm.neon.vld3dup.v8i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x8x3_t %tmp, %struct.uint16x8x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 6
  ret i8* %updated_src
}

define i8* @test_vld3q_dup_u16_update_reg(%struct.uint16x8x3_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld3q_dup_u16_update_reg:
; CHECK:         vld3.16 {d16[], d18[], d20[]}, [r1]
; CHECK-NEXT:    vld3.16 {d17[], d19[], d21[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint16x8x3_t @llvm.arm.neon.vld3dup.v8i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x8x3_t %tmp, %struct.uint16x8x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld3q_dup_u32_update(%struct.uint32x4x3_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld3q_dup_u32_update:
; CHECK:         vld3.32 {d16[], d18[], d20[]}, [r1]
; CHECK:         vld3.32 {d17[], d19[], d21[]}, [r1]!
entry:
  %tmp = tail call %struct.uint32x4x3_t @llvm.arm.neon.vld3dup.v4i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x4x3_t %tmp, %struct.uint32x4x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 12
  ret i8* %updated_src
}

define i8* @test_vld3q_dup_u32_update_reg(%struct.uint32x4x3_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld3q_dup_u32_update_reg:
; CHECK:         vld3.32 {d16[], d18[], d20[]}, [r1]
; CHECK-NEXT:    vld3.32 {d17[], d19[], d21[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint32x4x3_t @llvm.arm.neon.vld3dup.v4i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x4x3_t %tmp, %struct.uint32x4x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld3q_dup_u8_update(%struct.uint8x16x3_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld3q_dup_u8_update:
; CHECK:         vld3.8 {d16[], d18[], d20[]}, [r1]
; CHECK:         vld3.8 {d17[], d19[], d21[]}, [r1]!
entry:
  %tmp = tail call %struct.uint8x16x3_t @llvm.arm.neon.vld3dup.v16i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x16x3_t %tmp, %struct.uint8x16x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 3
  ret i8* %updated_src
}

define i8* @test_vld3q_dup_u8_update_reg(%struct.uint8x16x3_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld3q_dup_u8_update_reg:
; CHECK:         vld3.8 {d16[], d18[], d20[]}, [r1]
; CHECK-NEXT:    vld3.8 {d17[], d19[], d21[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint8x16x3_t @llvm.arm.neon.vld3dup.v16i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x16x3_t %tmp, %struct.uint8x16x3_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld4q_dup_u16_update(%struct.uint16x8x4_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld4q_dup_u16_update:
; CHECK:         vld4.16 {d16[], d18[], d20[], d22[]}, [r1]
; CHECK:         vld4.16 {d17[], d19[], d21[], d23[]}, [r1]!
entry:
  %tmp = tail call %struct.uint16x8x4_t @llvm.arm.neon.vld4dup.v8i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x8x4_t %tmp, %struct.uint16x8x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 8
  ret i8* %updated_src
}

define i8* @test_vld4q_dup_u16_update_reg(%struct.uint16x8x4_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld4q_dup_u16_update_reg:
; CHECK:         vld4.16 {d16[], d18[], d20[], d22[]}, [r1]
; CHECK-NEXT:    vld4.16 {d17[], d19[], d21[], d23[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint16x8x4_t @llvm.arm.neon.vld4dup.v8i16.p0i8(i8* %src, i32 2)
  store %struct.uint16x8x4_t %tmp, %struct.uint16x8x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld4q_dup_u32_update(%struct.uint32x4x4_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld4q_dup_u32_update:
; CHECK:         vld4.32 {d16[], d18[], d20[], d22[]}, [r1]
; CHECK:         vld4.32 {d17[], d19[], d21[], d23[]}, [r1]!
entry:
  %tmp = tail call %struct.uint32x4x4_t @llvm.arm.neon.vld4dup.v4i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x4x4_t %tmp, %struct.uint32x4x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 16
  ret i8* %updated_src
}

define i8* @test_vld4q_dup_u32_update_reg(%struct.uint32x4x4_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld4q_dup_u32_update_reg:
; CHECK:         vld4.32 {d16[], d18[], d20[], d22[]}, [r1]
; CHECK-NEXT:    vld4.32 {d17[], d19[], d21[], d23[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint32x4x4_t @llvm.arm.neon.vld4dup.v4i32.p0i8(i8* %src, i32 4)
  store %struct.uint32x4x4_t %tmp, %struct.uint32x4x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}

define i8* @test_vld4q_dup_u8_update(%struct.uint8x16x4_t* %dest, i8* %src) {
; CHECK-LABEL: test_vld4q_dup_u8_update:
; CHECK:         vld4.8 {d16[], d18[], d20[], d22[]}, [r1]
; CHECK:         vld4.8 {d17[], d19[], d21[], d23[]}, [r1]!
entry:
  %tmp = tail call %struct.uint8x16x4_t @llvm.arm.neon.vld4dup.v16i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x16x4_t %tmp, %struct.uint8x16x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 4
  ret i8* %updated_src
}

define i8* @test_vld4q_dup_u8_update_reg(%struct.uint8x16x4_t* %dest, i8* %src, i32 %inc) {
; CHECK-LABEL: test_vld4q_dup_u8_update_reg:
; CHECK:         vld4.8 {d16[], d18[], d20[], d22[]}, [r1]
; CHECK-NEXT:    vld4.8 {d17[], d19[], d21[], d23[]}, [r1], r2
entry:
  %tmp = tail call %struct.uint8x16x4_t @llvm.arm.neon.vld4dup.v16i8.p0i8(i8* %src, i32 1)
  store %struct.uint8x16x4_t %tmp, %struct.uint8x16x4_t* %dest, align 8
  %updated_src = getelementptr inbounds i8, i8* %src, i32 %inc
  ret i8* %updated_src
}
