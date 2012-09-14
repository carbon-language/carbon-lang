; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

define i32 @test0() {
; CHECK: @test0
; CHECK-NOT: alloca
; CHECK: ret i32

entry:
  %a1 = alloca i32
  %a2 = alloca float

  store i32 0, i32* %a1
  %v1 = load i32* %a1

  store float 0.0, float* %a2
  %v2 = load float * %a2
  %v2.int = bitcast float %v2 to i32
  %sum1 = add i32 %v1, %v2.int

  ret i32 %sum1
}

define i32 @test1() {
; CHECK: @test1
; CHECK-NOT: alloca
; CHECK: ret i32 0

entry:
  %X = alloca { i32, float }
  %Y = getelementptr { i32, float }* %X, i64 0, i32 0
  store i32 0, i32* %Y
  %Z = load i32* %Y
  ret i32 %Z
}

define i64 @test2(i64 %X) {
; CHECK: @test2
; CHECK-NOT: alloca
; CHECK: ret i64 %X

entry:
  %A = alloca [8 x i8]
  %B = bitcast [8 x i8]* %A to i64*
  store i64 %X, i64* %B
  br label %L2

L2:
  %Z = load i64* %B
  ret i64 %Z
}

define void @test3(i8* %dst, i8* %src) {
; CHECK: @test3

entry:
  %a = alloca [300 x i8]
; CHECK-NOT:  alloca
; CHECK:      %[[test3_a1:.*]] = alloca [42 x i8]
; CHECK-NEXT: %[[test3_a2:.*]] = alloca [99 x i8]
; CHECK-NEXT: %[[test3_a3:.*]] = alloca [16 x i8]
; CHECK-NEXT: %[[test3_a4:.*]] = alloca [42 x i8]
; CHECK-NEXT: %[[test3_a5:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test3_a6:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test3_a7:.*]] = alloca [85 x i8]

  %b = getelementptr [300 x i8]* %a, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b, i8* %src, i32 300, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %src, i32 42
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 42
; CHECK-NEXT: %[[test3_r1:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 43
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [99 x i8]* %[[test3_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 99
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 142
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 16
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 158
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 42
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 200
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 207
; CHECK-NEXT: %[[test3_r2:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 208
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 215
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 85

  ; Clobber a single element of the array, this should be promotable.
  %c = getelementptr [300 x i8]* %a, i64 0, i64 42
  store i8 0, i8* %c

  ; Make a sequence of overlapping stores to the array. These overlap both in
  ; forward strides and in shrinking accesses.
  %overlap.1.i8 = getelementptr [300 x i8]* %a, i64 0, i64 142
  %overlap.2.i8 = getelementptr [300 x i8]* %a, i64 0, i64 143
  %overlap.3.i8 = getelementptr [300 x i8]* %a, i64 0, i64 144
  %overlap.4.i8 = getelementptr [300 x i8]* %a, i64 0, i64 145
  %overlap.5.i8 = getelementptr [300 x i8]* %a, i64 0, i64 146
  %overlap.6.i8 = getelementptr [300 x i8]* %a, i64 0, i64 147
  %overlap.7.i8 = getelementptr [300 x i8]* %a, i64 0, i64 148
  %overlap.8.i8 = getelementptr [300 x i8]* %a, i64 0, i64 149
  %overlap.9.i8 = getelementptr [300 x i8]* %a, i64 0, i64 150
  %overlap.1.i16 = bitcast i8* %overlap.1.i8 to i16*
  %overlap.1.i32 = bitcast i8* %overlap.1.i8 to i32*
  %overlap.1.i64 = bitcast i8* %overlap.1.i8 to i64*
  %overlap.2.i64 = bitcast i8* %overlap.2.i8 to i64*
  %overlap.3.i64 = bitcast i8* %overlap.3.i8 to i64*
  %overlap.4.i64 = bitcast i8* %overlap.4.i8 to i64*
  %overlap.5.i64 = bitcast i8* %overlap.5.i8 to i64*
  %overlap.6.i64 = bitcast i8* %overlap.6.i8 to i64*
  %overlap.7.i64 = bitcast i8* %overlap.7.i8 to i64*
  %overlap.8.i64 = bitcast i8* %overlap.8.i8 to i64*
  %overlap.9.i64 = bitcast i8* %overlap.9.i8 to i64*
  store i8 1, i8* %overlap.1.i8
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 0
; CHECK-NEXT: store i8 1, i8* %[[gep]]
  store i16 1, i16* %overlap.1.i16
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [16 x i8]* %[[test3_a3]] to i16*
; CHECK-NEXT: store i16 1, i16* %[[bitcast]]
  store i32 1, i32* %overlap.1.i32
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [16 x i8]* %[[test3_a3]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i64 1, i64* %overlap.1.i64
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [16 x i8]* %[[test3_a3]] to i64*
; CHECK-NEXT: store i64 1, i64* %[[bitcast]]
  store i64 2, i64* %overlap.2.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 2, i64* %[[bitcast]]
  store i64 3, i64* %overlap.3.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 3, i64* %[[bitcast]]
  store i64 4, i64* %overlap.4.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 3
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 4, i64* %[[bitcast]]
  store i64 5, i64* %overlap.5.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 4
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 5, i64* %[[bitcast]]
  store i64 6, i64* %overlap.6.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 5
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 6, i64* %[[bitcast]]
  store i64 7, i64* %overlap.7.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 6
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 7, i64* %[[bitcast]]
  store i64 8, i64* %overlap.8.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 7
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 8, i64* %[[bitcast]]
  store i64 9, i64* %overlap.9.i64
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 8
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i64*
; CHECK-NEXT: store i64 9, i64* %[[bitcast]]

  ; Make two sequences of overlapping stores with more gaps and irregularities.
  %overlap2.1.0.i8 = getelementptr [300 x i8]* %a, i64 0, i64 200
  %overlap2.1.1.i8 = getelementptr [300 x i8]* %a, i64 0, i64 201
  %overlap2.1.2.i8 = getelementptr [300 x i8]* %a, i64 0, i64 202
  %overlap2.1.3.i8 = getelementptr [300 x i8]* %a, i64 0, i64 203

  %overlap2.2.0.i8 = getelementptr [300 x i8]* %a, i64 0, i64 208
  %overlap2.2.1.i8 = getelementptr [300 x i8]* %a, i64 0, i64 209
  %overlap2.2.2.i8 = getelementptr [300 x i8]* %a, i64 0, i64 210
  %overlap2.2.3.i8 = getelementptr [300 x i8]* %a, i64 0, i64 211

  %overlap2.1.0.i16 = bitcast i8* %overlap2.1.0.i8 to i16*
  %overlap2.1.0.i32 = bitcast i8* %overlap2.1.0.i8 to i32*
  %overlap2.1.1.i32 = bitcast i8* %overlap2.1.1.i8 to i32*
  %overlap2.1.2.i32 = bitcast i8* %overlap2.1.2.i8 to i32*
  %overlap2.1.3.i32 = bitcast i8* %overlap2.1.3.i8 to i32*
  store i8 1,  i8*  %overlap2.1.0.i8
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: store i8 1, i8* %[[gep]]
  store i16 1, i16* %overlap2.1.0.i16
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [7 x i8]* %[[test3_a5]] to i16*
; CHECK-NEXT: store i16 1, i16* %[[bitcast]]
  store i32 1, i32* %overlap2.1.0.i32
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [7 x i8]* %[[test3_a5]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i32 2, i32* %overlap2.1.1.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 2, i32* %[[bitcast]]
  store i32 3, i32* %overlap2.1.2.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 3, i32* %[[bitcast]]
  store i32 4, i32* %overlap2.1.3.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 3
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 4, i32* %[[bitcast]]

  %overlap2.2.0.i32 = bitcast i8* %overlap2.2.0.i8 to i32*
  %overlap2.2.1.i16 = bitcast i8* %overlap2.2.1.i8 to i16*
  %overlap2.2.1.i32 = bitcast i8* %overlap2.2.1.i8 to i32*
  %overlap2.2.2.i32 = bitcast i8* %overlap2.2.2.i8 to i32*
  %overlap2.2.3.i32 = bitcast i8* %overlap2.2.3.i8 to i32*
  store i32 1, i32* %overlap2.2.0.i32
; CHECK-NEXT: %[[bitcast:.*]] = bitcast [7 x i8]* %[[test3_a6]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i8 1,  i8*  %overlap2.2.1.i8
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: store i8 1, i8* %[[gep]]
  store i16 1, i16* %overlap2.2.1.i16
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 1, i16* %[[bitcast]]
  store i32 1, i32* %overlap2.2.1.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 1, i32* %[[bitcast]]
  store i32 3, i32* %overlap2.2.2.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 2
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 3, i32* %[[bitcast]]
  store i32 4, i32* %overlap2.2.3.i32
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 3
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT: store i32 4, i32* %[[bitcast]]

  %overlap2.prefix = getelementptr i8* %overlap2.1.1.i8, i64 -4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.prefix, i8* %src, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a4]], i64 0, i64 39
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %src, i32 3
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 3
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 5

  ; Bridge between the overlapping areas
  call void @llvm.memset.p0i8.i32(i8* %overlap2.1.2.i8, i8 42, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 2
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[gep]], i8 42, i32 5
; ...promoted i8 store...
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[gep]], i8 42, i32 2

  ; Entirely within the second overlap.
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.2.1.i8, i8* %src, i32 5, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 1
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 5

  ; Trailing past the second overlap.
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %overlap2.2.2.i8, i8* %src, i32 8, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 2
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 5
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 5
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 3

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %b, i32 300, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[gep]], i32 42
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 42
; CHECK-NEXT: store i8 0, i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 43
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [99 x i8]* %[[test3_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 99
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 142
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [16 x i8]* %[[test3_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 16
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 158
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [42 x i8]* %[[test3_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 42
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 200
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 207
; CHECK-NEXT: store i8 42, i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 208
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test3_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 215
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [85 x i8]* %[[test3_a7]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 85

  ret void
}

define void @test4(i8* %dst, i8* %src) {
; CHECK: @test4

entry:
  %a = alloca [100 x i8]
; CHECK-NOT:  alloca
; CHECK:      %[[test4_a1:.*]] = alloca [20 x i8]
; CHECK-NEXT: %[[test4_a2:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a3:.*]] = alloca [10 x i8]
; CHECK-NEXT: %[[test4_a4:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a5:.*]] = alloca [7 x i8]
; CHECK-NEXT: %[[test4_a6:.*]] = alloca [40 x i8]

  %b = getelementptr [100 x i8]* %a, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %b, i8* %src, i32 100, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [20 x i8]* %[[test4_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep]], i8* %src, i32 20
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 20
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r1:.*]] = load i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 22
; CHECK-NEXT: %[[test4_r2:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 23
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 30
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [10 x i8]* %[[test4_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 10
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 40
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r3:.*]] = load i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 42
; CHECK-NEXT: %[[test4_r4:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 43
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 50
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: %[[test4_r5:.*]] = load i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %src, i64 52
; CHECK-NEXT: %[[test4_r6:.*]] = load i8* %[[gep]]
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 53
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds i8* %src, i64 60
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [40 x i8]* %[[test4_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 40

  %a.src.1 = getelementptr [100 x i8]* %a, i64 0, i64 20
  %a.dst.1 = getelementptr [100 x i8]* %a, i64 0, i64 40
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.dst.1, i8* %a.src.1, i32 10, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7

  ; Clobber a single element of the array, this should be promotable, and be deleted.
  %c = getelementptr [100 x i8]* %a, i64 0, i64 42
  store i8 0, i8* %c

  %a.src.2 = getelementptr [100 x i8]* %a, i64 0, i64 50
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %a.dst.1, i8* %a.src.2, i32 10, i32 1, i1 false)
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %b, i32 100, i32 1, i1 false)
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds [20 x i8]* %[[test4_a1]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[gep]], i32 20
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 20
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r1]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 22
; CHECK-NEXT: store i8 %[[test4_r2]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 23
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a2]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 30
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [10 x i8]* %[[test4_a3]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 10
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 40
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r5]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 42
; CHECK-NEXT: store i8 %[[test4_r6]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 43
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a4]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 50
; CHECK-NEXT: %[[bitcast:.*]] = bitcast i8* %[[gep]] to i16*
; CHECK-NEXT: store i16 %[[test4_r5]], i16* %[[bitcast]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds i8* %dst, i64 52
; CHECK-NEXT: store i8 %[[test4_r6]], i8* %[[gep]]
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 53
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [7 x i8]* %[[test4_a5]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 7
; CHECK-NEXT: %[[gep_dst:.*]] = getelementptr inbounds i8* %dst, i64 60
; CHECK-NEXT: %[[gep_src:.*]] = getelementptr inbounds [40 x i8]* %[[test4_a6]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[gep_dst]], i8* %[[gep_src]], i32 40

  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define i16 @test5() {
; CHECK: @test5
; CHECK: alloca float
; CHECK: ret i16 %

entry:
  %a = alloca [4 x i8]
  %fptr = bitcast [4 x i8]* %a to float*
  store float 0.0, float* %fptr
  %ptr = getelementptr [4 x i8]* %a, i32 0, i32 2
  %iptr = bitcast i8* %ptr to i16*
  %val = load i16* %iptr
  ret i16 %val
}

define i32 @test6() {
; CHECK: @test6
; CHECK: alloca i32
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: load i32*
; CHECK-NEXT: ret i32

entry:
  %a = alloca [4 x i8]
  %ptr = getelementptr [4 x i8]* %a, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 42, i32 4, i32 1, i1 true)
  %iptr = bitcast i8* %ptr to i32*
  %val = load i32* %iptr
  ret i32 %val
}

define void @test7(i8* %src, i8* %dst) {
; CHECK: @test7
; CHECK: alloca i32
; CHECK-NEXT: bitcast i8* %src to i32*
; CHECK-NEXT: load volatile i32*
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: bitcast i8* %dst to i32*
; CHECK-NEXT: load volatile i32*
; CHECK-NEXT: store volatile i32
; CHECK-NEXT: ret

entry:
  %a = alloca [4 x i8]
  %ptr = getelementptr [4 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 true)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 true)
  ret void
}


%S1 = type { i32, i32, [16 x i8] }
%S2 = type { %S1*, %S2* }

define %S2 @test8(%S2* %s2) {
; CHECK: @test8
entry:
  %new = alloca %S2
; CHECK-NOT: alloca

  %s2.next.ptr = getelementptr %S2* %s2, i64 0, i32 1
  %s2.next = load %S2** %s2.next.ptr
; CHECK:      %[[gep:.*]] = getelementptr %S2* %s2, i64 0, i32 1
; CHECK-NEXT: %[[next:.*]] = load %S2** %[[gep]]

  %s2.next.s1.ptr = getelementptr %S2* %s2.next, i64 0, i32 0
  %s2.next.s1 = load %S1** %s2.next.s1.ptr
  %new.s1.ptr = getelementptr %S2* %new, i64 0, i32 0
  store %S1* %s2.next.s1, %S1** %new.s1.ptr
  %s2.next.next.ptr = getelementptr %S2* %s2.next, i64 0, i32 1
  %s2.next.next = load %S2** %s2.next.next.ptr
  %new.next.ptr = getelementptr %S2* %new, i64 0, i32 1
  store %S2* %s2.next.next, %S2** %new.next.ptr
; CHECK-NEXT: %[[gep:.*]] = getelementptr %S2* %[[next]], i64 0, i32 0
; CHECK-NEXT: %[[next_s1:.*]] = load %S1** %[[gep]]
; CHECK-NEXT: %[[gep:.*]] = getelementptr %S2* %[[next]], i64 0, i32 1
; CHECK-NEXT: %[[next_next:.*]] = load %S2** %[[gep]]

  %new.s1 = load %S1** %new.s1.ptr
  %result1 = insertvalue %S2 undef, %S1* %new.s1, 0
; CHECK-NEXT: %[[result1:.*]] = insertvalue %S2 undef, %S1* %[[next_s1]], 0
  %new.next = load %S2** %new.next.ptr
  %result2 = insertvalue %S2 %result1, %S2* %new.next, 1
; CHECK-NEXT: %[[result2:.*]] = insertvalue %S2 %[[result1]], %S2* %[[next_next]], 1
  ret %S2 %result2
; CHECK-NEXT: ret %S2 %[[result2]]
}

define i64 @test9() {
; Ensure we can handle loads off the end of an alloca even when wrapped in
; weird bit casts and types. The result is undef, but this shouldn't crash
; anything.
; CHECK: @test9
; CHECK-NOT: alloca
; CHECK: ret i64 undef

entry:
  %a = alloca { [3 x i8] }
  %gep1 = getelementptr inbounds { [3 x i8] }* %a, i32 0, i32 0, i32 0
  store i8 0, i8* %gep1, align 1
  %gep2 = getelementptr inbounds { [3 x i8] }* %a, i32 0, i32 0, i32 1
  store i8 0, i8* %gep2, align 1
  %gep3 = getelementptr inbounds { [3 x i8] }* %a, i32 0, i32 0, i32 2
  store i8 26, i8* %gep3, align 1
  %cast = bitcast { [3 x i8] }* %a to { i64 }*
  %elt = getelementptr inbounds { i64 }* %cast, i32 0, i32 0
  %result = load i64* %elt
  ret i64 %result
}

define %S2* @test10() {
; CHECK: @test10
; CHECK-NOT: alloca %S2*
; CHECK: ret %S2* null

entry:
  %a = alloca [8 x i8]
  %ptr = getelementptr [8 x i8]* %a, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 0, i32 8, i32 1, i1 false)
  %s2ptrptr = bitcast i8* %ptr to %S2**
  %s2ptr = load %S2** %s2ptrptr
  ret %S2* %s2ptr
}

define i32 @test11() {
; CHECK: @test11
; CHECK-NOT: alloca
; CHECK: ret i32 0

entry:
  %X = alloca i32
  br i1 undef, label %good, label %bad

good:
  %Y = getelementptr i32* %X, i64 0
  store i32 0, i32* %Y
  %Z = load i32* %Y
  ret i32 %Z

bad:
  %Y2 = getelementptr i32* %X, i64 1
  store i32 0, i32* %Y2
  %Z2 = load i32* %Y2
  ret i32 %Z2
}

define i32 @test12() {
; CHECK: @test12
; CHECK: alloca i24
;
; FIXME: SROA should promote accesses to this into whole i24 operations instead
; of i8 operations.
; CHECK: store i8 0
; CHECK: store i8 0
; CHECK: store i8 0
;
; CHECK: load i24*

entry:
  %a = alloca [3 x i8]
  %b0ptr = getelementptr [3 x i8]* %a, i64 0, i32 0
  store i8 0, i8* %b0ptr
  %b1ptr = getelementptr [3 x i8]* %a, i64 0, i32 1
  store i8 0, i8* %b1ptr
  %b2ptr = getelementptr [3 x i8]* %a, i64 0, i32 2
  store i8 0, i8* %b2ptr
  %iptr = bitcast [3 x i8]* %a to i24*
  %i = load i24* %iptr
  %ret = zext i24 %i to i32
  ret i32 %ret
}

define i32 @test13() {
; Ensure we don't crash and handle undefined loads that straddle the end of the
; allocation.
; CHECK: @test13
; CHECK: %[[ret:.*]] = zext i16 undef to i32
; CHECK: ret i32 %[[ret]]

entry:
  %a = alloca [3 x i8]
  %b0ptr = getelementptr [3 x i8]* %a, i64 0, i32 0
  store i8 0, i8* %b0ptr
  %b1ptr = getelementptr [3 x i8]* %a, i64 0, i32 1
  store i8 0, i8* %b1ptr
  %b2ptr = getelementptr [3 x i8]* %a, i64 0, i32 2
  store i8 0, i8* %b2ptr
  %iptrcast = bitcast [3 x i8]* %a to i16*
  %iptrgep = getelementptr i16* %iptrcast, i64 1
  %i = load i16* %iptrgep
  %ret = zext i16 %i to i32
  ret i32 %ret
}

%test14.struct = type { [3 x i32] }

define void @test14(...) nounwind uwtable {
; This is a strange case where we split allocas into promotable partitions, but
; also gain enough data to prove they must be dead allocas due to GEPs that walk
; across two adjacent allocas. Test that we don't try to promote or otherwise
; do bad things to these dead allocas, they should just be removed.
; CHECK: @test14
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void

entry:
  %a = alloca %test14.struct
  %p = alloca %test14.struct*
  %0 = bitcast %test14.struct* %a to i8*
  %1 = getelementptr i8* %0, i64 12
  %2 = bitcast i8* %1 to %test14.struct*
  %3 = getelementptr inbounds %test14.struct* %2, i32 0, i32 0
  %4 = getelementptr inbounds %test14.struct* %a, i32 0, i32 0
  %5 = bitcast [3 x i32]* %3 to i32*
  %6 = bitcast [3 x i32]* %4 to i32*
  %7 = load i32* %6, align 4
  store i32 %7, i32* %5, align 4
  %8 = getelementptr inbounds i32* %5, i32 1
  %9 = getelementptr inbounds i32* %6, i32 1
  %10 = load i32* %9, align 4
  store i32 %10, i32* %8, align 4
  %11 = getelementptr inbounds i32* %5, i32 2
  %12 = getelementptr inbounds i32* %6, i32 2
  %13 = load i32* %12, align 4
  store i32 %13, i32* %11, align 4
  ret void
}

define i32 @test15(i1 %flag) nounwind uwtable {
; Ensure that when there are dead instructions using an alloca that are not
; loads or stores we still delete them during partitioning and rewriting.
; Otherwise we'll go to promote them while thy still have unpromotable uses.
; CHECK: @test15
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   br label %loop

entry:
  %l0 = alloca i64
  %l1 = alloca i64
  %l2 = alloca i64
  %l3 = alloca i64
  br label %loop

loop:
  %dead3 = phi i8* [ %gep3, %loop ], [ null, %entry ]

  store i64 1879048192, i64* %l0, align 8
  %bc0 = bitcast i64* %l0 to i8*
  %gep0 = getelementptr i8* %bc0, i64 3
  %dead0 = bitcast i8* %gep0 to i64*

  store i64 1879048192, i64* %l1, align 8
  %bc1 = bitcast i64* %l1 to i8*
  %gep1 = getelementptr i8* %bc1, i64 3
  %dead1 = getelementptr i8* %gep1, i64 1

  store i64 1879048192, i64* %l2, align 8
  %bc2 = bitcast i64* %l2 to i8*
  %gep2.1 = getelementptr i8* %bc2, i64 1
  %gep2.2 = getelementptr i8* %bc2, i64 3
  ; Note that this select should get visited multiple times due to using two
  ; different GEPs off the same alloca. We should only delete it once.
  %dead2 = select i1 %flag, i8* %gep2.1, i8* %gep2.2

  store i64 1879048192, i64* %l3, align 8
  %bc3 = bitcast i64* %l3 to i8*
  %gep3 = getelementptr i8* %bc3, i64 3

  br label %loop
}

define void @test16(i8* %src, i8* %dst) {
; Ensure that we can promote an alloca of [3 x i8] to an i24 SSA value.
; CHECK: @test16
; CHECK-NOT: alloca
; CHECK:      %[[srccast:.*]] = bitcast i8* %src to i24*
; CHECK-NEXT: load i24* %[[srccast]]
; CHECK-NEXT: %[[dstcast:.*]] = bitcast i8* %dst to i24*
; CHECK-NEXT: store i24 0, i24* %[[dstcast]]
; CHECK-NEXT: ret void

entry:
  %a = alloca [3 x i8]
  %ptr = getelementptr [3 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 false)
  %cast = bitcast i8* %ptr to i24*
  store i24 0, i24* %cast
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 false)
  ret void
}

define void @test17(i8* %src, i8* %dst) {
; Ensure that we can rewrite unpromotable memcpys which extend past the end of
; the alloca.
; CHECK: @test17
; CHECK:      %[[a:.*]] = alloca [3 x i8]
; CHECK-NEXT: %[[ptr:.*]] = getelementptr [3 x i8]* %[[a]], i32 0, i32 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[ptr]], i8* %src,
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[ptr]],
; CHECK-NEXT: ret void

entry:
  %a = alloca [3 x i8]
  %ptr = getelementptr [3 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 4, i32 1, i1 true)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 4, i32 1, i1 true)
  ret void
}

define void @test18(i8* %src, i8* %dst, i32 %size) {
; Preserve transfer instrinsics with a variable size, even if they overlap with
; fixed size operations. Further, continue to split and promote allocas preceding
; the variable sized intrinsic.
; CHECK: @test18
; CHECK:      %[[a:.*]] = alloca [34 x i8]
; CHECK:      %[[srcgep1:.*]] = getelementptr inbounds i8* %src, i64 4
; CHECK-NEXT: %[[srccast1:.*]] = bitcast i8* %[[srcgep1]] to i32*
; CHECK-NEXT: %[[srcload:.*]] = load i32* %[[srccast1]]
; CHECK-NEXT: %[[agep1:.*]] = getelementptr inbounds [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %[[agep1]], i8* %src, i32 %size,
; CHECK-NEXT: %[[agep2:.*]] = getelementptr inbounds [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %[[agep2]], i8 42, i32 %size,
; CHECK-NEXT: %[[dstcast1:.*]] = bitcast i8* %dst to i32*
; CHECK-NEXT: store i32 42, i32* %[[dstcast1]]
; CHECK-NEXT: %[[dstgep1:.*]] = getelementptr inbounds i8* %dst, i64 4
; CHECK-NEXT: %[[dstcast2:.*]] = bitcast i8* %[[dstgep1]] to i32*
; CHECK-NEXT: store i32 %[[srcload]], i32* %[[dstcast2]]
; CHECK-NEXT: %[[agep3:.*]] = getelementptr inbounds [34 x i8]* %[[a]], i64 0, i64 0
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %[[agep3]], i32 %size,
; CHECK-NEXT: ret void

entry:
  %a = alloca [42 x i8]
  %ptr = getelementptr [42 x i8]* %a, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %src, i32 8, i32 1, i1 false)
  %ptr2 = getelementptr [42 x i8]* %a, i32 0, i32 8
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr2, i8* %src, i32 %size, i32 1, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %ptr2, i8 42, i32 %size, i32 1, i1 false)
  %cast = bitcast i8* %ptr to i32*
  store i32 42, i32* %cast
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr, i32 8, i32 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %ptr2, i32 %size, i32 1, i1 false)
  ret void
}

