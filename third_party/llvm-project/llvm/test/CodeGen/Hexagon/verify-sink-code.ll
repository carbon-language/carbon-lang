; RUN: llc -O3 -march=hexagon -verify-machineinstrs < %s
; REQUIRES: asserts
; Check for successful compilation.

target triple = "hexagon"

%s.0 = type { %s.1, [128 x %s.0*], i32, i32, i32, %s.6, i32, i32, i32, i32, i32, i32, i32, i32, i32, [1 x %s.9], %s.9*, [1 x %s.12], %s.12*, i32, [4 x [4 x [4 x i32]]*], [2 x [8 x [8 x i32]]*], [4 x [16 x i32]*], [2 x [64 x i32]*], [4 x [16 x i16]*], [2 x [64 x i16]*], [4 x [16 x i16]*], [2 x [64 x i16]*], [2 x [64 x i32]], [2 x [64 x i32]], [2 x i32], %s.13, %s.15, %s.16, %s.17*, %s.17*, i32, [19 x %s.17*], i32, [19 x %s.17*], [2 x i32], [4 x i8], %s.18, %s.20, %s.23*, %s.24, [7 x void (i8*)*], [7 x void (i8*)*], [12 x void (i8*, i8*)*], [12 x void (i8*)*], %s.26, %s.27, %s.28, %s.29, %s.30, %s.32, %s.33, [5 x %s.34*], %s.34*, [15 x %s.34*], [3 x %s.34*], [7 x %s.34*], [8 x i8] }
%s.1 = type { i32, i32, i32, i32, i32, i32, i32, i32, %s.2, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, [16 x i8], [16 x i8], [16 x i8], [16 x i8], [64 x i8], [64 x i8], void (i8*, i32, i8*, i8*)*, i8*, i32, i32, %s.3, %s.4, i32, i32, i32 }
%s.2 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%s.3 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, i32 }
%s.4 = type { i32, i32, i32, i32, i32, i32, float, float, i32, i32, float, float, float, i32, i8*, i32, i8*, i8*, float, float, float, %s.5*, i32, i8* }
%s.5 = type { i32, i32, i32, i32, float }
%s.6 = type { i32, [8 x %s.7], i32, i8*, %s.8, i32 }
%s.7 = type { i32, i32, i32, i8* }
%s.8 = type { i8*, i8*, i8*, i32, i32 }
%s.9 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [256 x i32], i32, i32, i32, i32, i32, i32, i32, i32, %s.10, i32, %s.11, i32 }
%s.10 = type { i32, i32, i32, i32 }
%s.11 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%s.12 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [6 x i8*] }
%s.13 = type { %s.9*, %s.12*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, i32, i32, i32, i32, i32, i32, [2 x [16 x %s.14]], i32, i32, i32, i32, i32, i32, i32, i32 }
%s.14 = type { i32, i32 }
%s.15 = type { [460 x i8], i32, i32, i32, i32, i32, i8*, i8*, i8* }
%s.16 = type { [19 x %s.17*], [19 x %s.17*], [292 x %s.17*], %s.17*, [18 x %s.17*], i32, i32, i32, i32, i32, i32, i32 }
%s.17 = type { i32, i32, i32, i64, i32, i32, i32, float, i32, [4 x i32], [4 x i32], i32, i32, [4 x i8*], [4 x i8*], [4 x i8*], i16*, [8 x i8*], [4 x i8*], i8*, [2 x [2 x i16]*], [2 x i8*], [2 x i32], [2 x [16 x i32]], [18 x [18 x i32]], i32, [18 x i32], [18 x [18 x i32*]], i32*, i32*, i32*, i32, i32, i32, i32 }
%s.18 = type { [16 x i32], [2 x [4 x i32]], [4 x [64 x i32]], [24 x %s.19] }
%s.19 = type { [16 x i32] }
%s.20 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [2 x i32], [2 x i32], [2 x i32], [2 x i32], i32, [4 x i32], [16 x i32], i32, i32, i32, i32, i32, i32, i8*, i8*, i16*, [7 x i8]*, [24 x i8]*, i8*, [2 x [2 x i16]*], [2 x [2 x i16]*], [2 x i8*], [2 x [32 x [2 x i16]*]], i8*, i8*, [2 x [3 x i8*]], [16 x i8]*, i32, i32, [4 x i32], i32, i32, i32, i32, i32, [8 x i8], %s.21, %s.22, i32, i32, i32, i32, i32, i32, i32, i32, [16 x [2 x i32]], [32 x [4 x i32]], [2 x i32], [16 x i32], [4 x i8] }
%s.21 = type { [384 x i8], [864 x i8], [3 x i8*], [3 x i8*], [2 x i32], [2 x [32 x [6 x i8*]]], [2 x [16 x i16*]], [3 x i32], [4 x i8] }
%s.22 = type { [48 x i32], [48 x i32], [2 x [48 x i8]], [2 x [48 x [2 x i16]]], [2 x [48 x [2 x i16]]], [48 x i8], [2 x [48 x [2 x i16]]], [2 x [48 x i8]], [2 x i32], i32, i32, i32 }
%s.23 = type opaque
%s.24 = type { %s.25, [5 x i32], [5 x i64], [5 x i32], [5 x i64], [5 x float], [5 x float], [5 x float], [5 x float], [5 x float], [5 x [19 x i64]], [2 x i64], [2 x [7 x i64]], [2 x [32 x i64]], [2 x i32], [2 x i32] }
%s.25 = type { i32, i32, i32, i32, [19 x i32], i32, i32, i32, [2 x i32], [7 x i32], [32 x i32], i32, i32, i32, [2 x i32] }
%s.26 = type { [7 x i32 (i8*, i32, i8*, i32)*], [7 x i32 (i8*, i32, i8*, i32)*], [7 x i32 (i8*, i32, i8*, i32)*], [7 x i32 (i8*, i32, i8*, i32)*], [4 x i32 (i8*, i32, i8*, i32)*], [7 x i32 (i8*, i32, i8*, i32)*], [7 x i32 (i8*, i32, i8*, i32)*], void (i8*, i32, i8*, i32, [4 x i32]*)*, float ([4 x i32]*, [4 x i32]*, i32)*, [7 x i32 (i8*, i32, i8*, i32, i32)*], [7 x void (i8*, i8*, i8*, i8*, i32, i32*)*], [7 x void (i8*, i8*, i8*, i8*, i8*, i32, i32*)*], [7 x void (i32*, i16*, i32, i16*, i32)*], void (i8*, i8*, i32*)*, void (i8*, i8*, i32*)*, void (i8*, i8*, i32*)*, void (i8*, i8*, i32*)* }
%s.27 = type { void (i8**, i32, i8*, i32, i32, i32, i32, i32)*, i8* (i8**, i32, i8*, i32*, i32, i32, i32, i32)*, void (i8*, i32, i8*, i32, i32, i32, i32, i32)*, [10 x void (i8*, i32, i8*, i32)*], [10 x void (i8*, i32, i8*, i32, i32)*], [7 x void (i8*, i32, i8*, i32, i32)*], void (i8*, i32, i8*, i32, i32, i32)*, void (i8*, i32, i8*, i32, i32)*, void (i8*, i32, i32)* }
%s.28 = type { void ([4 x i16]*, i8*, i8*)*, void (i8*, [4 x i16]*)*, void ([4 x [4 x i16]]*, i8*, i8*)*, void (i8*, [4 x [4 x i16]]*)*, void ([4 x [4 x i16]]*, i8*, i8*)*, void (i8*, [4 x [4 x i16]]*)*, void ([8 x i16]*, i8*, i8*)*, void (i8*, [8 x i16]*)*, void ([8 x [8 x i16]]*, i8*, i8*)*, void (i8*, [8 x [8 x i16]]*)*, void ([4 x i16]*)*, void ([4 x i16]*)*, void ([2 x i16]*)*, void ([2 x i16]*)* }
%s.29 = type { void (i32*, [8 x i16]*)*, void (i32*, [4 x i16]*)*, void (i32*, [4 x i16]*)*, void (i32*, i8*, i8*)*, void (i32*, i8*, i8*)* }
%s.30 = type { [9 x void (%s.27*, %s.17*, %s.31*, i32, i32)*] }
%s.31 = type { i32, i32, [4 x i32], [4 x i8*] }
%s.32 = type { void ([8 x i16]*, i16*, i16*)*, void ([4 x i16]*, i16*, i16*)*, void ([4 x i16]*, i32, i32)*, void ([2 x i16]*, i32, i32)*, void ([4 x i16]*, [4 x [4 x i32]]*, i32)*, void ([8 x i16]*, [8 x [8 x i32]]*, i32)* }
%s.33 = type { void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32, i8*)*, void (i8*, i32, i32, i32)*, void (i8*, i32, i32, i32)*, void (i8*, i32, i32, i32)*, void (i8*, i32, i32, i32)* }
%s.34 = type opaque

@g0 = private unnamed_addr constant [148 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", align 8
@g1 = private unnamed_addr constant [27 x i8] c"yyyyyyyyyyyyyyyyyyyyyyyyyy\00", align 8
@g2 = private unnamed_addr constant [148 x i8] c"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\00", align 8

; Function Attrs: nounwind
define void @f0(%s.0* %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 1
  %v1 = load i32, i32* %v0, align 4
  %v2 = mul nsw i32 %v1, %a2
  %v3 = add nsw i32 %v2, %a1
  %v4 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 3
  %v5 = load i32, i32* %v4, align 4
  %v6 = mul nsw i32 %v5, %a2
  %v7 = add nsw i32 %v6, %a1
  %v8 = mul nsw i32 %v7, 4
  %v9 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 2
  %v10 = load i32, i32* %v9, align 4
  %v11 = mul nsw i32 %v10, %a2
  %v12 = add nsw i32 %v11, %a1
  %v13 = mul nsw i32 %v12, 2
  %v14 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 14
  %v15 = load i32, i32* %v14, align 4
  %v16 = shl i32 1, %v15
  %v17 = sub nsw i32 %a2, %v16
  %v18 = mul nsw i32 %v17, %v1
  %v19 = add nsw i32 %v18, %a1
  %v20 = mul nsw i32 %v1, 2
  %v21 = icmp eq i32 %v10, %v20
  br i1 %v21, label %b2, label %b1

b1:                                               ; preds = %b0
  tail call void @f1(i8* getelementptr inbounds ([148 x i8], [148 x i8]* @g0, i32 0, i32 0), i8* getelementptr inbounds ([27 x i8], [27 x i8]* @g1, i32 0, i32 0)) #2
  %v22 = load i32, i32* %v4, align 4
  %v23 = load i32, i32* %v0, align 4
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v24 = phi i32 [ %v1, %b0 ], [ %v23, %b1 ]
  %v25 = phi i32 [ %v5, %b0 ], [ %v22, %b1 ]
  %v26 = mul nsw i32 %v24, 4
  %v27 = icmp eq i32 %v25, %v26
  br i1 %v27, label %b4, label %b3

b3:                                               ; preds = %b2
  tail call void @f1(i8* getelementptr inbounds ([148 x i8], [148 x i8]* @g2, i32 0, i32 0), i8* getelementptr inbounds ([27 x i8], [27 x i8]* @g1, i32 0, i32 0)) #2
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v28 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 4
  store i32 %a1, i32* %v28, align 4
  %v29 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 5
  store i32 %a2, i32* %v29, align 4
  %v30 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 6
  store i32 %v3, i32* %v30, align 4
  %v31 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 7
  store i32 %v13, i32* %v31, align 4
  %v32 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 8
  store i32 %v8, i32* %v32, align 4
  %v33 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 29
  store i32 %v19, i32* %v33, align 4
  %v34 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 21
  store i32 0, i32* %v34, align 4
  %v35 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 31, i32 3
  %v36 = load i32, i32* %v35, align 4
  %v37 = icmp slt i32 %v19, %v36
  br i1 %v37, label %b6, label %b5

b5:                                               ; preds = %b4
  %v38 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 30
  %v39 = load i8*, i8** %v38, align 4
  %v40 = getelementptr inbounds i8, i8* %v39, i32 %v19
  %v41 = load i8, i8* %v40, align 1
  %v42 = sext i8 %v41 to i32
  %v43 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 24
  store i32 %v42, i32* %v43, align 4
  store i32 2, i32* %v34, align 4
  %v44 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 33
  %v45 = load [7 x i8]*, [7 x i8]** %v44, align 4
  %v46 = getelementptr inbounds [7 x i8], [7 x i8]* %v45, i32 %v19, i32 0
  %v47 = load i8, i8* %v46, align 1
  %v48 = sext i8 %v47 to i32
  %v49 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 54, i32 0, i32 4
  store i32 %v48, i32* %v49, align 4
  %v50 = getelementptr inbounds [7 x i8], [7 x i8]* %v45, i32 %v19, i32 1
  %v51 = load i8, i8* %v50, align 1
  %v52 = sext i8 %v51 to i32
  %v53 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 54, i32 0, i32 5
  store i32 %v52, i32* %v53, align 4
  %v54 = getelementptr inbounds [7 x i8], [7 x i8]* %v45, i32 %v19, i32 2
  %v55 = load i8, i8* %v54, align 1
  %v56 = sext i8 %v55 to i32
  %v57 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 54, i32 0, i32 6
  store i32 %v56, i32* %v57, align 4
  %v58 = getelementptr inbounds [7 x i8], [7 x i8]* %v45, i32 %v19, i32 3
  %v59 = load i8, i8* %v58, align 1
  %v60 = sext i8 %v59 to i32
  %v61 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 54, i32 0, i32 7
  store i32 %v60, i32* %v61, align 4
  br label %b7

b6:                                               ; preds = %b4
  %v62 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 24
  store i32 -1, i32* %v62, align 4
  %v63 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 43, i32 54, i32 0, i32 4
  %v64 = bitcast i32* %v63 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %v64, i8 -1, i64 16, i1 false)
  br label %b7

b7:                                               ; preds = %b6, %b5
  ret void
}

; Function Attrs: nounwind
declare void @f1(i8*, i8*) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }
