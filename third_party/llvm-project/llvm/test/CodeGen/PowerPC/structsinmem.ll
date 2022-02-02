; RUN: llc -verify-machineinstrs -mcpu=ppc64 -O0 -frame-pointer=all -fast-isel=false < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.s1 = type { i8 }
%struct.s2 = type { i16 }
%struct.s4 = type { i32 }
%struct.t1 = type { i8 }
%struct.t3 = type <{ i16, i8 }>
%struct.t5 = type <{ i32, i8 }>
%struct.t6 = type <{ i32, i16 }>
%struct.t7 = type <{ i32, i16, i8 }>
%struct.s3 = type { i16, i8 }
%struct.s5 = type { i32, i8 }
%struct.s6 = type { i32, i16 }
%struct.s7 = type { i32, i16, i8 }
%struct.t2 = type <{ i16 }>
%struct.t4 = type <{ i32 }>

@caller1.p1 = private unnamed_addr constant %struct.s1 { i8 1 }, align 1
@caller1.p2 = private unnamed_addr constant %struct.s2 { i16 2 }, align 2
@caller1.p3 = private unnamed_addr constant { i16, i8, i8 } { i16 4, i8 8, i8 undef }, align 2
@caller1.p4 = private unnamed_addr constant %struct.s4 { i32 16 }, align 4
@caller1.p5 = private unnamed_addr constant { i32, i8, [3 x i8] } { i32 32, i8 64, [3 x i8] undef }, align 4
@caller1.p6 = private unnamed_addr constant { i32, i16, [2 x i8] } { i32 128, i16 256, [2 x i8] undef }, align 4
@caller1.p7 = private unnamed_addr constant { i32, i16, i8, i8 } { i32 512, i16 1024, i8 -3, i8 undef }, align 4
@caller2.p1 = private unnamed_addr constant %struct.t1 { i8 1 }, align 1
@caller2.p2 = private unnamed_addr constant { i16 } { i16 2 }, align 1
@caller2.p3 = private unnamed_addr constant %struct.t3 <{ i16 4, i8 8 }>, align 1
@caller2.p4 = private unnamed_addr constant { i32 } { i32 16 }, align 1
@caller2.p5 = private unnamed_addr constant %struct.t5 <{ i32 32, i8 64 }>, align 1
@caller2.p6 = private unnamed_addr constant %struct.t6 <{ i32 128, i16 256 }>, align 1
@caller2.p7 = private unnamed_addr constant %struct.t7 <{ i32 512, i16 1024, i8 -3 }>, align 1

define i32 @caller1() nounwind {
entry:
  %p1 = alloca %struct.s1, align 1
  %p2 = alloca %struct.s2, align 2
  %p3 = alloca %struct.s3, align 2
  %p4 = alloca %struct.s4, align 4
  %p5 = alloca %struct.s5, align 4
  %p6 = alloca %struct.s6, align 4
  %p7 = alloca %struct.s7, align 4
  %0 = bitcast %struct.s1* %p1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* getelementptr inbounds (%struct.s1, %struct.s1* @caller1.p1, i32 0, i32 0), i64 1, i1 false)
  %1 = bitcast %struct.s2* %p2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 2 %1, i8* align 2 bitcast (%struct.s2* @caller1.p2 to i8*), i64 2, i1 false)
  %2 = bitcast %struct.s3* %p3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 2 %2, i8* align 2 bitcast ({ i16, i8, i8 }* @caller1.p3 to i8*), i64 4, i1 false)
  %3 = bitcast %struct.s4* %p4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 bitcast (%struct.s4* @caller1.p4 to i8*), i64 4, i1 false)
  %4 = bitcast %struct.s5* %p5 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 bitcast ({ i32, i8, [3 x i8] }* @caller1.p5 to i8*), i64 8, i1 false)
  %5 = bitcast %struct.s6* %p6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %5, i8* align 4 bitcast ({ i32, i16, [2 x i8] }* @caller1.p6 to i8*), i64 8, i1 false)
  %6 = bitcast %struct.s7* %p7 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %6, i8* align 4 bitcast ({ i32, i16, i8, i8 }* @caller1.p7 to i8*), i64 8, i1 false)
  %call = call i32 @callee1(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, %struct.s1* byval(%struct.s1) %p1, %struct.s2* byval(%struct.s2) %p2, %struct.s3* byval(%struct.s3) %p3, %struct.s4* byval(%struct.s4) %p4, %struct.s5* byval(%struct.s5) %p5, %struct.s6* byval(%struct.s6) %p6, %struct.s7* byval(%struct.s7) %p7)
  ret i32 %call

; CHECK: stb {{[0-9]+}}, 119(1)
; CHECK: sth {{[0-9]+}}, 126(1)
; CHECK: stw {{[0-9]+}}, 132(1)
; CHECK: stw {{[0-9]+}}, 140(1)
; CHECK: std {{[0-9]+}}, 144(1)
; CHECK: std {{[0-9]+}}, 152(1)
; CHECK: std {{[0-9]+}}, 160(1)
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

define internal i32 @callee1(i32 %z1, i32 %z2, i32 %z3, i32 %z4, i32 %z5, i32 %z6, i32 %z7, i32 %z8, %struct.s1* byval(%struct.s1) %v1, %struct.s2* byval(%struct.s2) %v2, %struct.s3* byval(%struct.s3) %v3, %struct.s4* byval(%struct.s4) %v4, %struct.s5* byval(%struct.s5) %v5, %struct.s6* byval(%struct.s6) %v6, %struct.s7* byval(%struct.s7) %v7) nounwind {
entry:
  %z1.addr = alloca i32, align 4
  %z2.addr = alloca i32, align 4
  %z3.addr = alloca i32, align 4
  %z4.addr = alloca i32, align 4
  %z5.addr = alloca i32, align 4
  %z6.addr = alloca i32, align 4
  %z7.addr = alloca i32, align 4
  %z8.addr = alloca i32, align 4
  store i32 %z1, i32* %z1.addr, align 4
  store i32 %z2, i32* %z2.addr, align 4
  store i32 %z3, i32* %z3.addr, align 4
  store i32 %z4, i32* %z4.addr, align 4
  store i32 %z5, i32* %z5.addr, align 4
  store i32 %z6, i32* %z6.addr, align 4
  store i32 %z7, i32* %z7.addr, align 4
  store i32 %z8, i32* %z8.addr, align 4
  %a = getelementptr inbounds %struct.s1, %struct.s1* %v1, i32 0, i32 0
  %0 = load i8, i8* %a, align 1
  %conv = zext i8 %0 to i32
  %a1 = getelementptr inbounds %struct.s2, %struct.s2* %v2, i32 0, i32 0
  %1 = load i16, i16* %a1, align 2
  %conv2 = sext i16 %1 to i32
  %add = add nsw i32 %conv, %conv2
  %a3 = getelementptr inbounds %struct.s3, %struct.s3* %v3, i32 0, i32 0
  %2 = load i16, i16* %a3, align 2
  %conv4 = sext i16 %2 to i32
  %add5 = add nsw i32 %add, %conv4
  %a6 = getelementptr inbounds %struct.s4, %struct.s4* %v4, i32 0, i32 0
  %3 = load i32, i32* %a6, align 4
  %add7 = add nsw i32 %add5, %3
  %a8 = getelementptr inbounds %struct.s5, %struct.s5* %v5, i32 0, i32 0
  %4 = load i32, i32* %a8, align 4
  %add9 = add nsw i32 %add7, %4
  %a10 = getelementptr inbounds %struct.s6, %struct.s6* %v6, i32 0, i32 0
  %5 = load i32, i32* %a10, align 4
  %add11 = add nsw i32 %add9, %5
  %a12 = getelementptr inbounds %struct.s7, %struct.s7* %v7, i32 0, i32 0
  %6 = load i32, i32* %a12, align 4
  %add13 = add nsw i32 %add11, %6
  ret i32 %add13

; CHECK-DAG: lha {{[0-9]+}}, 126(1)
; CHECK-DAG: lha {{[0-9]+}}, 132(1)
; CHECK-DAG: lbz {{[0-9]+}}, 119(1)
; CHECK-DAG: lwz {{[0-9]+}}, 140(1)
; CHECK-DAG: lwz {{[0-9]+}}, 144(1)
; CHECK-DAG: lwz {{[0-9]+}}, 152(1)
; CHECK-DAG: lwz {{[0-9]+}}, 160(1)
}

define i32 @caller2() nounwind {
entry:
  %p1 = alloca %struct.t1, align 1
  %p2 = alloca %struct.t2, align 1
  %p3 = alloca %struct.t3, align 1
  %p4 = alloca %struct.t4, align 1
  %p5 = alloca %struct.t5, align 1
  %p6 = alloca %struct.t6, align 1
  %p7 = alloca %struct.t7, align 1
  %0 = bitcast %struct.t1* %p1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* getelementptr inbounds (%struct.t1, %struct.t1* @caller2.p1, i32 0, i32 0), i64 1, i1 false)
  %1 = bitcast %struct.t2* %p2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast ({ i16 }* @caller2.p2 to i8*), i64 2, i1 false)
  %2 = bitcast %struct.t3* %p3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* bitcast (%struct.t3* @caller2.p3 to i8*), i64 3, i1 false)
  %3 = bitcast %struct.t4* %p4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %3, i8* bitcast ({ i32 }* @caller2.p4 to i8*), i64 4, i1 false)
  %4 = bitcast %struct.t5* %p5 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %4, i8* bitcast (%struct.t5* @caller2.p5 to i8*), i64 5, i1 false)
  %5 = bitcast %struct.t6* %p6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %5, i8* bitcast (%struct.t6* @caller2.p6 to i8*), i64 6, i1 false)
  %6 = bitcast %struct.t7* %p7 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %6, i8* bitcast (%struct.t7* @caller2.p7 to i8*), i64 7, i1 false)
  %call = call i32 @callee2(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, %struct.t1* byval(%struct.t1) %p1, %struct.t2* byval(%struct.t2) %p2, %struct.t3* byval(%struct.t3) %p3, %struct.t4* byval(%struct.t4) %p4, %struct.t5* byval(%struct.t5) %p5, %struct.t6* byval(%struct.t6) %p6, %struct.t7* byval(%struct.t7) %p7)
  ret i32 %call

; CHECK: stb {{[0-9]+}}, 119(1)
; CHECK: sth {{[0-9]+}}, 126(1)
; CHECK: stb {{[0-9]+}}, 135(1)
; CHECK: sth {{[0-9]+}}, 133(1)
; CHECK: stw {{[0-9]+}}, 140(1)
; CHECK: stb {{[0-9]+}}, 151(1)
; CHECK: stw {{[0-9]+}}, 147(1)
; CHECK: sth {{[0-9]+}}, 158(1)
; CHECK: stw {{[0-9]+}}, 154(1)
; CHECK: stw {{[0-9]+}}, 164(1)
; CHECK: stw {{[0-9]+}}, 161(1)
}

define internal i32 @callee2(i32 %z1, i32 %z2, i32 %z3, i32 %z4, i32 %z5, i32 %z6, i32 %z7, i32 %z8, %struct.t1* byval(%struct.t1) %v1, %struct.t2* byval(%struct.t2) %v2, %struct.t3* byval(%struct.t3) %v3, %struct.t4* byval(%struct.t4) %v4, %struct.t5* byval(%struct.t5) %v5, %struct.t6* byval(%struct.t6) %v6, %struct.t7* byval(%struct.t7) %v7) nounwind {
entry:
  %z1.addr = alloca i32, align 4
  %z2.addr = alloca i32, align 4
  %z3.addr = alloca i32, align 4
  %z4.addr = alloca i32, align 4
  %z5.addr = alloca i32, align 4
  %z6.addr = alloca i32, align 4
  %z7.addr = alloca i32, align 4
  %z8.addr = alloca i32, align 4
  store i32 %z1, i32* %z1.addr, align 4
  store i32 %z2, i32* %z2.addr, align 4
  store i32 %z3, i32* %z3.addr, align 4
  store i32 %z4, i32* %z4.addr, align 4
  store i32 %z5, i32* %z5.addr, align 4
  store i32 %z6, i32* %z6.addr, align 4
  store i32 %z7, i32* %z7.addr, align 4
  store i32 %z8, i32* %z8.addr, align 4
  %a = getelementptr inbounds %struct.t1, %struct.t1* %v1, i32 0, i32 0
  %0 = load i8, i8* %a, align 1
  %conv = zext i8 %0 to i32
  %a1 = getelementptr inbounds %struct.t2, %struct.t2* %v2, i32 0, i32 0
  %1 = load i16, i16* %a1, align 1
  %conv2 = sext i16 %1 to i32
  %add = add nsw i32 %conv, %conv2
  %a3 = getelementptr inbounds %struct.t3, %struct.t3* %v3, i32 0, i32 0
  %2 = load i16, i16* %a3, align 1
  %conv4 = sext i16 %2 to i32
  %add5 = add nsw i32 %add, %conv4
  %a6 = getelementptr inbounds %struct.t4, %struct.t4* %v4, i32 0, i32 0
  %3 = load i32, i32* %a6, align 1
  %add7 = add nsw i32 %add5, %3
  %a8 = getelementptr inbounds %struct.t5, %struct.t5* %v5, i32 0, i32 0
  %4 = load i32, i32* %a8, align 1
  %add9 = add nsw i32 %add7, %4
  %a10 = getelementptr inbounds %struct.t6, %struct.t6* %v6, i32 0, i32 0
  %5 = load i32, i32* %a10, align 1
  %add11 = add nsw i32 %add9, %5
  %a12 = getelementptr inbounds %struct.t7, %struct.t7* %v7, i32 0, i32 0
  %6 = load i32, i32* %a12, align 1
  %add13 = add nsw i32 %add11, %6
  ret i32 %add13

; CHECK-DAG: lha {{[0-9]+}}, 126(1)
; CHECK-DAG: lha {{[0-9]+}}, 133(1)
; CHECK-DAG: lbz {{[0-9]+}}, 119(1)
; CHECK-DAG: lwz {{[0-9]+}}, 140(1)
; CHECK-DAG: lwz {{[0-9]+}}, 147(1)
; CHECK-DAG: lwz {{[0-9]+}}, 154(1)
; CHECK-DAG: lwz {{[0-9]+}}, 161(1)
}
