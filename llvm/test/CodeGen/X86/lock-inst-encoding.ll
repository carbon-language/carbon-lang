; RUN: llc -O0 --show-mc-encoding < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK: f1:
; CHECK: addq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x01,0x37]
; CHECK: ret
define void @f1(i64* %a, i64 %b) nounwind {
  call i64 @llvm.atomic.load.add.i64.p0i64(i64* %a, i64 %b) nounwind
  ret void
}

declare i64 @llvm.atomic.load.add.i64.p0i64(i64* nocapture, i64) nounwind

; CHECK: f2:
; CHECK: subq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x29,0x37]
; CHECK: ret
define void @f2(i64* %a, i64 %b) nounwind {
  call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %a, i64 %b) nounwind
  ret void
}

declare i64 @llvm.atomic.load.sub.i64.p0i64(i64* nocapture, i64) nounwind

; CHECK: f3:
; CHECK: andq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x21,0x37]
; CHECK: ret
define void @f3(i64* %a, i64 %b) nounwind {
  call i64 @llvm.atomic.load.and.i64.p0i64(i64* %a, i64 %b) nounwind
  ret void
}

declare i64 @llvm.atomic.load.and.i64.p0i64(i64* nocapture, i64) nounwind

; CHECK: f4:
; CHECK: orq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x09,0x37]
; CHECK: ret
define void @f4(i64* %a, i64 %b) nounwind {
  call i64 @llvm.atomic.load.or.i64.p0i64(i64* %a, i64 %b) nounwind
  ret void
}

declare i64 @llvm.atomic.load.or.i64.p0i64(i64* nocapture, i64) nounwind

; CHECK: f5:
; CHECK: xorq %{{.*}}, ({{.*}}){{.*}}encoding: [0xf0,0x48,0x31,0x37]
; CHECK: ret
define void @f5(i64* %a, i64 %b) nounwind {
  call i64 @llvm.atomic.load.xor.i64.p0i64(i64* %a, i64 %b) nounwind
  ret void
}

declare i64 @llvm.atomic.load.xor.i64.p0i64(i64* nocapture, i64) nounwind
