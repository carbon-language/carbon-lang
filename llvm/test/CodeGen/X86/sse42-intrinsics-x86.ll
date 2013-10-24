; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=-avx,+sse4.2 | FileCheck %s

define i32 @test_x86_sse42_pcmpestri128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl $7
  ; CHECK: movl $7
  ; CHECK: pcmpestri $7
  ; CHECK: movl
  %res = call i32 @llvm.x86.sse42.pcmpestri128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestri128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestri128_load(<16 x i8>* %a0, <16 x i8>* %a2) {
  ; CHECK: movl $7
  ; CHECK: movl $7
  ; CHECK: pcmpestri $7, (
  ; CHECK: movl
  %1 = load <16 x i8>* %a0
  %2 = load <16 x i8>* %a2
  %res = call i32 @llvm.x86.sse42.pcmpestri128(<16 x i8> %1, i32 7, <16 x i8> %2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}


define i32 @test_x86_sse42_pcmpestria128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: pcmpestri
  ; CHECK: seta
  %res = call i32 @llvm.x86.sse42.pcmpestria128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestria128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestric128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: pcmpestri
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.sse42.pcmpestric128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestric128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestrio128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: pcmpestri
  ; CHECK: seto
  %res = call i32 @llvm.x86.sse42.pcmpestrio128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestrio128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestris128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: pcmpestri
  ; CHECK: sets
  %res = call i32 @llvm.x86.sse42.pcmpestris128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestris128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpestriz128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: pcmpestri
  ; CHECK: sete
  %res = call i32 @llvm.x86.sse42.pcmpestriz128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpestriz128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define <16 x i8> @test_x86_sse42_pcmpestrm128(<16 x i8> %a0, <16 x i8> %a2) {
  ; CHECK: movl
  ; CHECK: movl
  ; CHECK: pcmpestrm
  ; CHECK-NOT: vmov
  %res = call <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8> %a0, i32 7, <16 x i8> %a2, i32 7, i8 7) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone


define <16 x i8> @test_x86_sse42_pcmpestrm128_load(<16 x i8> %a0, <16 x i8>* %a2) {
  ; CHECK: movl $7
  ; CHECK: movl $7
  ; CHECK: pcmpestrm $7,
  ; CHECK-NOT: vmov
  %1 = load <16 x i8>* %a2
  %res = call <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8> %a0, i32 7, <16 x i8> %1, i32 7, i8 7) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}


define i32 @test_x86_sse42_pcmpistri128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: pcmpistri $7
  ; CHECK: movl
  %res = call i32 @llvm.x86.sse42.pcmpistri128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistri128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistri128_load(<16 x i8>* %a0, <16 x i8>* %a1) {
  ; CHECK: pcmpistri $7, (
  ; CHECK: movl
  %1 = load <16 x i8>* %a0
  %2 = load <16 x i8>* %a1
  %res = call i32 @llvm.x86.sse42.pcmpistri128(<16 x i8> %1, <16 x i8> %2, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}


define i32 @test_x86_sse42_pcmpistria128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: pcmpistri
  ; CHECK: seta
  %res = call i32 @llvm.x86.sse42.pcmpistria128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistria128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistric128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: pcmpistri
  ; CHECK: sbbl
  %res = call i32 @llvm.x86.sse42.pcmpistric128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistric128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistrio128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: pcmpistri
  ; CHECK: seto
  %res = call i32 @llvm.x86.sse42.pcmpistrio128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistrio128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistris128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: pcmpistri
  ; CHECK: sets
  %res = call i32 @llvm.x86.sse42.pcmpistris128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistris128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define i32 @test_x86_sse42_pcmpistriz128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: pcmpistri
  ; CHECK: sete
  %res = call i32 @llvm.x86.sse42.pcmpistriz128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <i32> [#uses=1]
  ret i32 %res
}
declare i32 @llvm.x86.sse42.pcmpistriz128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define <16 x i8> @test_x86_sse42_pcmpistrm128(<16 x i8> %a0, <16 x i8> %a1) {
  ; CHECK: pcmpistrm $7
  ; CHECK-NOT: vmov
  %res = call <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8> %a0, <16 x i8> %a1, i8 7) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8>, <16 x i8>, i8) nounwind readnone


define <16 x i8> @test_x86_sse42_pcmpistrm128_load(<16 x i8> %a0, <16 x i8>* %a1) {
  ; CHECK: pcmpistrm $7, (
  ; CHECK-NOT: vmov
  %1 = load <16 x i8>* %a1
  %res = call <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8> %a0, <16 x i8> %1, i8 7) ; <<16 x i8>> [#uses=1]
  ret <16 x i8> %res
}
