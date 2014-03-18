; RUN: llc  < %s -march=mipsel -mcpu=mips32r2 | FileCheck %s -check-prefix=MIPS32
; RUN: llc  < %s -march=mips64el -mcpu=mips64r2 | FileCheck %s -check-prefix=MIPS64
; RUN: llc  < %s -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32r2 -mattr=+mips16 | FileCheck %s -check-prefix=mips16

define i32 @bswap32(i32 %x) nounwind readnone {
entry:
; MIPS32-LABEL: bswap32:
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16
; mips16: .ent bswap32
  %or.3 = call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %or.3
}

define i64 @bswap64(i64 %x) nounwind readnone {
entry:
; MIPS64-LABEL: bswap64:
; MIPS64: dsbh $[[R0:[0-9]+]]
; MIPS64: dshd ${{[0-9]+}}, $[[R0]]
; mips16: .ent bswap64
  %or.7 = call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %or.7
}

define <4 x i32> @bswapv4i32(<4 x i32> %x) nounwind readnone {
entry:
; MIPS32-LABEL: bswapv4i32:
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16
; MIPS32: wsbh $[[R0:[0-9]+]]
; MIPS32: rotr ${{[0-9]+}}, $[[R0]], 16
; mips16: .ent bswapv4i32
  %ret = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %x)
  ret <4 x i32> %ret
}



declare i32 @llvm.bswap.i32(i32) nounwind readnone

declare i64 @llvm.bswap.i64(i64) nounwind readnone

declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>) nounwind readnone
