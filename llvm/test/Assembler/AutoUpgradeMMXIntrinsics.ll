; Tests to make sure MMX intrinsics are automatically upgraded.
; RUN: llvm-as < %s | llvm-dis -o %t
; RUN: grep {llvm\\.x86\\.mmx} %t | not grep {\\\<1 x i64\\\>}
; RUN: grep {llvm\\.x86\\.mmx} %t | not grep {\\\<2 x i32\\\>}
; RUN: grep {llvm\\.x86\\.mmx} %t | not grep {\\\<4 x i16\\\>}
; RUN: grep {llvm\\.x86\\.mmx} %t | not grep {\\\<8 x i8\\\>}
; RUN: grep {llvm\\.x86\\.sse\\.pshuf\\.w} %t | not grep i32

; Addition
declare <8 x i8>  @llvm.x86.mmx.padd.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.padd.w(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.x86.mmx.padd.d(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.x86.mmx.padd.q(<1 x i64>, <1 x i64>) nounwind readnone
declare <8 x i8>  @llvm.x86.mmx.padds.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.padds.w(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i8>  @llvm.x86.mmx.paddus.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.paddus.w(<4 x i16>, <4 x i16>) nounwind readnone
define void @add(<8 x i8> %A,  <8 x i8> %B,  <4 x i16> %C, <4 x i16> %D,
                 <2 x i32> %E, <2 x i32> %F, <1 x i64> %G, <1 x i64> %H) {
  %r1 = call <8 x i8>  @llvm.x86.mmx.padd.b(<8 x i8> %A,  <8 x i8> %B)
  %r2 = call <4 x i16> @llvm.x86.mmx.padd.w(<4 x i16> %C, <4 x i16> %D)
  %r3 = call <2 x i32> @llvm.x86.mmx.padd.d(<2 x i32> %E, <2 x i32> %F)
  %r4 = call <1 x i64> @llvm.x86.mmx.padd.q(<1 x i64> %G, <1 x i64> %H)
  %r5 = call <8 x i8>  @llvm.x86.mmx.padds.b(<8 x i8>  %A, <8 x i8>  %B)
  %r6 = call <4 x i16> @llvm.x86.mmx.padds.w(<4 x i16> %C, <4 x i16> %D)
  %r7 = call <8 x i8>  @llvm.x86.mmx.paddus.b(<8 x i8>  %A, <8 x i8>  %B)
  %r8 = call <4 x i16> @llvm.x86.mmx.paddus.w(<4 x i16> %C, <4 x i16> %D)
  ret void
}

; Subtraction
declare <8 x i8>  @llvm.x86.mmx.psub.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.psub.w(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.x86.mmx.psub.d(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.x86.mmx.psub.q(<1 x i64>, <1 x i64>) nounwind readnone
declare <8 x i8>  @llvm.x86.mmx.psubs.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.psubs.w(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i8>  @llvm.x86.mmx.psubus.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.psubus.w(<4 x i16>, <4 x i16>) nounwind readnone
define void @sub(<8 x i8> %A,  <8 x i8> %B,  <4 x i16> %C, <4 x i16> %D,
                 <2 x i32> %E, <2 x i32> %F, <1 x i64> %G, <1 x i64> %H) {
  %r1 = call <8 x i8>  @llvm.x86.mmx.psub.b(<8 x i8> %A,  <8 x i8> %B)
  %r2 = call <4 x i16> @llvm.x86.mmx.psub.w(<4 x i16> %C, <4 x i16> %D)
  %r3 = call <2 x i32> @llvm.x86.mmx.psub.d(<2 x i32> %E, <2 x i32> %F)
  %r4 = call <1 x i64> @llvm.x86.mmx.psub.q(<1 x i64> %G, <1 x i64> %H)
  %r5 = call <8 x i8>  @llvm.x86.mmx.psubs.b(<8 x i8>  %A, <8 x i8>  %B)
  %r6 = call <4 x i16> @llvm.x86.mmx.psubs.w(<4 x i16> %C, <4 x i16> %D)
  %r7 = call <8 x i8>  @llvm.x86.mmx.psubus.b(<8 x i8>  %A, <8 x i8>  %B)
  %r8 = call <4 x i16> @llvm.x86.mmx.psubus.w(<4 x i16> %C, <4 x i16> %D)
  ret void
}

; Multiplication
declare <4 x i16> @llvm.x86.mmx.pmulh.w(<4 x i16>, <4 x i16>) nounwind readnone
declare <4 x i16> @llvm.x86.mmx.pmull.w(<4 x i16>, <4 x i16>) nounwind readnone
declare <4 x i16> @llvm.x86.mmx.pmulhu.w(<4 x i16>, <4 x i16>) nounwind readnone
declare <4 x i16> @llvm.x86.mmx.pmulu.dq(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.x86.mmx.pmadd.wd(<4 x i16>, <4 x i16>) nounwind readnone
define void @mul(<4 x i16> %A, <4 x i16> %B) {
  %r1 = call <4 x i16> @llvm.x86.mmx.pmulh.w(<4 x i16> %A, <4 x i16> %B)
  %r2 = call <4 x i16> @llvm.x86.mmx.pmull.w(<4 x i16> %A, <4 x i16> %B)
  %r3 = call <4 x i16> @llvm.x86.mmx.pmulhu.w(<4 x i16> %A, <4 x i16> %B)
  %r4 = call <4 x i16> @llvm.x86.mmx.pmulu.dq(<4 x i16> %A, <4 x i16> %B)
  %r5 = call <2 x i32> @llvm.x86.mmx.pmadd.wd(<4 x i16> %A, <4 x i16> %B)
  ret void
}

; Bitwise operations
declare <1 x i64> @llvm.x86.mmx.pand(<1 x i64>, <1 x i64>)  nounwind readnone
declare <1 x i64> @llvm.x86.mmx.pandn(<1 x i64>, <1 x i64>) nounwind readnone
declare <1 x i64> @llvm.x86.mmx.por(<1 x i64>, <1 x i64>)   nounwind readnone
declare <1 x i64> @llvm.x86.mmx.pxor(<1 x i64>, <1 x i64>)  nounwind readnone
define void @bit(<1 x i64> %A, <1 x i64> %B) {
  %r1 = call <1 x i64> @llvm.x86.mmx.pand(<1 x i64> %A, <1 x i64> %B)
  %r2 = call <1 x i64> @llvm.x86.mmx.pandn(<1 x i64> %A, <1 x i64> %B)
  %r3 = call <1 x i64> @llvm.x86.mmx.por(<1 x i64> %A, <1 x i64> %B)
  %r4 = call <1 x i64> @llvm.x86.mmx.pxor(<1 x i64> %A, <1 x i64> %B)
  ret void
}

; Averages
declare <8 x i8>  @llvm.x86.mmx.pavg.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.pavg.w(<4 x i16>, <4 x i16>) nounwind readnone
define void @avg(<8 x i8> %A, <8 x i8> %B, <4 x i16> %C, <4 x i16> %D) {
  %r1 = call <8 x i8>  @llvm.x86.mmx.pavg.b(<8 x i8>  %A, <8 x i8>  %B)
  %r2 = call <4 x i16> @llvm.x86.mmx.pavg.w(<4 x i16> %C, <4 x i16> %D)
  ret void
}

; Maximum
declare <8 x i8>  @llvm.x86.mmx.pmaxu.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.pmaxs.w(<4 x i16>, <4 x i16>) nounwind readnone
define void @max(<8 x i8> %A, <8 x i8> %B, <4 x i16> %C, <4 x i16> %D) {
  %r1 = call <8 x i8>  @llvm.x86.mmx.pmaxu.b(<8 x i8>  %A, <8 x i8>  %B)
  %r2 = call <4 x i16> @llvm.x86.mmx.pmaxs.w(<4 x i16> %C, <4 x i16> %D)
  ret void
}

; Minimum
declare <8 x i8>  @llvm.x86.mmx.pminu.b(<8 x i8>,  <8 x i8>)  nounwind readnone
declare <4 x i16> @llvm.x86.mmx.pmins.w(<4 x i16>, <4 x i16>) nounwind readnone
define void @min(<8 x i8> %A, <8 x i8> %B, <4 x i16> %C, <4 x i16> %D) {
  %r1 = call <8 x i8>  @llvm.x86.mmx.pminu.b(<8 x i8>  %A, <8 x i8>  %B)
  %r2 = call <4 x i16> @llvm.x86.mmx.pmins.w(<4 x i16> %C, <4 x i16> %D)
  ret void
}

; Packed sum of absolute differences
declare <4 x i16> @llvm.x86.mmx.psad.bw(<8 x i8>, <8 x i8>) nounwind readnone
define void @psad(<8 x i8> %A, <8 x i8> %B) {
  %r1 = call <4 x i16> @llvm.x86.mmx.psad.bw(<8 x i8> %A, <8 x i8> %B)
  ret void
}

; Shift left
declare <4 x i16> @llvm.x86.mmx.psll.w(<4 x i16>, <1 x i64>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.psll.d(<2 x i32>, <1 x i64>) nounwind readnone 
declare <1 x i64> @llvm.x86.mmx.psll.q(<1 x i64>, <1 x i64>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.pslli.w(<4 x i16>, i32) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.pslli.d(<2 x i32>, i32) nounwind readnone 
declare <1 x i64> @llvm.x86.mmx.pslli.q(<1 x i64>, i32) nounwind readnone 
define void @shl(<4 x i16> %A, <2 x i32> %B, <1 x i64> %C, i32 %D) {
  %r1 = call <4 x i16> @llvm.x86.mmx.psll.w(<4 x i16> %A, <1 x i64> %C)
  %r2 = call <2 x i32> @llvm.x86.mmx.psll.d(<2 x i32> %B, <1 x i64> %C)
  %r3 = call <1 x i64> @llvm.x86.mmx.psll.q(<1 x i64> %C, <1 x i64> %C)
  %r4 = call <4 x i16> @llvm.x86.mmx.pslli.w(<4 x i16> %A, i32 %D)
  %r5 = call <2 x i32> @llvm.x86.mmx.pslli.d(<2 x i32> %B, i32 %D)
  %r6 = call <1 x i64> @llvm.x86.mmx.pslli.q(<1 x i64> %C, i32 %D)
  ret void
}

; Shift right logical
declare <4 x i16> @llvm.x86.mmx.psrl.w(<4 x i16>, <1 x i64>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.psrl.d(<2 x i32>, <1 x i64>) nounwind readnone 
declare <1 x i64> @llvm.x86.mmx.psrl.q(<1 x i64>, <1 x i64>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.psrli.w(<4 x i16>, i32) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.psrli.d(<2 x i32>, i32) nounwind readnone 
declare <1 x i64> @llvm.x86.mmx.psrli.q(<1 x i64>, i32) nounwind readnone 
define void @shr(<4 x i16> %A, <2 x i32> %B, <1 x i64> %C, i32 %D) {
  %r1 = call <4 x i16> @llvm.x86.mmx.psrl.w(<4 x i16> %A, <1 x i64> %C)
  %r2 = call <2 x i32> @llvm.x86.mmx.psrl.d(<2 x i32> %B, <1 x i64> %C)
  %r3 = call <1 x i64> @llvm.x86.mmx.psrl.q(<1 x i64> %C, <1 x i64> %C)
  %r4 = call <4 x i16> @llvm.x86.mmx.psrli.w(<4 x i16> %A, i32 %D)
  %r5 = call <2 x i32> @llvm.x86.mmx.psrli.d(<2 x i32> %B, i32 %D)
  %r6 = call <1 x i64> @llvm.x86.mmx.psrli.q(<1 x i64> %C, i32 %D)
  ret void
}

; Shift right arithmetic
declare <4 x i16> @llvm.x86.mmx.psra.w(<4 x i16>, <1 x i64>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.psra.d(<2 x i32>, <1 x i64>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.psrai.w(<4 x i16>, i32) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.psrai.d(<2 x i32>, i32) nounwind readnone 
define void @sra(<4 x i16> %A, <2 x i32> %B, <1 x i64> %C, i32 %D) {
  %r1 = call <4 x i16> @llvm.x86.mmx.psra.w(<4 x i16> %A, <1 x i64> %C)
  %r2 = call <2 x i32> @llvm.x86.mmx.psra.d(<2 x i32> %B, <1 x i64> %C)
  %r3 = call <4 x i16> @llvm.x86.mmx.psrai.w(<4 x i16> %A, i32 %D)
  %r4 = call <2 x i32> @llvm.x86.mmx.psrai.d(<2 x i32> %B, i32 %D)
  ret void
}

; Pack/Unpack ops
declare <8 x i8>  @llvm.x86.mmx.packsswb(<4 x i16>, <4 x i16>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.packssdw(<2 x i32>, <2 x i32>) nounwind readnone 
declare <8 x i8>  @llvm.x86.mmx.packuswb(<4 x i16>, <4 x i16>) nounwind readnone 
declare <8 x i8>  @llvm.x86.mmx.punpckhbw(<8 x i8>, <8 x i8>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.punpckhwd(<4 x i16>, <4 x i16>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.punpckhdq(<2 x i32>, <2 x i32>) nounwind readnone 
declare <8 x i8>  @llvm.x86.mmx.punpcklbw(<8 x i8>, <8 x i8>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.punpcklwd(<4 x i16>, <4 x i16>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.punpckldq(<2 x i32>, <2 x i32>) nounwind readnone 
define void @pack_unpack(<8 x i8> %A, <8 x i8> %B, <4 x i16> %C, <4 x i16> %D,
                         <2 x i32> %E, <2 x i32> %F) {
  %r1 = call <8 x i8>  @llvm.x86.mmx.packsswb(<4 x i16> %C, <4 x i16> %D)
  %r2 = call <4 x i16> @llvm.x86.mmx.packssdw(<2 x i32> %E, <2 x i32> %F)
  %r3 = call <8 x i8>  @llvm.x86.mmx.packuswb(<4 x i16> %C, <4 x i16> %D)
  %r4 = call <8 x i8>  @llvm.x86.mmx.punpckhbw(<8 x i8>  %A, <8 x i8>  %B)
  %r5 = call <4 x i16> @llvm.x86.mmx.punpckhwd(<4 x i16> %C, <4 x i16> %D)
  %r6 = call <2 x i32> @llvm.x86.mmx.punpckhdq(<2 x i32> %E, <2 x i32> %F)
  %r7 = call <8 x i8>  @llvm.x86.mmx.punpcklbw(<8 x i8>  %A, <8 x i8>  %B)
  %r8 = call <4 x i16> @llvm.x86.mmx.punpcklwd(<4 x i16> %C, <4 x i16> %D)
  %r9 = call <2 x i32> @llvm.x86.mmx.punpckldq(<2 x i32> %E, <2 x i32> %F)
  ret void
}

; Integer comparison ops
declare <8 x i8>  @llvm.x86.mmx.pcmpeq.b(<8 x i8>, <8 x i8>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.pcmpeq.w(<4 x i16>, <4 x i16>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.pcmpeq.d(<2 x i32>, <2 x i32>) nounwind readnone 
declare <8 x i8>  @llvm.x86.mmx.pcmpgt.b(<8 x i8>, <8 x i8>) nounwind readnone 
declare <4 x i16> @llvm.x86.mmx.pcmpgt.w(<4 x i16>, <4 x i16>) nounwind readnone 
declare <2 x i32> @llvm.x86.mmx.pcmpgt.d(<2 x i32>, <2 x i32>) nounwind readnone 
define void @cmp(<8 x i8> %A, <8 x i8> %B, <4 x i16> %C, <4 x i16> %D,
                 <2 x i32> %E, <2 x i32> %F) {
  %r1 = call <8 x i8>  @llvm.x86.mmx.pcmpeq.b(<8 x i8>  %A, <8 x i8>  %B)
  %r2 = call <4 x i16> @llvm.x86.mmx.pcmpeq.w(<4 x i16> %C, <4 x i16> %D)
  %r3 = call <2 x i32> @llvm.x86.mmx.pcmpeq.d(<2 x i32> %E, <2 x i32> %F)
  %r4 = call <8 x i8>  @llvm.x86.mmx.pcmpgt.b(<8 x i8>  %A, <8 x i8>  %B)
  %r5 = call <4 x i16> @llvm.x86.mmx.pcmpgt.w(<4 x i16> %C, <4 x i16> %D)
  %r6 = call <2 x i32> @llvm.x86.mmx.pcmpgt.d(<2 x i32> %E, <2 x i32> %F)
  ret void
}

; Miscellaneous
declare void      @llvm.x86.mmx.maskmovq(<8 x i8>, <8 x i8>, i32*) nounwind readnone 
declare i32       @llvm.x86.mmx.pmovmskb(<8 x i8>) nounwind readnone 
declare void      @llvm.x86.mmx.movnt.dq(i32*, <1 x i64>) nounwind readnone 
declare <1 x i64> @llvm.x86.mmx.palignr.b(<1 x i64>, <1 x i64>,  i8) nounwind readnone 
declare i32       @llvm.x86.mmx.pextr.w(<1 x i64>, i32) nounwind readnone 
declare <1 x i64> @llvm.x86.mmx.pinsr.w(<1 x i64>, i32, i32) nounwind readnone 
declare <4 x i16> @llvm.x86.ssse3.pshuf.w(<4 x i16>, i32) nounwind readnone 
define void @misc(<8 x i8> %A, <8 x i8> %B, <4 x i16> %C, <4 x i16> %D,
                  <2 x i32> %E, <2 x i32> %F, <1 x i64> %G, <1 x i64> %H,
                  i32* %I, i8 %J, i16 %K, i32 %L) {
        call void      @llvm.x86.mmx.maskmovq(<8 x i8> %A, <8 x i8> %B, i32* %I)
  %r1 = call i32       @llvm.x86.mmx.pmovmskb(<8 x i8> %A)
        call void      @llvm.x86.mmx.movnt.dq(i32* %I, <1 x i64> %G)
  %r2 = call <1 x i64> @llvm.x86.mmx.palignr.b(<1 x i64> %G, <1 x i64> %H, i8 %J)
  %r3 = call i32       @llvm.x86.mmx.pextr.w(<1 x i64> %G, i32 37)
  %r4 = call <1 x i64> @llvm.x86.mmx.pinsr.w(<1 x i64> %G, i32 37, i32 927)
  %r5 = call <4 x i16> @llvm.x86.ssse3.pshuf.w(<4 x i16> %C, i32 37)
  ret void
}
