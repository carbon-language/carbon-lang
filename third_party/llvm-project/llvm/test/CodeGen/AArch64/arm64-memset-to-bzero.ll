; RUN: llc %s -enable-machine-outliner=never -mtriple=arm64-apple-darwin -o - \
; RUN: | FileCheck --check-prefixes=CHECK,CHECK-DARWIN %s
; RUN: llc %s -enable-machine-outliner=never -mtriple=arm64-linux-gnu -o - | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK-LINUX %s
; <rdar://problem/14199482> ARM64: Calls to bzero() replaced with calls to memset()

; CHECK-LABEL: fct1:
; For small size (<= 256), we do not change memset to bzero.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct1(i8* nocapture %ptr) minsize {
entry:
  tail call void @llvm.memset.p0i8.i64(i8* %ptr, i8 0, i64 256, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

; CHECK-LABEL: fct2:
; When the size is bigger than 256, change into bzero.
; CHECK-DARWIN: {{b|bl}} _bzero
; CHECK-LINUX: {{b|bl}} memset
define void @fct2(i8* nocapture %ptr) minsize {
entry:
  tail call void @llvm.memset.p0i8.i64(i8* %ptr, i8 0, i64 257, i1 false)
  ret void
}

; CHECK-LABEL: fct3:
; For unknown size, change to bzero.
; CHECK-DARWIN: {{b|bl}} _bzero
; CHECK-LINUX: {{b|bl}} memset
define void @fct3(i8* nocapture %ptr, i32 %unknown) minsize {
entry:
  %conv = sext i32 %unknown to i64
  tail call void @llvm.memset.p0i8.i64(i8* %ptr, i8 0, i64 %conv, i1 false)
  ret void
}

; CHECK-LABEL: fct4:
; Size <= 256, no change.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct4(i8* %ptr) minsize {
entry:
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 0, i64 256, i64 %tmp)
  ret void
}

declare i8* @__memset_chk(i8*, i32, i64, i64)

declare i64 @llvm.objectsize.i64(i8*, i1)

; CHECK-LABEL: fct5:
; Size > 256, change.
; CHECK-DARWIN: {{b|bl}} _bzero
; CHECK-LINUX: {{b|bl}} memset
define void @fct5(i8* %ptr) minsize {
entry:
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 0, i64 257, i64 %tmp)
  ret void
}

; CHECK-LABEL: fct6:
; Size = unknown, change.
; CHECK-DARWIN: {{b|bl}} _bzero
; CHECK-LINUX: {{b|bl}} memset
define void @fct6(i8* %ptr, i32 %unknown) minsize {
entry:
  %conv = sext i32 %unknown to i64
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 0, i64 %conv, i64 %tmp)
  ret void
}

; Next functions check that memset is not turned into bzero
; when the set constant is non-zero, whatever the given size.

; CHECK-LABEL: fct7:
; memset with something that is not a zero, no change.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct7(i8* %ptr) minsize {
entry:
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 1, i64 256, i64 %tmp)
  ret void
}

; CHECK-LABEL: fct8:
; memset with something that is not a zero, no change.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct8(i8* %ptr) minsize {
entry:
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 1, i64 257, i64 %tmp)
  ret void
}

; CHECK-LABEL: fct9:
; memset with something that is not a zero, no change.
; CHECK-DARWIN: {{b|bl}} _memset
; CHECK-LINUX: {{b|bl}} memset
define void @fct9(i8* %ptr, i32 %unknown) minsize {
entry:
  %conv = sext i32 %unknown to i64
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 1, i64 %conv, i64 %tmp)
  ret void
}
