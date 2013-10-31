; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
;
; Note: Print verbose stackmaps using -debug-only=stackmaps.

; CHECK-LABEL:  .section  __LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; CHECK-NEXT:   .long   0
; Num LargeConstants
; CHECK-NEXT:   .long   1
; CHECK-NEXT:   .quad   4294967296
; Num Callsites
; CHECK-NEXT:   .long   8

; Constant arguments
;
; CHECK-NEXT:   .long   1
; CHECK-NEXT:   .long   L{{.*}}-_constantargs
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  4
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   65535
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   65536
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   4294967295
; LargeConstant at index 0
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   0

define void @constantargs() {
entry:
  %0 = inttoptr i64 12345 to i8*
  tail call void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 1, i32 2, i8* %0, i32 0, i64 65535, i64 65536, i64 4294967295, i64 4294967296)
  ret void
}

; Inline OSR Exit
;
; CHECK-NEXT:   .long   3
; CHECK-NEXT:   .long   L{{.*}}-_osrinline
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long  0
define void @osrinline(i64 %a, i64 %b) {
entry:
  ; Runtime void->void call.
  call void inttoptr (i64 -559038737 to void ()*)()
  ; Followed by inline OSR patchpoint with 12-byte shadow and 2 live vars.
  call void (i32, i32, ...)* @llvm.experimental.stackmap(i32 3, i32 12, i64 %a, i64 %b)
  ret void
}

; Cold OSR Exit
;
; 2 live variables in register.
;
; CHECK-NEXT:   .long  4
; CHECK-NEXT:   .long   L{{.*}}-_osrcold
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long  0
define void @osrcold(i64 %a, i64 %b) {
entry:
  %test = icmp slt i64 %a, %b
  br i1 %test, label %ret, label %cold
cold:
  ; OSR patchpoint with 12-byte nop-slide and 2 live vars.
  %thunk = inttoptr i64 -559038737 to i8*
  call void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 4, i32 12, i8* %thunk, i32 0, i64 %a, i64 %b)
  unreachable
ret:
  ret void
}

; Property Read
; CHECK-NEXT:  .long  5
; CHECK-NEXT:   .long   L{{.*}}-_propertyRead
; CHECK-NEXT:  .short  0
; CHECK-NEXT:  .short  0
;
; FIXME: There are currently no stackmap entries. After moving to
; AnyRegCC, we will have entries for the object and return value.
define i64 @propertyRead(i64* %obj) {
entry:
  %resolveRead = inttoptr i64 -559038737 to i8*
  %result = call i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 5, i32 12, i8* %resolveRead, i32 1, i64* %obj)
  %add = add i64 %result, 3
  ret i64 %add
}

; Property Write
; CHECK-NEXT:  .long  6
; CHECK-NEXT:   .long   L{{.*}}-_propertyWrite
; CHECK-NEXT:  .short  0
; CHECK-NEXT:  .short  0
;
; FIXME: There are currently no stackmap entries. After moving to
; AnyRegCC, we will have entries for the object and return value.
define void @propertyWrite(i64 %dummy1, i64* %obj, i64 %dummy2, i64 %a) {
entry:
  %resolveWrite = inttoptr i64 -559038737 to i8*
  call void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 6, i32 12, i8* %resolveWrite, i32 2, i64* %obj, i64 %a)
  ret void
}

; Void JS Call
;
; 2 live variables in registers.
;
; CHECK-NEXT:   .long  7
; CHECK-NEXT:   .long   L{{.*}}-_jsVoidCall
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
define void @jsVoidCall(i64 %dummy1, i64* %obj, i64 %arg, i64 %l1, i64 %l2) {
entry:
  %resolveCall = inttoptr i64 -559038737 to i8*
  call void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 7, i32 12, i8* %resolveCall, i32 2, i64* %obj, i64 %arg, i64 %l1, i64 %l2)
  ret void
}

; i64 JS Call
;
; 2 live variables in registers.
;
; CHECK:        .long  8
; CHECK-NEXT:   .long   L{{.*}}-_jsIntCall
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
define i64 @jsIntCall(i64 %dummy1, i64* %obj, i64 %arg, i64 %l1, i64 %l2) {
entry:
  %resolveCall = inttoptr i64 -559038737 to i8*
  %result = call i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 8, i32 12, i8* %resolveCall, i32 2, i64* %obj, i64 %arg, i64 %l1, i64 %l2)
  %add = add i64 %result, 3
  ret i64 %add
}

; Spilled stack map values.
;
; Verify 17 stack map entries.
;
; CHECK:      .long 11
; CHECK-NEXT: .long L{{.*}}-_spilledValue
; CHECK-NEXT: .short 0
; CHECK-NEXT: .short 10
;
; Check that at least one is a spilled entry (Indirect).
; CHECK: .byte 3
; CHECK: .byte 0
define void @spilledValue(i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16) {
entry:
  %resolveCall = inttoptr i64 -559038737 to i8*
  call void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 11, i32 12, i8* %resolveCall, i32 5, i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9)

; FIXME: The Spiller needs to be able to fold all rematted loads! This
; can be seen by adding %l15 to the stackmap.
; <rdar:/15202984> [JS] Ran out of registers during register allocation
;  %resolveCall = inttoptr i64 -559038737 to i8*
;  call void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 12, i32 12, i8* %resolveCall, i32 5, i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16)
  ret void
}

declare void @llvm.experimental.stackmap(i32, i32, ...)
declare void @llvm.experimental.patchpoint.void(i32, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i32, i32, i8*, i32, ...)
