; RUN: llc < %s -mtriple=arm64-apple-darwin -enable-misched=0 -mcpu=cyclone | FileCheck %s

; Trivial patchpoint codegen
;
define i64 @trivial_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: trivial_patchpoint_codegen:
; CHECK:       movz x16, #57005, lsl #32
; CHECK-NEXT:  movk x16, #48879, lsl #16
; CHECK-NEXT:  movk x16, #51966
; CHECK-NEXT:  blr  x16
; CHECK:       movz x16, #57005, lsl #32
; CHECK-NEXT:  movk x16, #48879, lsl #16
; CHECK-NEXT:  movk x16, #51967
; CHECK-NEXT:  blr  x16
; CHECK:       ret
  %resolveCall2 = inttoptr i64 244837814094590 to i8*
  %result = tail call i64 (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i64 2, i32 20, i8* %resolveCall2, i32 4, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  %resolveCall3 = inttoptr i64 244837814094591 to i8*
  tail call void (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i64 3, i32 20, i8* %resolveCall3, i32 2, i64 %p1, i64 %result)
  ret i64 %result
}

; Caller frame metadata with stackmaps. This should not be optimized
; as a leaf function.
;
; CHECK-LABEL: caller_meta_leaf
; CHECK:       mov x29, sp
; CHECK-NEXT:  sub sp, sp, #32
; CHECK:       Ltmp
; CHECK:       mov sp, x29
; CHECK:       ret

define void @caller_meta_leaf() {
entry:
  %metadata = alloca i64, i32 3, align 8
  store i64 11, i64* %metadata
  store i64 12, i64* %metadata
  store i64 13, i64* %metadata
  call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 4, i32 0, i64* %metadata)
  ret void
}

; Test the webkit_jscc calling convention.
; One argument will be passed in register, the other will be pushed on the stack.
; Return value in x0.
define void @jscall_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen:
; CHECK:      Ltmp
; CHECK:      str x{{.+}}, [sp]
; CHECK-NEXT: mov  x0, x{{.+}}
; CHECK:      Ltmp
; CHECK-NEXT: movz  x16, #65535, lsl #32
; CHECK-NEXT: movk  x16, #57005, lsl #16
; CHECK-NEXT: movk  x16, #48879
; CHECK-NEXT: blr x16
  %resolveCall2 = inttoptr i64 281474417671919 to i8*
  %result = tail call webkit_jscc i64 (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i64 5, i32 20, i8* %resolveCall2, i32 2, i64 %p4, i64 %p2)
  %resolveCall3 = inttoptr i64 244837814038255 to i8*
  tail call webkit_jscc void (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i64 6, i32 20, i8* %resolveCall3, i32 2, i64 %p4, i64 %result)
  ret void
}

; Test if the arguments are properly aligned and that we don't store undef arguments.
define i64 @jscall_patchpoint_codegen2(i64 %callee) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen2:
; CHECK:      Ltmp
; CHECK:      orr x{{.+}}, xzr, #0x6
; CHECK-NEXT: str x{{.+}}, [sp, #24]
; CHECK-NEXT: orr w{{.+}}, wzr, #0x4
; CHECK-NEXT: str w{{.+}}, [sp, #16]
; CHECK-NEXT: orr x{{.+}}, xzr, #0x2
; CHECK-NEXT: str x{{.+}}, [sp]
; CHECK:      Ltmp
; CHECK-NEXT: movz  x16, #65535, lsl #32
; CHECK-NEXT: movk  x16, #57005, lsl #16
; CHECK-NEXT: movk  x16, #48879
; CHECK-NEXT: blr x16
  %call = inttoptr i64 281474417671919 to i8*
  %result = call webkit_jscc i64 (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i64 7, i32 20, i8* %call, i32 6, i64 %callee, i64 2, i64 undef, i32 4, i32 undef, i64 6)
  ret i64 %result
}

; Test if the arguments are properly aligned and that we don't store undef arguments.
define i64 @jscall_patchpoint_codegen3(i64 %callee) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen3:
; CHECK:      Ltmp
; CHECK:      movz  x{{.+}}, #10
; CHECK-NEXT: str x{{.+}}, [sp, #48]
; CHECK-NEXT: orr w{{.+}}, wzr, #0x8
; CHECK-NEXT: str w{{.+}}, [sp, #36]
; CHECK-NEXT: orr x{{.+}}, xzr, #0x6
; CHECK-NEXT: str x{{.+}}, [sp, #24]
; CHECK-NEXT: orr w{{.+}}, wzr, #0x4
; CHECK-NEXT: str w{{.+}}, [sp, #16]
; CHECK-NEXT: orr x{{.+}}, xzr, #0x2
; CHECK-NEXT: str x{{.+}}, [sp]
; CHECK:      Ltmp
; CHECK-NEXT: movz  x16, #65535, lsl #32
; CHECK-NEXT: movk  x16, #57005, lsl #16
; CHECK-NEXT: movk  x16, #48879
; CHECK-NEXT: blr x16
  %call = inttoptr i64 281474417671919 to i8*
  %result = call webkit_jscc i64 (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i64 7, i32 20, i8* %call, i32 10, i64 %callee, i64 2, i64 undef, i32 4, i32 undef, i64 6, i32 undef, i32 8, i32 undef, i64 10)
  ret i64 %result
}

; Test patchpoints reusing the same TargetConstant.
; <rdar:15390785> Assertion failed: (CI.getNumArgOperands() >= NumArgs + 4)
; There is no way to verify this, since it depends on memory allocation.
; But I think it's useful to include as a working example.
define i64 @testLowerConstant(i64 %arg, i64 %tmp2, i64 %tmp10, i64* %tmp33, i64 %tmp79) {
entry:
  %tmp80 = add i64 %tmp79, -16
  %tmp81 = inttoptr i64 %tmp80 to i64*
  %tmp82 = load i64* %tmp81, align 8
  tail call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 14, i32 8, i64 %arg, i64 %tmp2, i64 %tmp10, i64 %tmp82)
  tail call void (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i64 15, i32 32, i8* null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp82)
  %tmp83 = load i64* %tmp33, align 8
  %tmp84 = add i64 %tmp83, -24
  %tmp85 = inttoptr i64 %tmp84 to i64*
  %tmp86 = load i64* %tmp85, align 8
  tail call void (i64, i32, ...)* @llvm.experimental.stackmap(i64 17, i32 8, i64 %arg, i64 %tmp10, i64 %tmp86)
  tail call void (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i64 18, i32 32, i8* null, i32 3, i64 %arg, i64 %tmp10, i64 %tmp86)
  ret i64 10
}

; Test small patchpoints that don't emit calls.
define void @small_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: small_patchpoint_codegen:
; CHECK:      Ltmp
; CHECK:      nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NEXT: ldp
; CHECK-NEXT: ret
  %result = tail call i64 (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i64 5, i32 20, i8* null, i32 2, i64 %p1, i64 %p2)
  ret void
}

; Test that scratch registers are spilled around patchpoints
; CHECK: InlineAsm End
; CHECK-NEXT: mov x{{[0-9]+}}, x16
; CHECK-NEXT: mov x{{[0-9]+}}, x17
; CHECK-NEXT: Ltmp
; CHECK-NEXT: nop
define void @clobberScratch(i32* %p) {
  %v = load i32* %p
  tail call void asm sideeffect "nop", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{x29},~{x30},~{x31}"() nounwind
  tail call void (i64, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i64 5, i32 20, i8* null, i32 0, i32* %p, i32 %v)
  store i32 %v, i32* %p
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
