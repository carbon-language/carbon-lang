; RUN: llvm-as < %s | llc -march=x86-64 | %prcontext {setne	%al} 1 | grep test | count 2
; PR4814

; CodeGen shouldn't try to do a setne after an expanded 8-bit conditional
; move without recomputing EFLAGS, because the expansion of the conditional
; move with control flow may clobber EFLAGS (e.g., with xor, to set the
; register to zero).

; The prcontext usage above is a little awkward; the important part is that
; there's a test before the setne.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

@g_3 = external global i8                         ; <i8*> [#uses=1]
@g_96 = external global i8                        ; <i8*> [#uses=2]
@g_100 = external global i8                       ; <i8*> [#uses=2]
@_2E_str = external constant [15 x i8], align 1   ; <[15 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
  %0 = load i8* @g_3, align 1                     ; <i8> [#uses=2]
  %1 = sext i8 %0 to i32                          ; <i32> [#uses=1]
  %.lobit.i = lshr i8 %0, 7                       ; <i8> [#uses=1]
  %tmp.i = zext i8 %.lobit.i to i32               ; <i32> [#uses=1]
  %tmp.not.i = xor i32 %tmp.i, 1                  ; <i32> [#uses=1]
  %iftmp.17.0.i.i = ashr i32 %1, %tmp.not.i       ; <i32> [#uses=1]
  %retval56.i.i = trunc i32 %iftmp.17.0.i.i to i8 ; <i8> [#uses=1]
  %2 = icmp eq i8 %retval56.i.i, 0                ; <i1> [#uses=2]
  %g_96.promoted.i = load i8* @g_96               ; <i8> [#uses=3]
  %3 = icmp eq i8 %g_96.promoted.i, 0             ; <i1> [#uses=2]
  br i1 %3, label %func_4.exit.i, label %bb.i.i.i

bb.i.i.i:                                         ; preds = %entry
  %4 = volatile load i8* @g_100, align 1          ; <i8> [#uses=0]
  br label %func_4.exit.i

func_4.exit.i:                                    ; preds = %bb.i.i.i, %entry
  %.not.i = xor i1 %2, true                       ; <i1> [#uses=1]
  %brmerge.i = or i1 %3, %.not.i                  ; <i1> [#uses=1]
  %.mux.i = select i1 %2, i8 %g_96.promoted.i, i8 0 ; <i8> [#uses=1]
  br i1 %brmerge.i, label %func_1.exit, label %bb.i.i

bb.i.i:                                           ; preds = %func_4.exit.i
  %5 = volatile load i8* @g_100, align 1          ; <i8> [#uses=0]
  br label %func_1.exit

func_1.exit:                                      ; preds = %bb.i.i, %func_4.exit.i
  %g_96.tmp.0.i = phi i8 [ %g_96.promoted.i, %bb.i.i ], [ %.mux.i, %func_4.exit.i ] ; <i8> [#uses=2]
  store i8 %g_96.tmp.0.i, i8* @g_96
  %6 = zext i8 %g_96.tmp.0.i to i32               ; <i32> [#uses=1]
  %7 = tail call i32 (i8*, ...)* @printf(i8* noalias getelementptr ([15 x i8]* @_2E_str, i64 0, i64 0), i32 %6) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind
