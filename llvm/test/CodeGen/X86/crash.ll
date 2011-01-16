; RUN: llc -march=x86 %s -o -
; RUN: llc -march=x86-64 %s -o -

; PR6497

; Chain and flag folding issues.
define i32 @test1() nounwind ssp {
entry:
  %tmp5.i = volatile load i32* undef              ; <i32> [#uses=1]
  %conv.i = zext i32 %tmp5.i to i64               ; <i64> [#uses=1]
  %tmp12.i = volatile load i32* undef             ; <i32> [#uses=1]
  %conv13.i = zext i32 %tmp12.i to i64            ; <i64> [#uses=1]
  %shl.i = shl i64 %conv13.i, 32                  ; <i64> [#uses=1]
  %or.i = or i64 %shl.i, %conv.i                  ; <i64> [#uses=1]
  %add16.i = add i64 %or.i, 256                   ; <i64> [#uses=1]
  %shr.i = lshr i64 %add16.i, 8                   ; <i64> [#uses=1]
  %conv19.i = trunc i64 %shr.i to i32             ; <i32> [#uses=1]
  volatile store i32 %conv19.i, i32* undef
  ret i32 undef
}

; PR6533
define void @test2(i1 %x, i32 %y) nounwind {
  %land.ext = zext i1 %x to i32                   ; <i32> [#uses=1]
  %and = and i32 %y, 1                        ; <i32> [#uses=1]
  %xor = xor i32 %and, %land.ext                  ; <i32> [#uses=1]
  %cmp = icmp eq i32 %xor, 1                      ; <i1> [#uses=1]
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %land.end
  ret void

if.end:                                           ; preds = %land.end
  ret void
}

; PR6577
%pair = type { i64, double }

define void @test3() {
dependentGraph243.exit:
  %subject19 = load %pair* undef                     ; <%1> [#uses=1]
  %0 = extractvalue %pair %subject19, 1              ; <double> [#uses=2]
  %1 = select i1 undef, double %0, double undef   ; <double> [#uses=1]
  %2 = select i1 undef, double %1, double %0      ; <double> [#uses=1]
  %3 = insertvalue %pair undef, double %2, 1         ; <%1> [#uses=1]
  store %pair %3, %pair* undef
  ret void
}

; PR6605
define i64 @test4(i8* %P) nounwind ssp {
entry:
  %tmp1 = load i8* %P                           ; <i8> [#uses=3]
  %tobool = icmp eq i8 %tmp1, 0                   ; <i1> [#uses=1]
  %tmp58 = sext i1 %tobool to i8                  ; <i8> [#uses=1]
  %mul.i = and i8 %tmp58, %tmp1                   ; <i8> [#uses=1]
  %conv6 = zext i8 %mul.i to i32                  ; <i32> [#uses=1]
  %cmp = icmp ne i8 %tmp1, 1                      ; <i1> [#uses=1]
  %conv11 = zext i1 %cmp to i32                   ; <i32> [#uses=1]
  %call12 = tail call i32 @safe(i32 %conv11) nounwind ; <i32> [#uses=1]
  %and = and i32 %conv6, %call12                  ; <i32> [#uses=1]
  %tobool13 = icmp eq i32 %and, 0                 ; <i1> [#uses=1]
  br i1 %tobool13, label %if.else, label %return

if.else:                                          ; preds = %entry
  br label %return

return:                                           ; preds = %if.else, %entry
  ret i64 undef
}

declare i32 @safe(i32)

; PR6607
define fastcc void @test5(i32 %FUNC) nounwind {
foo:
  %0 = load i8* undef, align 1                    ; <i8> [#uses=3]
  %1 = sext i8 %0 to i32                          ; <i32> [#uses=2]
  %2 = zext i8 %0 to i32                          ; <i32> [#uses=1]
  %tmp1.i5037 = urem i32 %2, 10                   ; <i32> [#uses=1]
  %tmp.i5038 = icmp ugt i32 %tmp1.i5037, 15       ; <i1> [#uses=1]
  %3 = zext i1 %tmp.i5038 to i8                   ; <i8> [#uses=1]
  %4 = icmp slt i8 %0, %3                         ; <i1> [#uses=1]
  %5 = add nsw i32 %1, 256                        ; <i32> [#uses=1]
  %storemerge.i.i57 = select i1 %4, i32 %5, i32 %1 ; <i32> [#uses=1]
  %6 = shl i32 %storemerge.i.i57, 16              ; <i32> [#uses=1]
  %7 = sdiv i32 %6, -256                          ; <i32> [#uses=1]
  %8 = trunc i32 %7 to i8                         ; <i8> [#uses=1]
  store i8 %8, i8* undef, align 1
  ret void
}


; Crash commoning identical asms.
; PR6803
define void @test6(i1 %C) nounwind optsize ssp {
entry:
  br i1 %C, label %do.body55, label %do.body92

do.body55:                                        ; preds = %if.else36
  call void asm sideeffect "foo", "~{dirflag},~{fpsr},~{flags}"() nounwind, !srcloc !0
  ret void

do.body92:                                        ; preds = %if.then66
  call void asm sideeffect "foo", "~{dirflag},~{fpsr},~{flags}"() nounwind, !srcloc !1
  ret void
}

!0 = metadata !{i32 633550}                       
!1 = metadata !{i32 634261}                       


; Crash during XOR optimization.
; <rdar://problem/7869290>

define void @test7() nounwind ssp {
entry:
  br i1 undef, label %bb14, label %bb67

bb14:
  %tmp0 = trunc i16 undef to i1
  %tmp1 = load i8* undef, align 8
  %tmp2 = shl i8 %tmp1, 4
  %tmp3 = lshr i8 %tmp2, 7
  %tmp4 = trunc i8 %tmp3 to i1
  %tmp5 = icmp ne i1 %tmp0, %tmp4
  br i1 %tmp5, label %bb14, label %bb67

bb67:
  ret void
}

; Crash when trying to copy AH to AL.
; PR7540
define void @copy8bitregs() nounwind {
entry:
  %div.i = sdiv i32 115200, 0
  %shr8.i = lshr i32 %div.i, 8
  %conv4.i = trunc i32 %shr8.i to i8
  call void asm sideeffect "outb $0, ${1:w}", "{ax},N{dx},~{dirflag},~{fpsr},~{flags}"(i8 %conv4.i, i32 1017) nounwind
  unreachable
}

; Crash trying to form conditional increment with fp value.
; PR8981
define i32 @test9(double %X) ssp align 2 {
entry:
  %0 = fcmp one double %X, 0.000000e+00
  %cond = select i1 %0, i32 1, i32 2
  ret i32 %cond
}


; PR8514 - Crash in match address do to "heroics" turning and-of-shift into
; shift of and.
%struct.S0 = type { i8, [2 x i8], i8 }

define void @func_59(i32 %p_63) noreturn nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc44, %entry
  %p_63.addr.1 = phi i32 [ %p_63, %entry ], [ 0, %for.inc44 ]
  %l_74.0 = phi i32 [ 0, %entry ], [ %add46, %for.inc44 ]
  br i1 undef, label %for.inc44, label %bb.nph81

bb.nph81:                                         ; preds = %for.body
  %tmp98 = add i32 %p_63.addr.1, 0
  br label %for.body22

for.body22:                                       ; preds = %for.body22, %bb.nph81
  %l_75.077 = phi i64 [ %ins, %for.body22 ], [ undef, %bb.nph81 ]
  %tmp110 = trunc i64 %l_75.077 to i32
  %tmp111 = and i32 %tmp110, 65535
  %arrayidx32.0 = getelementptr [9 x [5 x [2 x %struct.S0]]]* undef, i32 0, i32 %l_74.0, i32 %tmp98, i32 %tmp111, i32 0
  store i8 1, i8* %arrayidx32.0, align 4
  %tmp106 = shl i32 %tmp110, 2
  %tmp107 = and i32 %tmp106, 262140
  %scevgep99.sum114 = or i32 %tmp107, 1
  %arrayidx32.1.1 = getelementptr [9 x [5 x [2 x %struct.S0]]]* undef, i32 0, i32 %l_74.0, i32 %tmp98, i32 0, i32 1, i32 %scevgep99.sum114
  store i8 0, i8* %arrayidx32.1.1, align 1
  %ins = or i64 undef, undef
  br label %for.body22

for.inc44:                                        ; preds = %for.body
  %add46 = add i32 %l_74.0, 1
  br label %for.body
}
