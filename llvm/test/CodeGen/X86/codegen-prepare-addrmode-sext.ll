; RUN: opt -S -codegenprepare %s -o - | FileCheck %s
; This file tests the different cases what are involved when codegen prepare
; tries to get sign extension out of the way of addressing mode.
; This tests require an actual target as addressing mode decisions depends
; on the target.

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"


; Check that we correctly promote both operands of the promotable add.
; CHECK-LABEL: @twoArgsPromotion
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i32 %arg1 to i64
; CHECK: [[ARG2SEXT:%[a-zA-Z_0-9-]+]] = sext i32 %arg2 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], [[ARG2SEXT]]
; CHECK: inttoptr i64 [[PROMOTED]] to i8*
; CHECK: ret
define i8 @twoArgsPromotion(i32 %arg1, i32 %arg2) {
  %add = add nsw i32 %arg1, %arg2 
  %sextadd = sext i32 %add to i64
  %base = inttoptr i64 %sextadd to i8*
  %res = load i8* %base
  ret i8 %res
}

; Check that we do not promote both operands of the promotable add when
; the instruction will not be folded into the addressing mode.
; Otherwise, we will increase the number of instruction executed.
; (This is a heuristic of course, because the new sext could have been
; merged with something else.)
; CHECK-LABEL: @twoArgsNoPromotion
; CHECK: add nsw i32 %arg1, %arg2
; CHECK: ret
define i8 @twoArgsNoPromotion(i32 %arg1, i32 %arg2, i8* %base) {
  %add = add nsw i32 %arg1, %arg2 
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  ret i8 %res
}

; Check that we do not promote when the related instruction does not have
; the nsw flag.
; CHECK-LABEL: @noPromotion
; CHECK-NOT: add i64
; CHECK: ret
define i8 @noPromotion(i32 %arg1, i32 %arg2, i8* %base) {
  %add = add i32 %arg1, %arg2 
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  ret i8 %res
}

; Check that we correctly promote constant arguments.
; CHECK-LABEL: @oneArgPromotion
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i32 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: getelementptr inbounds i8* %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotion(i32 %arg1, i8* %base) {
  %add = add nsw i32 %arg1, 1 
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  ret i8 %res
}

; Check that we do not promote truncate when we cannot determine the
; bits that are dropped.
; CHECK-LABEL: @oneArgPromotionBlockTrunc1
; CHECK: [[ARG1TRUNC:%[a-zA-Z_0-9-]+]] = trunc i32 %arg1 to i8
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i8 [[ARG1TRUNC]] to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: getelementptr inbounds i8* %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionBlockTrunc1(i32 %arg1, i8* %base) {
  %trunc = trunc i32 %arg1 to i8
  %add = add nsw i8 %trunc, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  ret i8 %res
}

; Check that we do not promote truncate when we cannot determine all the
; bits that are dropped.
; CHECK-LABEL: @oneArgPromotionBlockTrunc2
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i16 %arg1 to i32
; CHECK: [[ARG1TRUNC:%[a-zA-Z_0-9-]+]] = trunc i32 [[ARG1SEXT]] to i8
; CHECK: [[ARG1SEXT64:%[a-zA-Z_0-9-]+]] = sext i8 [[ARG1TRUNC]] to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT64]], 1
; CHECK: getelementptr inbounds i8* %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionBlockTrunc2(i16 %arg1, i8* %base) {
  %sextarg1 = sext i16 %arg1 to i32
  %trunc = trunc i32 %sextarg1 to i8
  %add = add nsw i8 %trunc, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  ret i8 %res
}

; Check that we are able to promote truncate when we know all the bits
; that are dropped.
; CHECK-LABEL: @oneArgPromotionPassTruncKeepSExt
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i1 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: getelementptr inbounds i8* %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionPassTruncKeepSExt(i1 %arg1, i8* %base) {
  %sextarg1 = sext i1 %arg1 to i32
  %trunc = trunc i32 %sextarg1 to i8
  %add = add nsw i8 %trunc, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  ret i8 %res
}

; On X86 truncate are free. Check that we are able to promote the add
; to be used as addressing mode and that we insert a truncate for the other
; use. 
; CHECK-LABEL: @oneArgPromotionTruncInsert
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: [[TRUNC:%[a-zA-Z_0-9-]+]] = trunc i64 [[PROMOTED]] to i8
; CHECK: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i8* %base, i64 [[PROMOTED]]
; CHECK: [[LOAD:%[a-zA-Z_0-9-]+]] = load i8* [[GEP]]
; CHECK: add i8 [[LOAD]], [[TRUNC]]
; CHECK: ret
define i8 @oneArgPromotionTruncInsert(i8 %arg1, i8* %base) {
  %add = add nsw i8 %arg1, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  %finalres = add i8 %res, %add
  ret i8 %finalres
}

; Cannot sext from a larger type than the promoted type.
; CHECK-LABEL: @oneArgPromotionLargerType
; CHECK: [[ARG1TRUNC:%[a-zA-Z_0-9-]+]] = trunc i128 %arg1 to i8
; CHECK: [[ARG1SEXT64:%[a-zA-Z_0-9-]+]] = sext i8 [[ARG1TRUNC]] to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT64]], 1
; CHECK: getelementptr inbounds i8* %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionLargerType(i128 %arg1, i8* %base) {
  %trunc = trunc i128 %arg1 to i8
  %add = add nsw i8 %trunc, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  %finalres = add i8 %res, %add
  ret i8 %finalres
}

; Use same inserted trunc
; On X86 truncate are free. Check that we are able to promote the add
; to be used as addressing mode and that we insert a truncate for
; *all* the other uses. 
; CHECK-LABEL: @oneArgPromotionTruncInsertSeveralUse
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: [[TRUNC:%[a-zA-Z_0-9-]+]] = trunc i64 [[PROMOTED]] to i8
; CHECK: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i8* %base, i64 [[PROMOTED]]
; CHECK: [[LOAD:%[a-zA-Z_0-9-]+]] = load i8* [[GEP]]
; CHECK: [[ADDRES:%[a-zA-Z_0-9-]+]] = add i8 [[LOAD]], [[TRUNC]]
; CHECK: add i8 [[ADDRES]], [[TRUNC]]
; CHECK: ret
define i8 @oneArgPromotionTruncInsertSeveralUse(i8 %arg1, i8* %base) {
  %add = add nsw i8 %arg1, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  %almostfinalres = add i8 %res, %add
  %finalres = add i8 %almostfinalres, %add
  ret i8 %finalres
}

; Check that the promoted instruction is used for all uses of the original
; sign extension.
; CHECK-LABEL: @oneArgPromotionSExtSeveralUse
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i8* %base, i64 [[PROMOTED]]
; CHECK: [[LOAD:%[a-zA-Z_0-9-]+]] = load i8* [[GEP]]
; CHECK: [[ADDRES:%[a-zA-Z_0-9-]+]] = zext i8 [[LOAD]] to i64
; CHECK: add i64 [[ADDRES]], [[PROMOTED]]
; CHECK: ret
define i64 @oneArgPromotionSExtSeveralUse(i8 %arg1, i8* %base) {
  %add = add nsw i8 %arg1, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  %almostfinalres = zext i8 %res to i64
  %finalres = add i64 %almostfinalres, %sextadd
  ret i64 %finalres
}

; Check all types of rollback mechanism.
; For this test, the sign extension stays in place.
; However, the matching process goes until promoting both the operands
; of the first promotable add implies.
; At this point the rollback mechanism kicks in and restores the states
; until the addressing mode matcher is able to match something: in that
; case promote nothing.
; Along the way, the promotion mechanism involves:
; - Mutating the type of %promotableadd1 and %promotableadd2.
; - Creating a sext for %arg1 and %arg2.
; - Creating a trunc for a use of %promotableadd1.
; - Replacing a bunch of uses.
; - Setting the operands of the promoted instruction with the promoted values.
; - Moving instruction around (mainly sext when promoting instruction).
; Each type of those promotions has to be undo at least once during this
; specific test. 
; CHECK-LABEL: @twoArgsPromotionNest
; CHECK: [[ORIG:%[a-zA-Z_0-9-]+]] = add nsw i32 %arg1, %arg2
; CHECK: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ORIG]], [[ORIG]]
; CHECK: [[SEXT:%[a-zA-Z_0-9-]+]] = sext i32 [[ADD]] to i64
; CHECK: getelementptr inbounds i8* %base, i64 [[SEXT]]
; CHECK: ret
define i8 @twoArgsPromotionNest(i32 %arg1, i32 %arg2, i8* %base) {
  %promotableadd1 = add nsw i32 %arg1, %arg2
  %promotableadd2 = add nsw i32 %promotableadd1, %promotableadd1 
  %sextadd = sext i32 %promotableadd2 to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  ret i8 %res
}

; Test the InstructionRemover undo, which was the only one not
; kicked in the previous test.
; The matcher first promotes the add, removes the trunc and promotes
; the sext of arg1.
; Then, the matcher cannot use an addressing mode r + r + r, thus it
; rolls back. 
; CHECK-LABEL: @twoArgsNoPromotionRemove
; CHECK: [[SEXTARG1:%[a-zA-Z_0-9-]+]] = sext i1 %arg1 to i32
; CHECK: [[TRUNC:%[a-zA-Z_0-9-]+]] = trunc i32 [[SEXTARG1]] to i8
; CHECK: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i8 [[TRUNC]], %arg2
; CHECK: [[SEXT:%[a-zA-Z_0-9-]+]] = sext i8 [[ADD]] to i64
; CHECK: getelementptr inbounds i8* %base, i64 [[SEXT]]
; CHECK: ret
define i8 @twoArgsNoPromotionRemove(i1 %arg1, i8 %arg2, i8* %base) {
  %sextarg1 = sext i1 %arg1 to i32
  %trunc = trunc i32 %sextarg1 to i8
  %add = add nsw i8 %trunc, %arg2 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8* %base, i64 %sextadd
  %res = load i8* %arrayidx
  ret i8 %res
}
