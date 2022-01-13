; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win64 | FileCheck %s
; RUN: opt -codegenprepare < %s -mtriple=x86_64-apple-macosx -S | FileCheck %s --check-prefix=OPTALL --check-prefix=OPT --check-prefix=NONSTRESS
; RUN: opt -codegenprepare < %s -mtriple=x86_64-apple-macosx -S -stress-cgp-ext-ld-promotion | FileCheck %s --check-prefix=OPTALL --check-prefix=OPT --check-prefix=STRESS
; RUN: opt -codegenprepare < %s -mtriple=x86_64-apple-macosx -S -disable-cgp-ext-ld-promotion | FileCheck %s --check-prefix=OPTALL --check-prefix=DISABLE

; rdar://7304838
; CodeGenPrepare should move the zext into the block with the load
; so that SelectionDAG can select it with the load.
;
; CHECK-LABEL: foo:
; CHECK: movsbl ({{%rdi|%rcx}}), %eax
;
; OPTALL-LABEL: @foo
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p
; OPTALL-NEXT: [[ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; OPTALL: store i32 [[ZEXT]], i32* %q
; OPTALL: ret
define void @foo(i8* %p, i32* %q) {
entry:
  %t = load i8, i8* %p
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %t to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}

; Check that we manage to form a zextload is an operation with only one
; argument to explicitly extend is in the way.
; OPTALL-LABEL: @promoteOneArg
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p
; OPT-NEXT: [[ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; OPT-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT]], 2
; Make sure the operation is not promoted when the promotion pass is disabled.
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8 [[LD]], 2
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
; OPTALL: store i32 [[RES]], i32* %q
; OPTALL: ret
define void @promoteOneArg(i8* %p, i32* %q) {
entry:
  %t = load i8, i8* %p
  %add = add nuw i8 %t, 2
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %add to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}

; Check that we manage to form a sextload is an operation with only one
; argument to explicitly extend is in the way.
; Version with sext.
; OPTALL-LABEL: @promoteOneArgSExt
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p
; OPT-NEXT: [[SEXT:%[a-zA-Z_0-9-]+]] = sext i8 [[LD]] to i32
; OPT-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i32 [[SEXT]], 2
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i8 [[LD]], 2
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = sext i8 [[ADD]] to i32
; OPTALL: store i32 [[RES]], i32* %q
; OPTALL: ret
define void @promoteOneArgSExt(i8* %p, i32* %q) {
entry:
  %t = load i8, i8* %p
  %add = add nsw i8 %t, 2
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = sext i8 %add to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}

; Check that we manage to form a zextload is an operation with two
; arguments to explicitly extend is in the way.
; Extending %add will create two extensions:
; 1. One for %b.
; 2. One for %t.
; #1 will not be removed as we do not know anything about %b.
; #2 may not be merged with the load because %t is used in a comparison.
; Since two extensions may be emitted in the end instead of one before the
; transformation, the regular heuristic does not apply the optimization. 
; 
; OPTALL-LABEL: @promoteTwoArgZext
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p
;
; STRESS-NEXT: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; STRESS-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = zext i8 %b to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXTLD]], [[ZEXTB]]
;
; NONSTRESS: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8 [[LD]], %b
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
;
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8 [[LD]], %b
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
;
; OPTALL: store i32 [[RES]], i32* %q
; OPTALL: ret
define void @promoteTwoArgZext(i8* %p, i32* %q, i8 %b) {
entry:
  %t = load i8, i8* %p
  %add = add nuw i8 %t, %b
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %add to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}

; Check that we manage to form a sextload is an operation with two
; arguments to explicitly extend is in the way.
; Version with sext.
; OPTALL-LABEL: @promoteTwoArgSExt
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p
;
; STRESS-NEXT: [[SEXTLD:%[a-zA-Z_0-9-]+]] = sext i8 [[LD]] to i32
; STRESS-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i8 %b to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i32 [[SEXTLD]], [[SEXTB]]
;
; NONSTRESS: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i8 [[LD]], %b
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = sext i8 [[ADD]] to i32
;
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i8 [[LD]], %b
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = sext i8 [[ADD]] to i32
; OPTALL: store i32 [[RES]], i32* %q
; OPTALL: ret
define void @promoteTwoArgSExt(i8* %p, i32* %q, i8 %b) {
entry:
  %t = load i8, i8* %p
  %add = add nsw i8 %t, %b
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = sext i8 %add to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}

; Check that we do not a zextload if we need to introduce more than
; one additional extension.
; OPTALL-LABEL: @promoteThreeArgZext
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p
;
; STRESS-NEXT: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; STRESS-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = zext i8 %b to i32
; STRESS-NEXT: [[TMP:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXTLD]], [[ZEXTB]]
; STRESS-NEXT: [[ZEXTC:%[a-zA-Z_0-9-]+]] = zext i8 %c to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw i32 [[TMP]], [[ZEXTC]]
;
; NONSTRESS-NEXT: [[TMP:%[a-zA-Z_0-9-]+]] = add nuw i8 [[LD]], %b
; NONSTRESS-NEXT: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8 [[TMP]], %c
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
;
; DISABLE: add nuw i8
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
;
; OPTALL: store i32 [[RES]], i32* %q
; OPTALL: ret
define void @promoteThreeArgZext(i8* %p, i32* %q, i8 %b, i8 %c) {
entry:
  %t = load i8, i8* %p
  %tmp = add nuw i8 %t, %b
  %add = add nuw i8 %tmp, %c
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %add to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}

; Check that we manage to form a zextload after promoting and merging
; two extensions.
; OPTALL-LABEL: @promoteMergeExtArgZExt
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p
;
; STRESS-NEXT: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; STRESS-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = zext i16 %b to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXTLD]], [[ZEXTB]]
;
; NONSTRESS: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i16
; NONSTRESS: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i16 [[ZEXTLD]], %b
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = zext i16 [[ADD]] to i32
;
; DISABLE: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i16
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i16 [[ZEXTLD]], %b
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = zext i16 [[ADD]] to i32
;
; OPTALL: store i32 [[RES]], i32* %q
; OPTALL: ret
define void @promoteMergeExtArgZExt(i8* %p, i32* %q, i16 %b) {
entry:
  %t = load i8, i8* %p
  %ext = zext i8 %t to i16
  %add = add nuw i16 %ext, %b
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i16 %add to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}

; Check that we manage to form a sextload after promoting and merging
; two extensions.
; Version with sext.
; OPTALL-LABEL: @promoteMergeExtArgSExt
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p
;
; STRESS-NEXT: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; STRESS-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = sext i16 %b to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXTLD]], [[ZEXTB]]
;
; NONSTRESS: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i16
; NONSTRESS: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i16 [[ZEXTLD]], %b
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = sext i16 [[ADD]] to i32
;
; DISABLE: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i16
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i16 [[ZEXTLD]], %b
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = sext i16 [[ADD]] to i32
; OPTALL: store i32 [[RES]], i32* %q
; OPTALL: ret
define void @promoteMergeExtArgSExt(i8* %p, i32* %q, i16 %b) {
entry:
  %t = load i8, i8* %p
  %ext = zext i8 %t to i16
  %add = add nsw i16 %ext, %b
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = sext i16 %add to i32
  store i32 %s, i32* %q
  ret void
false:
  ret void
}

; Check that we manage to catch all the extload opportunities that are exposed
; by the different iterations of codegen prepare.
; Moreover, check that we do not promote more than we need to.
; Here is what is happening in this test (not necessarly in this order):
; 1. We try to promote the operand of %sextadd.
;    a. This creates one sext of %ld2 and one of %zextld
;    b. The sext of %ld2 can be combine with %ld2, so we remove one sext but
;       introduced one. This is fine with the current heuristic: neutral.
;    => We have one zext of %zextld left and we created one sext of %ld2.
; 2. We try to promote the operand of %sextaddza.
;    a. This creates one sext of %zexta and one of %zextld
;    b. The sext of %zexta can be combined with the zext of %a.
;    c. The sext of %zextld leads to %ld and can be combined with it. This is
;       done by promoting %zextld. This is fine with the current heuristic:
;       neutral.
;    => We have created a new zext of %ld and we created one sext of %zexta.
; 3. We try to promote the operand of %sextaddb.
;    a. This creates one sext of %b and one of %zextld
;    b. The sext of %b is a dead-end, nothing to be done.
;    c. Same thing as 2.c. happens.
;    => We have created a new zext of %ld and we created one sext of %b.
; 4. We try to promote the operand of the zext of %zextld introduced in #1.
;    a. Same thing as 2.c. happens.
;    b. %zextld does not have any other uses. It is dead coded.
;    => We have created a new zext of %ld and we removed a zext of %zextld and
;       a zext of %ld.
; Currently we do not try to reuse existing extensions, so in the end we have
; 3 identical zext of %ld. The extensions will be CSE'ed by SDag.
;
; OPTALL-LABEL: @severalPromotions
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %addr1
; OPT-NEXT: [[ZEXTLD1_1:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[ZEXTLD1_2:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[LD2:%[a-zA-Z_0-9-]+]] = load i32, i32* %addr2
; OPT-NEXT: [[SEXTLD2:%[a-zA-Z_0-9-]+]] = sext i32 [[LD2]] to i64
; OPT-NEXT: [[ZEXTLD1_3:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i64 [[SEXTLD2]], [[ZEXTLD1_3]]
; OPT-NEXT: [[ZEXTLD1_4:%[a-zA-Z_0-9-]+]] = zext i8 %a to i64
; OPT-NEXT: [[RESZA:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ZEXTLD1_4]], [[ZEXTLD1_2]]
; OPT-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i32 %b to i64
; OPT-NEXT: [[RESB:%[a-zA-Z_0-9-]+]] = add nsw i64 [[SEXTB]], [[ZEXTLD1_1]]
;
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i32
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]]  = sext i32 [[ADD]] to i64
; DISABLE: [[ADDZA:%[a-zA-Z_0-9-]+]] = add nsw i32
; DISABLE: [[RESZA:%[a-zA-Z_0-9-]+]]  = sext i32 [[ADDZA]] to i64
; DISABLE: [[ADDB:%[a-zA-Z_0-9-]+]] = add nsw i32
; DISABLE: [[RESB:%[a-zA-Z_0-9-]+]]  = sext i32 [[ADDB]] to i64
;
; OPTALL: call void @dummy(i64 [[RES]], i64 [[RESZA]], i64 [[RESB]])
; OPTALL: ret
define void @severalPromotions(i8* %addr1, i32* %addr2, i8 %a, i32 %b) {
  %ld = load i8, i8* %addr1
  %zextld = zext i8 %ld to i32
  %ld2 = load i32, i32* %addr2
  %add = add nsw i32 %ld2, %zextld
  %sextadd = sext i32 %add to i64
  %zexta = zext i8 %a to i32
  %addza = add nsw i32 %zexta, %zextld
  %sextaddza = sext i32 %addza to i64
  %addb = add nsw i32 %b, %zextld
  %sextaddb = sext i32 %addb to i64
  call void @dummy(i64 %sextadd, i64 %sextaddza, i64 %sextaddb)
  ret void
}

declare void @dummy(i64, i64, i64)

; Make sure we do not try to promote vector types since the type promotion
; helper does not support them for now.
; OPTALL-LABEL: @vectorPromotion
; OPTALL: [[SHL:%[a-zA-Z_0-9-]+]] = shl nuw nsw <2 x i32> zeroinitializer, <i32 8, i32 8>
; OPTALL: [[ZEXT:%[a-zA-Z_0-9-]+]] = zext <2 x i32> [[SHL]] to <2 x i64>
; OPTALL: ret
define void @vectorPromotion() {
entry:
  %a = shl nuw nsw <2 x i32> zeroinitializer, <i32 8, i32 8>
  %b = zext <2 x i32> %a to <2 x i64>
  ret void
}

@a = common global i32 0, align 4
@c = common global [2 x i32] zeroinitializer, align 4

; PR21978.
; Make sure we support promotion of operands that produces a Value as opposed
; to an instruction.
; This used to cause a crash.
; OPTALL-LABEL: @promotionOfArgEndsUpInValue
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i16, i16* %addr

; OPT-NEXT: [[SEXT:%[a-zA-Z_0-9-]+]] = sext i16 [[LD]] to i32
; OPT-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw nsw i32 [[SEXT]], zext (i1 icmp ne (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @c, i64 0, i64 1), i32* @a) to i32)
;
; DISABLE-NEXT: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw nsw i16 [[LD]], zext (i1 icmp ne (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @c, i64 0, i64 1), i32* @a) to i16)
; DISABLE-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = sext i16 [[ADD]] to i32
;
; OPTALL-NEXT: ret i32 [[RES]]
define i32 @promotionOfArgEndsUpInValue(i16* %addr) {
entry:
  %val = load i16, i16* %addr
  %add = add nuw nsw i16 %val, zext (i1 icmp ne (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @c, i64 0, i64 1), i32* @a) to i16)
  %conv3 = sext i16 %add to i32
  ret i32 %conv3
}

; Check that we see that one zext can be derived from the other for free.
; OPTALL-LABEL: @promoteTwoArgZextWithSourceExtendedTwice
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, i8* %p

; OPT-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; OPT-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT32]], %b
; OPT-NEXT: [[RES64:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ZEXT64]], 12
; OPT-NEXT: store i32 [[RES32]], i32* %addr
; OPT-NEXT: store i64 [[RES64]], i64* %q
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[RES2_32:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT32]], 12
; DISABLE-NEXT: store i32 [[RES32]], i32* %addr
; DISABLE-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i32 [[RES2_32]] to i64
; DISABLE-NEXT: store i64 [[ZEXT64]], i64* %q
;
; OPTALL-NEXT: ret void
define void @promoteTwoArgZextWithSourceExtendedTwice(i8* %p, i64* %q, i32 %b, i32* %addr) {
entry:
  %t = load i8, i8* %p
  %zextt = zext i8 %t to i32
  %add = add nuw i32 %zextt, %b
  %add2 = add nuw i32 %zextt, 12
  store i32 %add, i32 *%addr
  %s = zext i32 %add2 to i64
  store i64 %s, i64* %q
  ret void
}
