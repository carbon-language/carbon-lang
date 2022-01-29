target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"
; This file mainly tests that one of the ISEL instruction in the group uses the same register for operand RT, RA, RB
; This redudant ISEL is introduced during simple register coalescing stage.

; Simple register coalescing first create the foldable ISEL instruction as we have seen in expand-foldable-isel.ll:
; %vreg85<def> = ISEL8 %vreg83, %vreg83, %vreg33:sub_eq

; Later the register coalescer figures out it could further coalesce %vreg85 with %vreg83:
; merge %vreg85:1@2288r into %vreg83:5@400B --> @400B
; erased:	2288r	%vreg85<def> = COPY %vreg83

; After that we have:
; updated: 1504B	%vreg83<def> = ISEL8 %vreg83, %vreg83, %vreg33:sub_eq

; RUN: llc -verify-machineinstrs -O2 -ppc-asm-full-reg-names -mcpu=pwr7 -ppc-gen-isel=true < %s | FileCheck %s --check-prefix=CHECK-GEN-ISEL-TRUE
; RUN: llc -verify-machineinstrs -O2 -ppc-asm-full-reg-names -mcpu=pwr7 -ppc-gen-isel=false < %s | FileCheck %s --implicit-check-not isel

@.str = private unnamed_addr constant [3 x i8] c"]]\00", align 1
@.str.1 = private unnamed_addr constant [35 x i8] c"Index < Length && \22Invalid index!\22\00", align 1
@.str.2 = private unnamed_addr constant [50 x i8] c"/home/jtony/src/llvm/include/llvm/ADT/StringRef.h\00", align 1
@__PRETTY_FUNCTION__._ZNK4llvm9StringRefixEm = private unnamed_addr constant [47 x i8] c"char llvm::StringRef::operator[](size_t) const\00", align 1
@.str.3 = private unnamed_addr constant [95 x i8] c"(data || length == 0) && \22StringRef cannot be built from a NULL argument with non-null length\22\00", align 1
@__PRETTY_FUNCTION__._ZN4llvm9StringRefC2EPKcm = private unnamed_addr constant [49 x i8] c"llvm::StringRef::StringRef(const char *, size_t)\00", align 1
define i64 @_Z3fn1N4llvm9StringRefE([2 x i64] %Str.coerce) {
entry:
  %Str.coerce.fca.0.extract = extractvalue [2 x i64] %Str.coerce, 0
  %Str.coerce.fca.1.extract = extractvalue [2 x i64] %Str.coerce, 1
  br label %while.cond.outer
while.cond.outer:
  %Str.sroa.0.0.ph = phi i64 [ %8, %_ZNK4llvm9StringRef6substrEmm.exit ], [ %Str.coerce.fca.0.extract, %entry ]
  %.sink.ph = phi i64 [ %sub.i, %_ZNK4llvm9StringRef6substrEmm.exit ], [ %Str.coerce.fca.1.extract, %entry ]
  %BracketDepth.0.ph = phi i64 [ %BracketDepth.1, %_ZNK4llvm9StringRef6substrEmm.exit ], [ undef, %entry ]
  %cmp65 = icmp eq i64 %BracketDepth.0.ph, 0
  br i1 %cmp65, label %while.cond.us.preheader, label %while.cond.preheader
while.cond.us.preheader:
  br label %while.cond.us
while.cond.preheader:
  %cmp.i34129 = icmp eq i64 %.sink.ph, 0
  br i1 %cmp.i34129, label %cond.false.i.loopexit135, label %_ZNK4llvm9StringRefixEm.exit.preheader
_ZNK4llvm9StringRefixEm.exit.preheader:
  br label %_ZNK4llvm9StringRefixEm.exit
while.cond.us:
  %Str.sroa.0.0.us = phi i64 [ %3, %_ZNK4llvm9StringRef6substrEmm.exit50.us ], [ %Str.sroa.0.0.ph, %while.cond.us.preheader ]
  %.sink.us = phi i64 [ %sub.i41.us, %_ZNK4llvm9StringRef6substrEmm.exit50.us ], [ %.sink.ph, %while.cond.us.preheader ]
  %cmp.i30.us = icmp ult i64 %.sink.us, 2
  br i1 %cmp.i30.us, label %if.end.us, label %if.end.i.i.us
if.end.i.i.us:
  %0 = inttoptr i64 %Str.sroa.0.0.us to i8*
  %call.i.i.us = tail call signext i32 @memcmp(i8* %0, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i64 2)
  %phitmp.i.us = icmp eq i32 %call.i.i.us, 0
  br i1 %phitmp.i.us, label %if.then, label %_ZNK4llvm9StringRefixEm.exit.us
if.end.us:
  %cmp.i34.us = icmp eq i64 %.sink.us, 0
  br i1 %cmp.i34.us, label %cond.false.i.loopexit, label %_ZNK4llvm9StringRefixEm.exit.us
_ZNK4llvm9StringRefixEm.exit.us:
  %1 = inttoptr i64 %Str.sroa.0.0.us to i8*
  %2 = load i8, i8* %1, align 1
  switch i8 %2, label %_ZNK4llvm9StringRef6substrEmm.exit.loopexit [
    i8 92, label %if.then4.us
    i8 93, label %if.then9
  ]
if.then4.us:
  %.sroa.speculated12.i38.us = select i1 %cmp.i30.us, i64 %.sink.us, i64 2
  %add.ptr.i40.us = getelementptr inbounds i8, i8* %1, i64 %.sroa.speculated12.i38.us
  %sub.i41.us = sub i64 %.sink.us, %.sroa.speculated12.i38.us
  %tobool.i.i44.us = icmp ne i8* %add.ptr.i40.us, null
  %cmp.i4.i45.us = icmp eq i64 %sub.i41.us, 0
  %or.cond.i.i46.us = or i1 %tobool.i.i44.us, %cmp.i4.i45.us
  br i1 %or.cond.i.i46.us, label %_ZNK4llvm9StringRef6substrEmm.exit50.us, label %cond.false.i.i47.loopexit
_ZNK4llvm9StringRef6substrEmm.exit50.us:
  %3 = ptrtoint i8* %add.ptr.i40.us to i64
  br label %while.cond.us
if.then:
  ret i64 undef
cond.false.i.loopexit:
  br label %cond.false.i
cond.false.i.loopexit134:
  br label %cond.false.i
cond.false.i.loopexit135:
  br label %cond.false.i
cond.false.i:
  tail call void @__assert_fail(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i64 0, i64 0), i8* getelementptr inbounds ([50 x i8], [50 x i8]* @.str.2, i64 0, i64 0), i32 zeroext 225, i8* getelementptr inbounds ([47 x i8], [47 x i8]* @__PRETTY_FUNCTION__._ZNK4llvm9StringRefixEm, i64 0, i64 0))
  unreachable
_ZNK4llvm9StringRefixEm.exit:
  %.sink131 = phi i64 [ %sub.i41, %_ZNK4llvm9StringRef6substrEmm.exit50 ], [ %.sink.ph, %_ZNK4llvm9StringRefixEm.exit.preheader ]
  %Str.sroa.0.0130 = phi i64 [ %6, %_ZNK4llvm9StringRef6substrEmm.exit50 ], [ %Str.sroa.0.0.ph, %_ZNK4llvm9StringRefixEm.exit.preheader ]
  %4 = inttoptr i64 %Str.sroa.0.0130 to i8*
  %5 = load i8, i8* %4, align 1
  switch i8 %5, label %_ZNK4llvm9StringRef6substrEmm.exit.loopexit132 [
    i8 92, label %if.then4
    i8 93, label %if.end10
  ]
if.then4:
  %cmp.i.i37 = icmp ult i64 %.sink131, 2
  %.sroa.speculated12.i38 = select i1 %cmp.i.i37, i64 %.sink131, i64 2
  %add.ptr.i40 = getelementptr inbounds i8, i8* %4, i64 %.sroa.speculated12.i38
  %sub.i41 = sub i64 %.sink131, %.sroa.speculated12.i38
  %tobool.i.i44 = icmp ne i8* %add.ptr.i40, null
  %cmp.i4.i45 = icmp eq i64 %sub.i41, 0
  %or.cond.i.i46 = or i1 %tobool.i.i44, %cmp.i4.i45
  br i1 %or.cond.i.i46, label %_ZNK4llvm9StringRef6substrEmm.exit50, label %cond.false.i.i47.loopexit133
cond.false.i.i47.loopexit:
  br label %cond.false.i.i47
cond.false.i.i47.loopexit133:
  br label %cond.false.i.i47
cond.false.i.i47:
  tail call void @__assert_fail(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([50 x i8], [50 x i8]* @.str.2, i64 0, i64 0), i32 zeroext 90, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @__PRETTY_FUNCTION__._ZN4llvm9StringRefC2EPKcm, i64 0, i64 0))
  unreachable
_ZNK4llvm9StringRef6substrEmm.exit50:
  %6 = ptrtoint i8* %add.ptr.i40 to i64
  %cmp.i34 = icmp eq i64 %sub.i41, 0
  br i1 %cmp.i34, label %cond.false.i.loopexit134, label %_ZNK4llvm9StringRefixEm.exit
if.then9:
  tail call void @exit(i32 signext 1)
  unreachable
if.end10:
  %dec = add i64 %BracketDepth.0.ph, -1
  br label %_ZNK4llvm9StringRef6substrEmm.exit
_ZNK4llvm9StringRef6substrEmm.exit.loopexit:
  br label %_ZNK4llvm9StringRef6substrEmm.exit
_ZNK4llvm9StringRef6substrEmm.exit.loopexit132:
  br label %_ZNK4llvm9StringRef6substrEmm.exit
_ZNK4llvm9StringRef6substrEmm.exit:
  %.sink76 = phi i64 [ %.sink131, %if.end10 ], [ %.sink.us, %_ZNK4llvm9StringRef6substrEmm.exit.loopexit ], [ %.sink131, %_ZNK4llvm9StringRef6substrEmm.exit.loopexit132 ]
  %7 = phi i8* [ %4, %if.end10 ], [ %1, %_ZNK4llvm9StringRef6substrEmm.exit.loopexit ], [ %4, %_ZNK4llvm9StringRef6substrEmm.exit.loopexit132 ]
  %BracketDepth.1 = phi i64 [ %dec, %if.end10 ], [ 0, %_ZNK4llvm9StringRef6substrEmm.exit.loopexit ], [ %BracketDepth.0.ph, %_ZNK4llvm9StringRef6substrEmm.exit.loopexit132 ]
  %sub.i = add i64 %.sink76, -1
  %add.ptr.i = getelementptr inbounds i8, i8* %7, i64 1
  %8 = ptrtoint i8* %add.ptr.i to i64
  br label %while.cond.outer

; CHECK-LABEL: @_Z3fn1N4llvm9StringRefE
; Unecessary ISEL (all the registers are the same) is always removed
; CHECK-GEN-ISEL-TRUE-NOT: iseleq [[SAME:r[0-9]+]], [[SAME]], [[SAME]]
; CHECK-GEN-ISEL-TRUE: iseleq [[SAME:r[0-9]+]], {{r[0-9]+}}, [[SAME]]
; CHECK: bc 12, eq, [[TRUE:.LBB[0-9]+]]
; CHECK-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NEXT: [[TRUE]]
; CHECK-NEXT: # in Loop: Header
; CHECK-NEXT: addi {{r[0-9]+}}, {{r[0-9]+}}, 0
; CHECK-NEXT: [[SUCCESSOR]]
}



declare void @exit(i32 signext)
declare signext i32 @memcmp(i8* nocapture, i8* nocapture, i64)
declare void @__assert_fail(i8*, i8*, i32 zeroext, i8*)
