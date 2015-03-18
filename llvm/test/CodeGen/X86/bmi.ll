; RUN: llc < %s -march=x86-64 -mattr=+bmi,+bmi2 | FileCheck %s

declare i8 @llvm.cttz.i8(i8, i1) nounwind readnone
declare i16 @llvm.cttz.i16(i16, i1) nounwind readnone
declare i32 @llvm.cttz.i32(i32, i1) nounwind readnone
declare i64 @llvm.cttz.i64(i64, i1) nounwind readnone

define i8 @t1(i8 %x) nounwind  {
  %tmp = tail call i8 @llvm.cttz.i8( i8 %x, i1 false )
  ret i8 %tmp
; CHECK-LABEL: t1:
; CHECK: tzcntl
}

define i16 @t2(i16 %x) nounwind  {
  %tmp = tail call i16 @llvm.cttz.i16( i16 %x, i1 false )
  ret i16 %tmp
; CHECK-LABEL: t2:
; CHECK: tzcntw
}

define i32 @t3(i32 %x) nounwind  {
  %tmp = tail call i32 @llvm.cttz.i32( i32 %x, i1 false )
  ret i32 %tmp
; CHECK-LABEL: t3:
; CHECK: tzcntl
}

define i32 @tzcnt32_load(i32* %x) nounwind  {
  %x1 = load i32, i32* %x
  %tmp = tail call i32 @llvm.cttz.i32(i32 %x1, i1 false )
  ret i32 %tmp
; CHECK-LABEL: tzcnt32_load:
; CHECK: tzcntl ({{.*}})
}

define i64 @t4(i64 %x) nounwind  {
  %tmp = tail call i64 @llvm.cttz.i64( i64 %x, i1 false )
  ret i64 %tmp
; CHECK-LABEL: t4:
; CHECK: tzcntq
}

define i8 @t5(i8 %x) nounwind  {
  %tmp = tail call i8 @llvm.cttz.i8( i8 %x, i1 true )
  ret i8 %tmp
; CHECK-LABEL: t5:
; CHECK: tzcntl
}

define i16 @t6(i16 %x) nounwind  {
  %tmp = tail call i16 @llvm.cttz.i16( i16 %x, i1 true )
  ret i16 %tmp
; CHECK-LABEL: t6:
; CHECK: tzcntw
}

define i32 @t7(i32 %x) nounwind  {
  %tmp = tail call i32 @llvm.cttz.i32( i32 %x, i1 true )
  ret i32 %tmp
; CHECK-LABEL: t7:
; CHECK: tzcntl
}

define i64 @t8(i64 %x) nounwind  {
  %tmp = tail call i64 @llvm.cttz.i64( i64 %x, i1 true )
  ret i64 %tmp
; CHECK-LABEL: t8:
; CHECK: tzcntq
}

define i32 @andn32(i32 %x, i32 %y) nounwind readnone {
  %tmp1 = xor i32 %x, -1
  %tmp2 = and i32 %y, %tmp1
  ret i32 %tmp2
; CHECK-LABEL: andn32:
; CHECK: andnl
}

define i32 @andn32_load(i32 %x, i32* %y) nounwind readnone {
  %y1 = load i32, i32* %y
  %tmp1 = xor i32 %x, -1
  %tmp2 = and i32 %y1, %tmp1
  ret i32 %tmp2
; CHECK-LABEL: andn32_load:
; CHECK: andnl ({{.*}})
}

define i64 @andn64(i64 %x, i64 %y) nounwind readnone {
  %tmp1 = xor i64 %x, -1
  %tmp2 = and i64 %tmp1, %y
  ret i64 %tmp2
; CHECK-LABEL: andn64:
; CHECK: andnq
}

define i32 @bextr32(i32 %x, i32 %y) nounwind readnone {
  %tmp = tail call i32 @llvm.x86.bmi.bextr.32(i32 %x, i32 %y)
  ret i32 %tmp
; CHECK-LABEL: bextr32:
; CHECK: bextrl
}

define i32 @bextr32_load(i32* %x, i32 %y) nounwind readnone {
  %x1 = load i32, i32* %x
  %tmp = tail call i32 @llvm.x86.bmi.bextr.32(i32 %x1, i32 %y)
  ret i32 %tmp
; CHECK-LABEL: bextr32_load:
; CHECK: bextrl {{.*}}, ({{.*}}), {{.*}}
}

declare i32 @llvm.x86.bmi.bextr.32(i32, i32) nounwind readnone

define i32 @bextr32b(i32 %x) nounwind uwtable readnone ssp {
  %1 = lshr i32 %x, 4
  %2 = and i32 %1, 4095
  ret i32 %2
; CHECK-LABEL: bextr32b:
; CHECK: bextrl
}

define i32 @bextr32b_load(i32* %x) nounwind uwtable readnone ssp {
  %1 = load i32, i32* %x
  %2 = lshr i32 %1, 4
  %3 = and i32 %2, 4095
  ret i32 %3
; CHECK-LABEL: bextr32b_load:
; CHECK: bextrl {{.*}}, ({{.*}}), {{.*}}
}

define i64 @bextr64(i64 %x, i64 %y) nounwind readnone {
  %tmp = tail call i64 @llvm.x86.bmi.bextr.64(i64 %x, i64 %y)
  ret i64 %tmp
; CHECK-LABEL: bextr64:
; CHECK: bextrq
}

declare i64 @llvm.x86.bmi.bextr.64(i64, i64) nounwind readnone

define i64 @bextr64b(i64 %x) nounwind uwtable readnone ssp {
  %1 = lshr i64 %x, 4
  %2 = and i64 %1, 4095
  ret i64 %2
; CHECK-LABEL: bextr64b:
; CHECK: bextrq
}

define i64 @bextr64b_load(i64* %x) {
  %1 = load i64, i64* %x, align 8
  %2 = lshr i64 %1, 4
  %3 = and i64 %2, 4095
  ret i64 %3
; CHECK-LABEL: bextr64b_load:
; CHECK: bextrq {{.*}}, ({{.*}}), {{.*}}
}

define i32 @non_bextr32(i32 %x) {
entry:
  %shr = lshr i32 %x, 2
  %and = and i32 %shr, 111
  ret i32 %and
; CHECK-LABEL: non_bextr32:
; CHECK: shrl $2
; CHECK: andl $111
}

define i64 @non_bextr64(i64 %x) {
entry:
  %shr = lshr i64 %x, 2
  %and = and i64 %shr, 8589934590
  ret i64 %and
; CHECK-LABEL: non_bextr64:
; CHECK: shrq $2
; CHECK: movabsq $8589934590
; CHECK: andq 
}

define i32 @bzhi32(i32 %x, i32 %y) nounwind readnone {
  %tmp = tail call i32 @llvm.x86.bmi.bzhi.32(i32 %x, i32 %y)
  ret i32 %tmp
; CHECK-LABEL: bzhi32:
; CHECK: bzhil
}

define i32 @bzhi32_load(i32* %x, i32 %y) nounwind readnone {
  %x1 = load i32, i32* %x
  %tmp = tail call i32 @llvm.x86.bmi.bzhi.32(i32 %x1, i32 %y)
  ret i32 %tmp
; CHECK-LABEL: bzhi32_load:
; CHECK: bzhil {{.*}}, ({{.*}}), {{.*}}
}

declare i32 @llvm.x86.bmi.bzhi.32(i32, i32) nounwind readnone

define i64 @bzhi64(i64 %x, i64 %y) nounwind readnone {
  %tmp = tail call i64 @llvm.x86.bmi.bzhi.64(i64 %x, i64 %y)
  ret i64 %tmp
; CHECK-LABEL: bzhi64:
; CHECK: bzhiq
}

declare i64 @llvm.x86.bmi.bzhi.64(i64, i64) nounwind readnone

define i32 @bzhi32b(i32 %x, i8 zeroext %index) #0 {
entry:
  %conv = zext i8 %index to i32
  %shl = shl i32 1, %conv
  %sub = add nsw i32 %shl, -1
  %and = and i32 %sub, %x
  ret i32 %and
; CHECK-LABEL: bzhi32b:
; CHECK: bzhil
}

define i32 @bzhi32b_load(i32* %w, i8 zeroext %index) #0 {
entry:
  %x = load i32, i32* %w
  %conv = zext i8 %index to i32
  %shl = shl i32 1, %conv
  %sub = add nsw i32 %shl, -1
  %and = and i32 %sub, %x
  ret i32 %and
; CHECK-LABEL: bzhi32b_load:
; CHECK: bzhil {{.*}}, ({{.*}}), {{.*}}
}

define i32 @bzhi32c(i32 %x, i8 zeroext %index) #0 {
entry:
  %conv = zext i8 %index to i32
  %shl = shl i32 1, %conv
  %sub = add nsw i32 %shl, -1
  %and = and i32 %x, %sub
  ret i32 %and
; CHECK-LABEL: bzhi32c:
; CHECK: bzhil
}

define i64 @bzhi64b(i64 %x, i8 zeroext %index) #0 {
entry:
  %conv = zext i8 %index to i64
  %shl = shl i64 1, %conv
  %sub = add nsw i64 %shl, -1
  %and = and i64 %x, %sub
  ret i64 %and
; CHECK-LABEL: bzhi64b:
; CHECK: bzhiq
}

define i64 @bzhi64_constant_mask(i64 %x) #0 {
entry:
  %and = and i64 %x, 4611686018427387903
  ret i64 %and
; CHECK-LABEL: bzhi64_constant_mask:
; CHECK: movb    $62, %al
; CHECK: bzhiq   %rax, %r[[ARG1:di|cx]], %rax
}

define i64 @bzhi64_small_constant_mask(i64 %x) #0 {
entry:
  %and = and i64 %x, 2147483647
  ret i64 %and
; CHECK-LABEL: bzhi64_small_constant_mask:
; CHECK: andq  $2147483647, %r[[ARG1]]
}

define i32 @blsi32(i32 %x) nounwind readnone {
  %tmp = sub i32 0, %x
  %tmp2 = and i32 %x, %tmp
  ret i32 %tmp2
; CHECK-LABEL: blsi32:
; CHECK: blsil
}

define i32 @blsi32_load(i32* %x) nounwind readnone {
  %x1 = load i32, i32* %x
  %tmp = sub i32 0, %x1
  %tmp2 = and i32 %x1, %tmp
  ret i32 %tmp2
; CHECK-LABEL: blsi32_load:
; CHECK: blsil ({{.*}})
}

define i64 @blsi64(i64 %x) nounwind readnone {
  %tmp = sub i64 0, %x
  %tmp2 = and i64 %tmp, %x
  ret i64 %tmp2
; CHECK-LABEL: blsi64:
; CHECK: blsiq
}

define i32 @blsmsk32(i32 %x) nounwind readnone {
  %tmp = sub i32 %x, 1
  %tmp2 = xor i32 %x, %tmp
  ret i32 %tmp2
; CHECK-LABEL: blsmsk32:
; CHECK: blsmskl
}

define i32 @blsmsk32_load(i32* %x) nounwind readnone {
  %x1 = load i32, i32* %x
  %tmp = sub i32 %x1, 1
  %tmp2 = xor i32 %x1, %tmp
  ret i32 %tmp2
; CHECK-LABEL: blsmsk32_load:
; CHECK: blsmskl ({{.*}})
}

define i64 @blsmsk64(i64 %x) nounwind readnone {
  %tmp = sub i64 %x, 1
  %tmp2 = xor i64 %tmp, %x
  ret i64 %tmp2
; CHECK-LABEL: blsmsk64:
; CHECK: blsmskq
}

define i32 @blsr32(i32 %x) nounwind readnone {
  %tmp = sub i32 %x, 1
  %tmp2 = and i32 %x, %tmp
  ret i32 %tmp2
; CHECK-LABEL: blsr32:
; CHECK: blsrl
}

define i32 @blsr32_load(i32* %x) nounwind readnone {
  %x1 = load i32, i32* %x
  %tmp = sub i32 %x1, 1
  %tmp2 = and i32 %x1, %tmp
  ret i32 %tmp2
; CHECK-LABEL: blsr32_load:
; CHECK: blsrl ({{.*}})
}

define i64 @blsr64(i64 %x) nounwind readnone {
  %tmp = sub i64 %x, 1
  %tmp2 = and i64 %tmp, %x
  ret i64 %tmp2
; CHECK-LABEL: blsr64:
; CHECK: blsrq
}

define i32 @pdep32(i32 %x, i32 %y) nounwind readnone {
  %tmp = tail call i32 @llvm.x86.bmi.pdep.32(i32 %x, i32 %y)
  ret i32 %tmp
; CHECK-LABEL: pdep32:
; CHECK: pdepl
}

define i32 @pdep32_load(i32 %x, i32* %y) nounwind readnone {
  %y1 = load i32, i32* %y
  %tmp = tail call i32 @llvm.x86.bmi.pdep.32(i32 %x, i32 %y1)
  ret i32 %tmp
; CHECK-LABEL: pdep32_load:
; CHECK: pdepl ({{.*}})
}

declare i32 @llvm.x86.bmi.pdep.32(i32, i32) nounwind readnone

define i64 @pdep64(i64 %x, i64 %y) nounwind readnone {
  %tmp = tail call i64 @llvm.x86.bmi.pdep.64(i64 %x, i64 %y)
  ret i64 %tmp
; CHECK-LABEL: pdep64:
; CHECK: pdepq
}

declare i64 @llvm.x86.bmi.pdep.64(i64, i64) nounwind readnone

define i32 @pext32(i32 %x, i32 %y) nounwind readnone {
  %tmp = tail call i32 @llvm.x86.bmi.pext.32(i32 %x, i32 %y)
  ret i32 %tmp
; CHECK-LABEL: pext32:
; CHECK: pextl
}

define i32 @pext32_load(i32 %x, i32* %y) nounwind readnone {
  %y1 = load i32, i32* %y
  %tmp = tail call i32 @llvm.x86.bmi.pext.32(i32 %x, i32 %y1)
  ret i32 %tmp
; CHECK-LABEL: pext32_load:
; CHECK: pextl ({{.*}})
}

declare i32 @llvm.x86.bmi.pext.32(i32, i32) nounwind readnone

define i64 @pext64(i64 %x, i64 %y) nounwind readnone {
  %tmp = tail call i64 @llvm.x86.bmi.pext.64(i64 %x, i64 %y)
  ret i64 %tmp
; CHECK-LABEL: pext64:
; CHECK: pextq
}

declare i64 @llvm.x86.bmi.pext.64(i64, i64) nounwind readnone

