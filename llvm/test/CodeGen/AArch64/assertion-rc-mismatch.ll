; RUN: llc < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s
; Test case related to <rdar://problem/15633429>.

; CHECK-LABEL: small
define i64 @small(i64 %encodedBase) {
cmp:
  %lnot.i.i = icmp eq i64 %encodedBase, 0
  br i1 %lnot.i.i, label %if, label %else
if:
  %tmp1 = call i8* @llvm.returnaddress(i32 0)
  br label %end
else:
  %tmp3 = call i8* @llvm.returnaddress(i32 0)
  %ptr = getelementptr inbounds i8* %tmp3, i64 -16
  %ld = load i8* %ptr, align 4
  %tmp2 = inttoptr i8 %ld to i8*
  br label %end
end:
  %tmp = phi i8* [ %tmp1, %if ], [ %tmp2, %else ]
  %coerce.val.pi56 = ptrtoint i8* %tmp to i64
  ret i64 %coerce.val.pi56
}

declare i8* @llvm.returnaddress(i32)
