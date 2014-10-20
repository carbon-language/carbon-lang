; RUN: llc -verify-machineinstrs -mtriple=i386--netbsd < %s | FileCheck %s
; Regression test for http://reviews.llvm.org/D5701

; ModuleID = 'xxhash.i'
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386--netbsd"

; CHECK-LABEL: fn1
; CHECK:       shldl {{.*#+}} 4-byte Folded Spill
; CHECK:       orl   {{.*#+}} 4-byte Folded Reload
; CHECK:       shldl {{.*#+}} 4-byte Folded Spill
; CHECK:       orl   {{.*#+}} 4-byte Folded Reload
; CHECK:       addl  {{.*#+}} 4-byte Folded Reload
; CHECK:       imull {{.*#+}} 4-byte Folded Reload
; CHECK:       orl   {{.*#+}} 4-byte Folded Reload
; CHECK:       retl

%struct.XXH_state64_t = type { i32, i32, i64, i64, i64 }

@a = common global i32 0, align 4
@b = common global i64 0, align 8

; Function Attrs: nounwind uwtable
define i64 @fn1() #0 {
entry:
  %0 = load i32* @a, align 4, !tbaa !1
  %1 = inttoptr i32 %0 to %struct.XXH_state64_t*
  %total_len = getelementptr inbounds %struct.XXH_state64_t* %1, i32 0, i32 0
  %2 = load i32* %total_len, align 4, !tbaa !5
  %tobool = icmp eq i32 %2, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %v3 = getelementptr inbounds %struct.XXH_state64_t* %1, i32 0, i32 3
  %3 = load i64* %v3, align 4, !tbaa !8
  %v4 = getelementptr inbounds %struct.XXH_state64_t* %1, i32 0, i32 4
  %4 = load i64* %v4, align 4, !tbaa !9
  %v2 = getelementptr inbounds %struct.XXH_state64_t* %1, i32 0, i32 2
  %5 = load i64* %v2, align 4, !tbaa !10
  %shl = shl i64 %5, 1
  %or = or i64 %shl, %5
  %shl2 = shl i64 %3, 2
  %shr = lshr i64 %3, 1
  %or3 = or i64 %shl2, %shr
  %add = add i64 %or, %or3
  %mul = mul i64 %4, -4417276706812531889
  %shl4 = mul i64 %4, -8834553413625063778
  %shr5 = ashr i64 %mul, 3
  %or6 = or i64 %shr5, %shl4
  %mul7 = mul nsw i64 %or6, 1400714785074694791
  %xor = xor i64 %add, %mul7
  store i64 %xor, i64* @b, align 8, !tbaa !11
  %mul8 = mul nsw i64 %xor, 1400714785074694791
  br label %if.end

if.else:                                          ; preds = %entry
  %6 = load i64* @b, align 8, !tbaa !11
  %xor10 = xor i64 %6, -4417276706812531889
  %mul11 = mul nsw i64 %xor10, 400714785074694791
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %storemerge.in = phi i64 [ %mul11, %if.else ], [ %mul8, %if.then ]
  %storemerge = add i64 %storemerge.in, -8796714831421723037
  store i64 %storemerge, i64* @b, align 8, !tbaa !11
  ret i64 undef
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.6 (trunk 219587)"}
!1 = metadata !{metadata !2, metadata !2, i64 0}
!2 = metadata !{metadata !"int", metadata !3, i64 0}
!3 = metadata !{metadata !"omnipotent char", metadata !4, i64 0}
!4 = metadata !{metadata !"Simple C/C++ TBAA"}
!5 = metadata !{metadata !6, metadata !2, i64 0}
!6 = metadata !{metadata !"XXH_state64_t", metadata !2, i64 0, metadata !2, i64 4, metadata !7, i64 8, metadata !7, i64 16, metadata !7, i64 24}
!7 = metadata !{metadata !"long long", metadata !3, i64 0}
!8 = metadata !{metadata !6, metadata !7, i64 16}
!9 = metadata !{metadata !6, metadata !7, i64 24}
!10 = metadata !{metadata !6, metadata !7, i64 8}
!11 = metadata !{metadata !7, metadata !7, i64 0}
