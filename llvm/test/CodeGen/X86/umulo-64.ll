; RUN: llc < %s -mtriple=i386-apple-darwin

%0 = type { i64, i1 }

define i32 @f0(i64 %a, i64 %b) nounwind ssp {
  %1 = alloca i64, align 4
  %2 = alloca i64, align 4
  store i64 %a, i64* %1, align 8
  store i64 %b, i64* %2, align 8
  %3 = load i64* %1, align 8
  %4 = load i64* %2, align 8
  %5 = call %0 @llvm.smul.with.overflow.i64(i64 %3, i64 %4)
  %6 = extractvalue %0 %5, 0
  %7 = extractvalue %0 %5, 1
  br i1 %7, label %8, label %9

; <label>:8                                       ; preds = %0
  call void @llvm.trap()
  unreachable

; <label>:9                                       ; preds = %0
  %10 = trunc i64 %6 to i32
  ret i32 %10
}

declare %0 @llvm.smul.with.overflow.i64(i64, i64) nounwind readnone

declare void @llvm.trap() nounwind
