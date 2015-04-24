; RUN: llc -march=hexagon -mcpu=hexagonv5 <%s | \
; RUN:   FileCheck %s --check-prefix=CHECK-ONE
; RUN: llc -march=hexagon -mcpu=hexagonv5 <%s | \
; RUN:   FileCheck %s --check-prefix=CHECK-TWO
; RUN: llc -march=hexagon -mcpu=hexagonv5 <%s | \
; RUN:   FileCheck %s --check-prefix=CHECK-THREE

%struct.test_struct = type { i32, i8, i64 }
%struct.test_struct_long = type { i8, i64 }

@mystruct = external global %struct.test_struct*, align 4

; CHECK-ONE:    memw(r29+#48) = r2
; CHECK-TWO:    memw(r29+#52) = r2
; CHECK-THREE:  memw(r29+#56) = r2
; Function Attrs: nounwind
define void @foo(%struct.test_struct* noalias sret %agg.result, i32 %a, i8 zeroext %c, %struct.test_struct* byval %s, %struct.test_struct_long* byval %t) #0 {
entry:
  %a.addr = alloca i32, align 4
  %c.addr = alloca i8, align 1
  %z = alloca i32, align 4
  %ret = alloca %struct.test_struct, align 8
  store i32 %a, i32* %a.addr, align 4
  store i8 %c, i8* %c.addr, align 1
  %0 = bitcast i32* %z to i8*
  call void @llvm.lifetime.start(i64 4, i8* %0) #1
  store i32 45, i32* %z, align 4
  %1 = bitcast %struct.test_struct* %ret to i8*
  call void @llvm.lifetime.start(i64 16, i8* %1) #1
  %2 = load i32, i32* %z, align 4
  %3 = load %struct.test_struct*, %struct.test_struct** @mystruct, align 4
  %4 = load %struct.test_struct*, %struct.test_struct** @mystruct, align 4
  %5 = load i8, i8* %c.addr, align 1
  %6 = load i32, i32* %a.addr, align 4
  %conv = sext i32 %6 to i64
  %add = add nsw i64 %conv, 1
  %7 = load i32, i32* %a.addr, align 4
  %add1 = add nsw i32 %7, 2
  %8 = load i32, i32* %a.addr, align 4
  %conv2 = sext i32 %8 to i64
  %add3 = add nsw i64 %conv2, 3
  %9 = load i8, i8* %c.addr, align 1
  %10 = load i8, i8* %c.addr, align 1
  %11 = load i8, i8* %c.addr, align 1
  %12 = load i32, i32* %z, align 4
  call void @bar(%struct.test_struct* sret %ret, i32 %2, %struct.test_struct* byval %3, %struct.test_struct* byval %4, i8 zeroext %5, i64 %add, i32 %add1, i64 %add3, i8 zeroext %9, i8 zeroext %10, i8 zeroext %11, i32 %12)
  %x = getelementptr inbounds %struct.test_struct, %struct.test_struct* %ret, i32 0, i32 0
  store i32 20, i32* %x, align 4
  %13 = bitcast %struct.test_struct* %agg.result to i8*
  %14 = bitcast %struct.test_struct* %ret to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %13, i8* %14, i32 16, i32 8, i1 false)
  %15 = bitcast %struct.test_struct* %ret to i8*
  call void @llvm.lifetime.end(i64 16, i8* %15) #1
  %16 = bitcast i32* %z to i8*
  call void @llvm.lifetime.end(i64 4, i8* %16) #1
  ret void
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

declare void @bar(%struct.test_struct* sret, i32, %struct.test_struct* byval, %struct.test_struct* byval, i8 zeroext, i64, i32, i64, i8 zeroext, i8 zeroext, i8 zeroext, i32) #2

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv4" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv4" "unsafe-fp-math"="false" "use-soft-float"="false" }

