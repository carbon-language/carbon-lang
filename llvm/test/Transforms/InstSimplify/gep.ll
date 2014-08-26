; RUN: opt -S -instsimplify < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

%struct.A = type { [7 x i8] }

define %struct.A* @test1(%struct.A* %b, %struct.A* %e) {
  %e_ptr = ptrtoint %struct.A* %e to i64
  %b_ptr = ptrtoint %struct.A* %b to i64
  %sub = sub i64 %e_ptr, %b_ptr
  %sdiv = sdiv exact i64 %sub, 7
  %gep = getelementptr inbounds %struct.A* %b, i64 %sdiv
  ret %struct.A* %gep
; CHECK-LABEL: @test1
; CHECK-NEXT: ret %struct.A* %e
}

define i8* @test2(i8* %b, i8* %e) {
  %e_ptr = ptrtoint i8* %e to i64
  %b_ptr = ptrtoint i8* %b to i64
  %sub = sub i64 %e_ptr, %b_ptr
  %gep = getelementptr inbounds i8* %b, i64 %sub
  ret i8* %gep
; CHECK-LABEL: @test2
; CHECK-NEXT: ret i8* %e
}

define i64* @test3(i64* %b, i64* %e) {
  %e_ptr = ptrtoint i64* %e to i64
  %b_ptr = ptrtoint i64* %b to i64
  %sub = sub i64 %e_ptr, %b_ptr
  %ashr = ashr exact i64 %sub, 3
  %gep = getelementptr inbounds i64* %b, i64 %ashr
  ret i64* %gep
; CHECK-LABEL: @test3
; CHECK-NEXT: ret i64* %e
}
