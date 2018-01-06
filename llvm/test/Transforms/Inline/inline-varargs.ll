; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline,function(instcombine))' -S | FileCheck %s

declare void @ext_method(i8*, i32)
declare void @vararg_fn(i8*, ...)

define linkonce_odr void @thunk(i8* %this, ...) {
  %this_adj = getelementptr i8, i8* %this, i32 4
  musttail call void (i8*, ...) bitcast (void (i8*, i32)* @ext_method to void (i8*, ...)*)(i8* %this_adj, ...)
  ret void
}

define void @thunk_caller(i8* %p) {
  call void (i8*, ...) @thunk(i8* %p, i32 42)
  ret void
}
; CHECK-LABEL: define void @thunk_caller(i8* %p)
; CHECK: call void (i8*, ...) bitcast (void (i8*, i32)* @ext_method to void (i8*, ...)*)(i8* %this_adj.i, i32 42)

define void @test_callee_2(i8* %this, ...) {
  %this_adj = getelementptr i8, i8* %this, i32 4
  musttail call void (i8*, ...) @vararg_fn(i8* %this_adj, ...)
  ret void
}

define void @test_caller_2(i8* %p) {
  call void (i8*, ...) @test_callee_2(i8* %p)
  ret void
}
; CHECK-LABEL: define void @test_caller_2(i8* %p)
; CHECK: call void (i8*, ...) @vararg_fn(i8* %this_adj.i)


define internal i32 @varg_accessed(...) {
entry:
  %vargs = alloca i8*, align 8
  %vargs.ptr = bitcast i8** %vargs to i8*
  call void @llvm.va_start(i8* %vargs.ptr)
  %va1 = va_arg i8** %vargs, i32
  call void @llvm.va_end(i8* %vargs.ptr)
  ret i32 %va1
}

define i32 @call_vargs() {
  %res = call i32 (...) @varg_accessed(i32 10)
  ret i32 %res
}
; CHECK-LABEL: @call_vargs
; CHECK: %res = call i32 (...) @varg_accessed(i32 10)

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
