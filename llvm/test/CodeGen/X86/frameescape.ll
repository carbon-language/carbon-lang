; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

declare void @llvm.frameescape(...)
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.framerecover(i8*, i8*, i32)
declare i32 @printf(i8*, ...)

@str = internal constant [10 x i8] c"asdf: %d\0A\00"

define void @print_framealloc_from_fp(i8* %fp) {
  %a.i8 = call i8* @llvm.framerecover(i8* bitcast (void()* @alloc_func to i8*), i8* %fp, i32 0)
  %a = bitcast i8* %a.i8 to i32*
  %a.val = load i32, i32* %a
  call i32 (i8*, ...)* @printf(i8* getelementptr ([10 x i8], [10 x i8]* @str, i32 0, i32 0), i32 %a.val)
  %b.i8 = call i8* @llvm.framerecover(i8* bitcast (void()* @alloc_func to i8*), i8* %fp, i32 1)
  %b = bitcast i8* %b.i8 to i32*
  %b.val = load i32, i32* %b
  call i32 (i8*, ...)* @printf(i8* getelementptr ([10 x i8], [10 x i8]* @str, i32 0, i32 0), i32 %b.val)
  store i32 42, i32* %b
  ret void
}

; CHECK-LABEL: print_framealloc_from_fp:
; CHECK: movq %rcx, %[[parent_fp:[a-z]+]]
; CHECK: movl .Lalloc_func$frame_escape_0(%[[parent_fp]]), %edx
; CHECK: leaq {{.*}}(%rip), %[[str:[a-z]+]]
; CHECK: movq %[[str]], %rcx
; CHECK: callq printf
; CHECK: movl .Lalloc_func$frame_escape_1(%[[parent_fp]]), %edx
; CHECK: movq %[[str]], %rcx
; CHECK: callq printf
; CHECK: movl    $42, .Lalloc_func$frame_escape_1(%[[parent_fp]])
; CHECK: retq

define void @alloc_func() {
  %a = alloca i32
  %b = alloca i32
  call void (...)* @llvm.frameescape(i32* %a, i32* %b)
  store i32 42, i32* %a
  store i32 13, i32* %b
  %fp = call i8* @llvm.frameaddress(i32 0)
  call void @print_framealloc_from_fp(i8* %fp)
  ret void
}

; CHECK-LABEL: alloc_func:
; CHECK: subq    $48, %rsp
; CHECK: .seh_stackalloc 48
; CHECK: leaq    48(%rsp), %rbp
; CHECK: .seh_setframe 5, 48
; CHECK: .Lalloc_func$frame_escape_0 = 44
; CHECK: .Lalloc_func$frame_escape_1 = 40
; CHECK: movl $42, -4(%rbp)
; CHECK: movl $13, -8(%rbp)
; CHECK: leaq    -48(%rbp), %rcx
; CHECK: callq print_framealloc_from_fp
; CHECK: retq

; Helper to make this a complete program so it can be compiled and tested.
define i32 @main() {
  call void @alloc_func()
  ret i32 0
}
