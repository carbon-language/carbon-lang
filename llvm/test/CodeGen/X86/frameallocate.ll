; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

declare i8* @llvm.frameallocate(i32)
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.framerecover(i8*, i8*)
declare i32 @printf(i8*, ...)

@str = internal constant [10 x i8] c"asdf: %d\0A\00"

define void @print_framealloc_from_fp(i8* %fp) {
  %alloc = call i8* @llvm.framerecover(i8* bitcast (void(i32*, i32*)* @alloc_func to i8*), i8* %fp)
  %alloc_i32 = bitcast i8* %alloc to i32*
  %r = load i32, i32* %alloc_i32
  call i32 (i8*, ...)* @printf(i8* getelementptr ([10 x i8]* @str, i32 0, i32 0), i32 %r)
  ret void
}

; CHECK-LABEL: print_framealloc_from_fp:
; CHECK: movabsq $.Lframeallocation_alloc_func, %[[offs:[a-z]+]]
; CHECK: movl (%rcx,%[[offs]]), %edx
; CHECK: leaq {{.*}}(%rip), %rcx
; CHECK: callq printf
; CHECK: retq

define void @alloc_func(i32* %s, i32* %d) {
  %alloc = call i8* @llvm.frameallocate(i32 16)
  %alloc_i32 = bitcast i8* %alloc to i32*
  store i32 42, i32* %alloc_i32
  %fp = call i8* @llvm.frameaddress(i32 0)
  call void @print_framealloc_from_fp(i8* %fp)
  ret void
}

; CHECK-LABEL: alloc_func:
; CHECK: subq    $48, %rsp
; CHECK: .seh_stackalloc 48
; CHECK: leaq    48(%rsp), %rbp
; CHECK: .seh_setframe 5, 48
; CHECK: .Lframeallocation_alloc_func = -[[offs:[0-9]+]]
; CHECK: movl $42, -[[offs]](%rbp)
; CHECK: leaq    -48(%rbp), %rcx
; CHECK: callq print_framealloc_from_fp
; CHECK: retq
