; RUN: opt < %s -instcombine -S | FileCheck %s
; PR1201
define i32 @main(i32 %argc, i8** %argv) {
        %c_19 = alloca i8*
        %malloc_206 = malloc i8, i32 10
; CHECK-NOT: malloc
        store i8* %malloc_206, i8** %c_19
        %tmp_207 = load i8** %c_19
        free i8* %tmp_207
; CHECK-NOT: free
        ret i32 0
; CHECK: ret i32 0
}

declare i8* @malloc(i32)
declare void @free(i8*)

define i1 @foo() {
; CHECK: @foo
; CHECK-NEXT: ret i1 false
  %m = call i8* @malloc(i32 1)
  %z = icmp eq i8* %m, null
  call void @free(i8* %m)
  ret i1 %z
}
