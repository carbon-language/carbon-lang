; RUN: opt < %s -instcombine -globaldce -S | FileCheck %s
; PR1201
define i32 @main(i32 %argc, i8** %argv) {
        %c_19 = alloca i8*              ; <i8**> [#uses=2]
        %malloc_206 = malloc i8, i32 10         ; <i8*> [#uses=1]
; CHECK-NOT: malloc
        store i8* %malloc_206, i8** %c_19
        %tmp_207 = load i8** %c_19              ; <i8*> [#uses=1]
        free i8* %tmp_207
; CHECK-NOT: free
        ret i32 0
; CHECK: ret i32 0
}
