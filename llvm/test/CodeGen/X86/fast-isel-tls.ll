; RUN: llc < %s -march=x86 -relocation-model=pic -mtriple=i686-unknown-linux-gnu -fast-isel | grep __tls_get_addr
; PR3654

@v = thread_local global i32 0
define i32 @f() nounwind {
entry:
          %t = load i32* @v
          %s = add i32 %t, 1
          ret i32 %s
}
