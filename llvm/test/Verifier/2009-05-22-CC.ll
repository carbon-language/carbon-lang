; RUN: not llvm-as < %s |& grep {Calling convention does not match function signature}
; PR 4239

; Verify that the calling convention on the call instruction matches the
; declared calling convention
declare x86_stdcallcc i32 @re_string_construct(i8* inreg %pstr, i8* inreg %str, i32 inreg %len, i8* %trans, i32 %icase, i8* %dfa);
define void @main() {
entry:
	%0 = call i32 (...)* bitcast (i32 (i8*, i8*, i32, i8*, i32, i8*)* @re_string_construct to i32 (...)*)(i32 inreg 0, i32 inreg 0, i32 inreg 0, i32 0, i32 0, i8* inttoptr (i32 673194176 to i8*))
        ret void
}
