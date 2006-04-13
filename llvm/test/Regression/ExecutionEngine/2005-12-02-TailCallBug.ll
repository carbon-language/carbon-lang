; PR672
; RUN: llvm-as < %s | lli

; This doesn't work on darwin/x86, xfail until PR736 is resolved.
; XFAIL: i686-apple-darwin

int %main(){ 
 %f   = cast int (int, int*, int)* %check_tail to int*
 %res = tail call fastcc int %check_tail( int 10, int* %f,int 10)
 ret int %res
}
fastcc int %check_tail(int %x, int* %f, int %g) {
	%tmp1 = setgt int %x, 0
	br bool %tmp1, label %if-then, label %if-else

if-then:
	%fun_ptr = cast int* %f to int(int, int*, int)*	
	%arg1    = add int %x, -1		
	%res = tail call fastcc int %fun_ptr( int %arg1, int * %f, int %g)
	ret int %res

if-else:
        ret int %x
}
