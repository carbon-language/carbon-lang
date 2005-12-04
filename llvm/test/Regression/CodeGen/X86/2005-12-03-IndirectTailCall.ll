; RUN: llvm-as < %s | llc -march=x86 -enable-x86-fastcc | grep 'jmp \*%ecx'
declare int %putchar(int)

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
        call int %putchar(int 90)       
        %res = tail call fastcc int %fun_ptr( int %arg1, int * %f, int %g)
        ret int %res

if-else:
        ret int %x
}

