; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 31 bits
;
%b = constant i31 add(i31 2147483647, i31 1)
%c = constant i31 add(i31 2147483647, i31 2147483647)
%d = constant i31 add(i31 2147483640, i31 8)
%e = constant i31 sub(i31 0 , i31 1)
%f = constant i31 sub(i31 0 , i31 2147483647)
%g = constant i31 sub(i31 2 , i31 2147483647)

%h = constant i31 shl(i31 1 , i8 31)
%i = constant i31 shl(i31 1 , i8 30)
%j = constant i31 lshr(i31 2147483647 , i8 30)
%k = constant i31 lshr(i31 2147483647 , i8 31)
%l = constant i31 ashr(i31 2147483647 , i8 30)
%m = constant i31 ashr(i31 2147483647 , i8 31)

%n = constant i31 mul(i31 2147483647, i31 2)
%o = constant i31 trunc( i32 2147483648  to i31 )
%p = constant i31 trunc( i32 2147483647  to i31 ) 
