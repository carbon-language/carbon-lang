; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 15 bits
;
%b = constant i15 add(i15 32767, i15 1)
%c = constant i15 add(i15 32767, i15 32767)
%d = constant i15 add(i15 32760, i15 8)
%e = constant i15 sub(i15 0 , i15 1)
%f = constant i15 sub(i15 0 , i15 32767)
%g = constant i15 sub(i15 2 , i15 32767)

%h = constant i15 shl(i15 1 , i8 15)
%i = constant i15 shl(i15 1 , i8 14)
%j = constant i15 lshr(i15 32767 , i8 14)
%k = constant i15 lshr(i15 32767 , i8 15)
%l = constant i15 ashr(i15 32767 , i8 14)
%m = constant i15 ashr(i15 32767 , i8 15)

%n = constant i15 mul(i15 32767, i15 2)
%o = constant i15 trunc( i16 32768  to i15 )
%p = constant i15 trunc( i16 32767  to i15 )
 
