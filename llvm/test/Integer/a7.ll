; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 7 bits
;
%b = constant i7 add(i7 127, i7 1)
%c = constant i7 add(i7 127, i7 127)
%d = constant i7 add(i7 120, i7 8)
%e = constant i7 sub(i7 0 , i7 1)
%f = constant i7 sub(i7 0 , i7 127)
%g = constant i7 sub(i7 2 , i7 127)

%h = constant i7 shl(i7 1 , i8 7)
%i = constant i7 shl(i7 1 , i8 6)
%j = constant i7 lshr(i7 127 , i8 6)
%k = constant i7 lshr(i7 127 , i8 7)
%l = constant i7 ashr(i7 127 , i8 6)
%m = constant i7 ashr(i7 127 , i8 7)

%n = constant i7 mul(i7 127, i7 2)
%o = constant i7 trunc( i8 128  to i7 )
%p = constant i7 trunc( i8 255  to i7 )
 
