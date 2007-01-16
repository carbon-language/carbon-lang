; RUN: llvm-as %s -o - | llvm-dis > %t.ll
; RUN: diff %t.ll %s.out

; test 63 bits
;
%b = constant i63 add(i63 9223372036854775807, i63 1)
%c = constant i63 add(i63 9223372036854775807, i63 9223372036854775807)
%d = constant i63 add(i63 9223372036854775800, i63 8)
%e = constant i63 sub(i63 0 , i63 1)
%f = constant i63 sub(i63 0 , i63 9223372036854775807)
%g = constant i63 sub(i63 2 , i63 9223372036854775807)

%h = constant i63 shl(i63 1 , i8 63)
%i = constant i63 shl(i63 1 , i8 62)
%j = constant i63 lshr(i63 9223372036854775807 , i8 62)
%k = constant i63 lshr(i63 9223372036854775807 , i8 63)
%l = constant i63 ashr(i63 9223372036854775807 , i8 62)
%m = constant i63 ashr(i63 9223372036854775807 , i8 63)

%n = constant i63 mul(i63 9223372036854775807, i63 2) 
%o = constant i63 trunc( i64 9223372036854775808 to i63 )
%p = constant i63 trunc( i64 9223372036854775807  to i63 )
