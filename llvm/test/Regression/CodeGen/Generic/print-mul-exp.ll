; RUN: llvm-as < %s | llc

%a_str = internal constant [8 x sbyte] c"a = %d\0A\00"
%a_mul_str = internal constant [13 x sbyte] c"a * %d = %d\0A\00"
%A = global int 2
declare int %printf(sbyte*, ...)

int %main() {  
  %a = load int* %A
  %a_s = getelementptr [8 x sbyte]* %a_str, long 0, long 0
  %a_mul_s = getelementptr [13 x sbyte]* %a_mul_str, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %a_s, int %a)

  %r_0 = mul int %a, 0
  %r_1 = mul int %a, 1
  %r_2 = mul int %a, 2
  %r_3 = mul int %a, 3
  %r_4 = mul int %a, 4
  %r_5 = mul int %a, 5
  %r_6 = mul int %a, 6
  %r_7 = mul int %a, 7
  %r_8 = mul int %a, 8
  %r_9 = mul int %a, 9
  %r_10 = mul int %a, 10
  %r_11 = mul int %a, 11
  %r_12 = mul int %a, 12
  %r_13 = mul int %a, 13
  %r_14 = mul int %a, 14
  %r_15 = mul int %a, 15
  %r_16 = mul int %a, 16
  %r_17 = mul int %a, 17
  %r_18 = mul int %a, 18
  %r_19 = mul int %a, 19

  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 0, int %r_0)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 1, int %r_1)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 2, int %r_2)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 3, int %r_3)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 4, int %r_4)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 5, int %r_5)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 6, int %r_6)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 7, int %r_7)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 8, int %r_8)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 9, int %r_9)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 10, int %r_10)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 11, int %r_11)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 12, int %r_12)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 13, int %r_13)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 14, int %r_14)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 15, int %r_15)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 16, int %r_16)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 17, int %r_17)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 18, int %r_18)
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int 19, int %r_19)

  ret int 0
}
