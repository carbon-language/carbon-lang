; RUN: llvm-as < %s | llc

%a_str = internal constant [8 x sbyte] c"a = %d\0A\00"
%b_str = internal constant [8 x sbyte] c"b = %d\0A\00"
;; binary ops: arith
%add_str = internal constant [12 x sbyte] c"a + b = %d\0A\00"
%sub_str = internal constant [12 x sbyte] c"a - b = %d\0A\00"
%mul_str = internal constant [12 x sbyte] c"a * b = %d\0A\00"
%div_str = internal constant [12 x sbyte] c"b / a = %d\0A\00"
%rem_str = internal constant [13 x sbyte] c"b \% a = %d\0A\00"
;; binary ops: setcc
%lt_str  = internal constant [12 x sbyte] c"a < b = %d\0A\00"
%le_str  = internal constant [13 x sbyte] c"a <= b = %d\0A\00"
%gt_str  = internal constant [12 x sbyte] c"a > b = %d\0A\00"
%ge_str  = internal constant [13 x sbyte] c"a >= b = %d\0A\00"
%eq_str  = internal constant [13 x sbyte] c"a == b = %d\0A\00"
%ne_str  = internal constant [13 x sbyte] c"a != b = %d\0A\00"
;; logical
%and_str = internal constant [12 x sbyte] c"a & b = %d\0A\00"
%or_str  = internal constant [12 x sbyte] c"a | b = %d\0A\00"
%xor_str = internal constant [12 x sbyte] c"a ^ b = %d\0A\00"
%shl_str = internal constant [13 x sbyte] c"b << a = %d\0A\00"
%shr_str = internal constant [13 x sbyte] c"b >> a = %d\0A\00"

declare int %printf(sbyte*, ...)
%A = global int 2
%B = global int 5

int %main() {  
  ;; main vars
  %a = load int* %A
  %b = load int* %B

  %a_s = getelementptr [8 x sbyte]* %a_str, long 0, long 0
  %b_s = getelementptr [8 x sbyte]* %b_str, long 0, long 0
  
  call int (sbyte*, ...)* %printf(sbyte* %a_s, int %a)
  call int (sbyte*, ...)* %printf(sbyte* %b_s, int %b)

  ;; arithmetic
  %add_r  = add int %a, %b
  %sub_r  = sub int %a, %b
  %mul_r  = mul int %a, %b
  %div_r  = div int %b, %a
  %rem_r  = rem int %b, %a

  %add_s = getelementptr [12 x sbyte]* %add_str, long 0, long 0
  %sub_s = getelementptr [12 x sbyte]* %sub_str, long 0, long 0
  %mul_s = getelementptr [12 x sbyte]* %mul_str, long 0, long 0
  %div_s = getelementptr [12 x sbyte]* %div_str, long 0, long 0
  %rem_s = getelementptr [13 x sbyte]* %rem_str, long 0, long 0

  call int (sbyte*, ...)* %printf(sbyte* %add_s, int %add_r)
  call int (sbyte*, ...)* %printf(sbyte* %sub_s, int %sub_r)
  call int (sbyte*, ...)* %printf(sbyte* %mul_s, int %mul_r)
  call int (sbyte*, ...)* %printf(sbyte* %div_s, int %div_r)
  call int (sbyte*, ...)* %printf(sbyte* %rem_s, int %rem_r)

  ;; setcc
  %lt_r = setlt int %a, %b
  %le_r = setle int %a, %b
  %gt_r = setgt int %a, %b 
  %ge_r = setge int %a, %b
  %eq_r = seteq int %a, %b
  %ne_r = setne int %a, %b
  
  %lt_s = getelementptr [12 x sbyte]* %lt_str, long 0, long 0
  %le_s = getelementptr [13 x sbyte]* %le_str, long 0, long 0
  %gt_s = getelementptr [12 x sbyte]* %gt_str, long 0, long 0
  %ge_s = getelementptr [13 x sbyte]* %ge_str, long 0, long 0
  %eq_s = getelementptr [13 x sbyte]* %eq_str, long 0, long 0
  %ne_s = getelementptr [13 x sbyte]* %ne_str, long 0, long 0

  call int (sbyte*, ...)* %printf(sbyte* %lt_s, bool %lt_r)
  call int (sbyte*, ...)* %printf(sbyte* %le_s, bool %le_r)
  call int (sbyte*, ...)* %printf(sbyte* %gt_s, bool %gt_r)
  call int (sbyte*, ...)* %printf(sbyte* %ge_s, bool %ge_r)
  call int (sbyte*, ...)* %printf(sbyte* %eq_s, bool %eq_r)
  call int (sbyte*, ...)* %printf(sbyte* %ne_s, bool %ne_r)

  ;; logical
  %and_r = and int %a, %b
  %or_r  = or  int %a, %b
  %xor_r = xor int %a, %b
  %u = cast int %a to ubyte
  %shl_r = shl int %b, ubyte %u
  %shr_r = shr int %b, ubyte %u
  
  %and_s = getelementptr [12 x sbyte]* %and_str, long 0, long 0
  %or_s  = getelementptr [12 x sbyte]* %or_str,  long 0, long 0
  %xor_s = getelementptr [12 x sbyte]* %xor_str, long 0, long 0
  %shl_s = getelementptr [13 x sbyte]* %shl_str, long 0, long 0
  %shr_s = getelementptr [13 x sbyte]* %shr_str, long 0, long 0

  call int (sbyte*, ...)* %printf(sbyte* %and_s, int %and_r)
  call int (sbyte*, ...)* %printf(sbyte* %or_s,  int %or_r)
  call int (sbyte*, ...)* %printf(sbyte* %xor_s, int %xor_r)
  call int (sbyte*, ...)* %printf(sbyte* %shl_s, int %shl_r)
  call int (sbyte*, ...)* %printf(sbyte* %shr_s, int %shr_r)

  ret int 0
}
