; RUN: llvm-as < %s | llc

%a_str = internal constant [8 x sbyte] c"a = %f\0A\00"
%b_str = internal constant [8 x sbyte] c"b = %f\0A\00"
;; binary ops: arith
%add_str = internal constant [12 x sbyte] c"a + b = %f\0A\00"
%sub_str = internal constant [12 x sbyte] c"a - b = %f\0A\00"
%mul_str = internal constant [12 x sbyte] c"a * b = %f\0A\00"
%div_str = internal constant [12 x sbyte] c"b / a = %f\0A\00"
%rem_str = internal constant [13 x sbyte] c"b %% a = %f\0A\00"
;; binary ops: setcc
%lt_str  = internal constant [12 x sbyte] c"a < b = %d\0A\00"
%le_str  = internal constant [13 x sbyte] c"a <= b = %d\0A\00"
%gt_str  = internal constant [12 x sbyte] c"a > b = %d\0A\00"
%ge_str  = internal constant [13 x sbyte] c"a >= b = %d\0A\00"
%eq_str  = internal constant [13 x sbyte] c"a == b = %d\0A\00"
%ne_str  = internal constant [13 x sbyte] c"a != b = %d\0A\00"

declare int %printf(sbyte*, ...)
%A = global double 2.0
%B = global double 5.0

int %main() {  
  ;; main vars
  %a = load double* %A
  %b = load double* %B

  %a_s = getelementptr [8 x sbyte]* %a_str, long 0, long 0
  %b_s = getelementptr [8 x sbyte]* %b_str, long 0, long 0
  
  call int (sbyte*, ...)* %printf(sbyte* %a_s, double %a)
  call int (sbyte*, ...)* %printf(sbyte* %b_s, double %b)

  ;; arithmetic
  %add_r  = add double %a, %b
  %sub_r  = sub double %a, %b
  %mul_r  = mul double %a, %b
  %div_r  = div double %b, %a
  %rem_r  = rem double %b, %a

  %add_s = getelementptr [12 x sbyte]* %add_str, long 0, long 0
  %sub_s = getelementptr [12 x sbyte]* %sub_str, long 0, long 0
  %mul_s = getelementptr [12 x sbyte]* %mul_str, long 0, long 0
  %div_s = getelementptr [12 x sbyte]* %div_str, long 0, long 0
  %rem_s = getelementptr [13 x sbyte]* %rem_str, long 0, long 0

  call int (sbyte*, ...)* %printf(sbyte* %add_s, double %add_r)
  call int (sbyte*, ...)* %printf(sbyte* %sub_s, double %sub_r)
  call int (sbyte*, ...)* %printf(sbyte* %mul_s, double %mul_r)
  call int (sbyte*, ...)* %printf(sbyte* %div_s, double %div_r)
  call int (sbyte*, ...)* %printf(sbyte* %rem_s, double %rem_r)

  ;; setcc
  %lt_r = setlt double %a, %b
  %le_r = setle double %a, %b
  %gt_r = setgt double %a, %b 
  %ge_r = setge double %a, %b
  %eq_r = seteq double %a, %b
  %ne_r = setne double %a, %b
  
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

  ret int 0
}
