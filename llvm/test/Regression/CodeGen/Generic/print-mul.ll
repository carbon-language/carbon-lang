; RUN: llvm-as < %s | llc

%a_str = internal constant [8 x sbyte] c"a = %d\0A\00"
%b_str = internal constant [8 x sbyte] c"b = %d\0A\00"

;; mul
%a_mul_str = internal constant [13 x sbyte] c"a * %d = %d\0A\00"

declare int %printf(sbyte*, ...)
%A = global int 2
%B = global int 5

int %main() {  
entry:
  %a = load int* %A
  %b = load int* %B
  %a_s = getelementptr [8 x sbyte]* %a_str, long 0, long 0
  %b_s = getelementptr [8 x sbyte]* %b_str, long 0, long 0
  %a_mul_s = getelementptr [13 x sbyte]* %a_mul_str, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %a_s, int %a)
  call int (sbyte*, ...)* %printf(sbyte* %b_s, int %b)
  br label %shl_test

shl_test:
  ;; test mul by 0-255
  %s = phi int [ 0, %entry ], [ %s_inc, %shl_test ]
  %result = mul int %a, %s
  call int (sbyte*, ...)* %printf(sbyte* %a_mul_s, int %s, int %result)
  %s_inc = add int %s, 1 
  %done = seteq int %s, 256
  br bool %done, label %fini, label %shl_test

fini:
  ret int 0
}
