; RUN: llvm-as < %s | llc

%a_str = internal constant [8 x sbyte] c"a = %d\0A\00"
%b_str = internal constant [8 x sbyte] c"b = %d\0A\00"

;; shl
%a_shl_str = internal constant [14 x sbyte] c"a << %d = %d\0A\00"

declare int %printf(sbyte*, ...)
%A = global int 2
%B = global int 5

int %main() {  
entry:
  %a = load int* %A
  %b = load int* %B
  %a_s = getelementptr [8 x sbyte]* %a_str, long 0, long 0
  %b_s = getelementptr [8 x sbyte]* %b_str, long 0, long 0
  %a_shl_s = getelementptr [14 x sbyte]* %a_shl_str, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %a_s, int %a)
  call int (sbyte*, ...)* %printf(sbyte* %b_s, int %b)
  br label %shl_test

shl_test:
  ;; test left shifts 0-31
  %s = phi ubyte [ 0, %entry ], [ %s_inc, %shl_test ]
  %result = shl int %a, ubyte %s
  call int (sbyte*, ...)* %printf(sbyte* %a_shl_s, ubyte %s, int %result)
  %s_inc = add ubyte %s, 1 
  %done = seteq ubyte %s, 32
  br bool %done, label %fini, label %shl_test

fini:
  ret int 0
}
