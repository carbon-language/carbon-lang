; RUN: llvm-as < %s | llc

%a_fstr = internal constant [8 x sbyte] c"a = %f\0A\00"
%a_lstr = internal constant [10 x sbyte] c"a = %lld\0A\00"
%a_dstr = internal constant [8 x sbyte] c"a = %d\0A\00"

%b_dstr = internal constant [8 x sbyte] c"b = %d\0A\00"
%b_fstr = internal constant [8 x sbyte] c"b = %f\0A\00"

declare int %printf(sbyte*, ...)
%A = global double 2.0
%B = global int 2

int %main() {  
  ;; A
  %a = load double* %A
  %a_fs = getelementptr [8 x sbyte]* %a_fstr, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %a_fs, double %a)

  ;; cast double to long
  %a_d2l = cast double %a to long
  %a_ls = getelementptr [10 x sbyte]* %a_lstr, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %a_ls, long %a_d2l)

  ;; cast double to int
  %a_d2i = cast double %a to int
  %a_ds = getelementptr [8 x sbyte]* %a_dstr, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %a_ds, int %a_d2i)

  ;; cast double to sbyte
  %a_d2sb = cast double %a to sbyte
  call int (sbyte*, ...)* %printf(sbyte* %a_ds, sbyte %a_d2sb)

  ;; cast int to sbyte
  %a_d2i2sb = cast int %a_d2i to sbyte
  call int (sbyte*, ...)* %printf(sbyte* %a_ds, sbyte %a_d2i2sb)

  ;; B
  %b = load int* %B
  %b_ds = getelementptr [8 x sbyte]* %b_dstr, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %b_ds, int %b)

  ;; cast int to double
  %b_i2d = cast int %b to double
  %b_fs = getelementptr [8 x sbyte]* %b_fstr, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %b_fs, double %b_i2d)

  ret int 0
}
