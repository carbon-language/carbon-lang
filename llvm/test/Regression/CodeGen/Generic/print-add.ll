; RUN: llvm-as < %s | llc

%.str_1 = internal constant [4 x sbyte] c"%d\0A\00"

declare int %printf(sbyte*, ...)

int %main() {  
  %f = getelementptr [4 x sbyte]* %.str_1, long 0, long 0
  %d = add int 1, 0
  call int (sbyte*, ...)* %printf(sbyte* %f, int %d)
  %e = add int 38, 2
  call int (sbyte*, ...)* %printf(sbyte* %f, int %e)
  %g = add int %d, %d
  %h = add int %e, %g
  call int (sbyte*, ...)* %printf(sbyte* %f, int %h)
  ret int 0
}

