; RUN: llvm-as < %s | llc

%.str_1 = internal constant [4 x sbyte] c"%d\0A\00"

declare int %printf(sbyte*, ...)

int %main() {  
  %f = getelementptr [4 x sbyte]* %.str_1, long 0, long 0
  %d = add int 0, 0
  %tmp.0 = call int (sbyte*, ...)* %printf(sbyte* %f, int %d)
  ret int 0
}

