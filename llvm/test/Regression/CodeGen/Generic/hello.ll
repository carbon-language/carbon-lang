; RUN: llvm-as < %s | llc

%.str_1 = internal constant [7 x sbyte] c"hello\0A\00"

declare int %printf(sbyte*, ...)

int %main() {  
  %s = getelementptr [7 x sbyte]* %.str_1, long 0, long 0
  call int (sbyte*, ...)* %printf(sbyte* %s)
  ret int 0
}
