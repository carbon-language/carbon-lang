; RUN: llvm-as %s -o /dev/null
  
 %Hosp = type { { \2*, { \2, %Hosp }* }, { \2*, { \2, %Hosp }* } }
