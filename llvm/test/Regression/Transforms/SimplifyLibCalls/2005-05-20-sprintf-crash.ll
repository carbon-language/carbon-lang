; RUN: llvm-as < %s | opt -simplify-libcalls -disable-output

%G = constant [3 x sbyte] c"%s\00"

declare int %sprintf(sbyte*, sbyte*, ...)

void %foo(sbyte*%P, int *%X) {
  call int(sbyte*,sbyte*, ...)* %sprintf(sbyte* %P, sbyte* getelementptr ([3 x sbyte]* %G, int 0, int 0), int* %X)
  ret void
}
