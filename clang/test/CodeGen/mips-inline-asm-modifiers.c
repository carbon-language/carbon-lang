// REQUIRES: mips-registered-target
// RUN: %clang -target mipsel-unknown-linux -S -o - -emit-llvm %s \
// RUN: | FileCheck %s

// This checks that the frontend will accept inline asm operand modifiers

int printf(const char*, ...);

  // CHECK: %{{[0-9]+}} = call i32 asm ".set noreorder;\0Alw    $0,$1;\0A.set reorder;\0A", "=r,*m"(i32* getelementptr inbounds ([8 x i32]* @b, i32 {{[0-9]+}}, i32 {{[0-9]+}})) #2, !srcloc !0
  // CHECK: %{{[0-9]+}} = call i32 asm "lw    $0,${1:D};\0A", "=r,*m"(i32* getelementptr inbounds ([8 x i32]* @b, i32 {{[0-9]+}}, i32 {{[0-9]+}})) #2, !srcloc !1
int b[8] = {0,1,2,3,4,5,6,7};
int  main()
{
  int i;

  // The first word. Notice, no 'D'
  {asm (
  ".set noreorder;\n"
  "lw    %0,%1;\n"
  ".set reorder;\n"
  : "=r" (i)
  : "m" (*(b+4)));}

  printf("%d\n",i);

  // The second word
  {asm (
  "lw    %0,%D1;\n"
  : "=r" (i)
  : "m" (*(b+4))
  );}

  printf("%d\n",i);

  return 1;
}
