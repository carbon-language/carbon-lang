// RUN: %llvmgcc %s -c -m32 -fasm-blocks -o /dev/null
// This should not warn about unreferenced label. 7729514.
// XFAIL: *
// XTARGET: x86,i386,i686

void quarterAsm(int array[], int len)
{
  __asm
  {
    mov esi, array;
    mov ecx, len;
    shr ecx, 2;
loop:
    movdqa xmm0, [esi];
    psrad xmm0, 2;
    movdqa [esi], xmm0;
    add esi, 16;
    sub ecx, 1;
    jnz loop;
  }
}
