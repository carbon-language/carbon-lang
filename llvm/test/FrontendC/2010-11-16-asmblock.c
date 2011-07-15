// RUN: %llvmgcc -S %s -fasm-blocks -o - | FileCheck %s
// XFAIL: *
// XTARGET: x86,i386,i686
// 84282548

void foo()
{
// CHECK:  %0 = call i32 asm sideeffect "", "={ecx}"() nounwind
// CHECK:  %1 = call i32 asm sideeffect alignstack "sall $$3, $0", "={ecx},{ecx},~{dirflag},~{fpsr},~{flags},~{memory}"(i32 %0) nounwind
// CHECK:  store i32 %asmtmp, i32* %"%ecx"
 __asm {
   sal ecx, 3;
   add esi, ecx;
   add edi, ecx;
 }
}
