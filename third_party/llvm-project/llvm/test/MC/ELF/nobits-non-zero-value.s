# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s

## -filetype=asm does not check the error.
# RUN: llvm-mc -triple=x86_64 %s

.section .tbss,"aw",@nobits
# MCRelaxableFragment
# CHECK: {{.*}}.s:[[#@LINE+1]]:3: error: SHT_NOBITS section '.tbss' cannot have instructions
  jmp foo

.bss
# CHECK: {{.*}}.s:[[#@LINE+1]]:3: error: SHT_NOBITS section '.bss' cannot have instructions
  addb %al,(%rax)

# CHECK: <unknown>:0: error: SHT_NOBITS section '.bss' cannot have non-zero initializers
  .long 1
