# --allow-shlib-undefined and --no-allow-shlib-undefined are fully
# ignored in linker implementation.
# --allow-shlib-undefined is set by default
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
# RUN: %p/Inputs/allow-shlib-undefined.s -o %t
# RUN: lld -shared -flavor gnu2 %t -o %t.so
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1

# Executable: should link with DSO containing undefined symbols in any case.
# RUN: lld -flavor gnu2 %t1 %t.so -o %t2
# RUN: lld -flavor gnu2 --no-allow-shlib-undefined %t1 %t.so -o %t2
# RUN: lld -flavor gnu2 --allow-shlib-undefined %t1 %t.so -o %t2

# DSO with undefines:
# should link with or without any of these options.
# RUN: lld -shared -flavor gnu2 %t -o %t.so
# RUN: lld -shared --allow-shlib-undefined -flavor gnu2 %t -o %t.so
# RUN: lld -shared --no-allow-shlib-undefined -flavor gnu2 %t -o %t.so

# Executable still should not link when have undefines inside.
# RUN: not lld -flavor gnu2 %t -o %t.so

.globl _start
_start:
  call _shared
