# REQUIRES: x86

# RUN: rm -f %t-no-syms.a
# RUN: rm -f %t-syms.a
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-ar rcTS %t-no-syms.a %t.o
# RUN: llvm-ar rcT %t-syms.a %t.o
# RUN: rm %t.o

# Test error when loading symbols from missing thin archive member.
# RUN: not ld.lld --entry=_Z1fi %t-no-syms.a -o /dev/null 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=ERR1
# ERR1: {{.*}}-no-syms.a: could not get the buffer for a child of the archive: '{{.*}}.o': [[MSG]]

# Test error when thin archive has symbol table but member is missing.
# RUN: not ld.lld --entry=_Z1fi -m elf_amd64_fbsd %t-syms.a -o /dev/null 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=ERR2
# ERR2: {{.*}}-syms.a: could not get the buffer for the member defining symbol f(int): '{{.*}}.o': [[MSG]]
# RUN: not ld.lld --entry=_Z1fi --no-demangle -m elf_amd64_fbsd %t-syms.a -o /dev/null 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=ERR2MANGLE
# ERR2MANGLE: {{.*}}-syms.a: could not get the buffer for the member defining symbol _Z1fi: '{{.*}}.o': [[MSG]]

# Test error when thin archive is linked using --whole-archive but member is missing.
# RUN: not ld.lld --entry=_Z1fi --whole-archive %t-syms.a -o /dev/null 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=ERR3
# ERR3: {{.*}}-syms.a: could not get the buffer for a child of the archive: '{{.*}}.o': [[MSG]]

.global _Z1fi
_Z1fi:
    nop
