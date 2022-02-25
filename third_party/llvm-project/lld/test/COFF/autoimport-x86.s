# REQUIRES: x86

# RUN: echo -e ".global variable\n.global DllMainCRTStartup\n.text\nDllMainCRTStartup:\nret\n.data\nvariable:\n.long 42" > %t-lib.s
# RUN: llvm-mc -triple=x86_64-windows-gnu %t-lib.s -filetype=obj -o %t-lib.obj
# RUN: lld-link -out:%t-lib.dll -dll -entry:DllMainCRTStartup %t-lib.obj -lldmingw -implib:%t-lib.lib

# RUN: llvm-mc -triple=x86_64-windows-gnu -defsym listptrs=1 %s -filetype=obj -o %t.obj
# RUN: lld-link -lldmingw -debug:symtab -out:%t.exe -entry:main %t.obj %t-lib.lib -verbose

# RUN: llvm-mc -triple=x86_64-windows-gnu -defsym listptrs=0 %s -filetype=obj -o %t.noptrs.obj
# RUN: not lld-link -lldmingw -runtime-pseudo-reloc:no -debug:symtab -out:%t.exe -entry:main %t.noptrs.obj %t-lib.lib 2>&1 | FileCheck --check-prefix=DISABLED %s

# RUN: llvm-readobj --coff-imports %t.exe | FileCheck -check-prefix=IMPORTS %s
# RUN: llvm-objdump -d %t.exe | FileCheck --check-prefix=DISASM %s
# RUN: llvm-objdump -s %t.exe | FileCheck --check-prefix=CONTENTS %s
# RUN: llvm-nm %t.exe | FileCheck -check-prefix=SYMBOLS %s

# IMPORTS: Import {
# IMPORTS-NEXT: Name: autoimport-x86.s.tmp-lib.dll
# IMPORTS-NEXT: ImportLookupTableRVA: 0x2070
# IMPORTS-NEXT: ImportAddressTableRVA: 0x2080
# IMPORTS-NEXT: Symbol: variable (0)
# IMPORTS-NEXT: }

# DISASM: Disassembly of section .text:
# DISASM-EMPTY:
# DISASM: <main>:
# Relative offset at 0x1002 pointing at the IAT at 0x2080.
# DISASM: 140001000:      8b 05 7a 10 00 00       movl    4218(%rip), %eax
# DISASM: 140001006:      c3      retq

# Runtime pseudo reloc list header consisting of 0x0, 0x0, 0x1.
# First runtime pseudo reloc, with import from 0x2080,
# applied at 0x1002, with a size of 32 bits.
# Second runtime pseudo reloc, with import from 0x2080,
# applied at 0x3000, with a size of 64 bits.
# CONTENTS: Contents of section .rdata:
# CONTENTS:  140002000 00000000 00000000 01000000 80200000
# CONTENTS:  140002010 02100000 20000000 80200000 00300000
# CONTENTS:  140002020 40000000
# ptr: pointing at the IAT RVA at 0x2080
# relocs: pointing at the runtime pseudo reloc list at
# 0x2000 - 0x2024.
# CONTENTS: Contents of section .data:
# CONTENTS:  140003000 80200040 01000000 00200040 01000000
# CONTENTS:  140003010 24200040 01000000

# Check that the automatically imported symbol "variable" is not listed in
# the symbol table.
# SYMBOLS-NOT: variable

# DISABLED: error: automatic dllimport of variable in {{.*}}autoimport-x86.s.tmp.noptrs.obj requires pseudo relocations

    .global main
    .text
main:
    movl variable(%rip), %eax
    ret
    .data
ptr:
    .quad variable
.if listptrs==1
relocs:
    .quad __RUNTIME_PSEUDO_RELOC_LIST__
    .quad __RUNTIME_PSEUDO_RELOC_LIST_END__
.endif
