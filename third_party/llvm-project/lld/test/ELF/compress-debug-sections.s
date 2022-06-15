# REQUIRES: x86, zlib

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1 --compress-debug-sections=zlib

# RUN: llvm-objdump -s %t1 | FileCheck %s --check-prefix=ZLIBCONTENT
# ZLIBCONTENT:     Contents of section .debug_str:
# ZLIBCONTENT-NOT: AAAAAAAAA

# RUN: llvm-readobj -S %t1 | FileCheck %s --check-prefix=ZLIBFLAGS
# ZLIBFLAGS:       Section {
# ZLIBFLAGS:         Index:
# ZLIBFLAGS:         Name: .debug_str
# ZLIBFLAGS-NEXT:    Type: SHT_PROGBITS
# ZLIBFLAGS-NEXT:    Flags [
# ZLIBFLAGS-NEXT:      SHF_COMPRESSED

# RUN: llvm-dwarfdump %t1 -debug-str | \
# RUN:   FileCheck %s --check-prefix=DEBUGSTR
# DEBUGSTR:     .debug_str contents:
# DEBUGSTR-NEXT:  BBBBBBBBBBBBBBBBBBBBBBBBBBB
# DEBUGSTR-NEXT:  AAAAAAAAAAAAAAAAAAAAAAAAAAA

## Test alias.
# RUN: ld.lld %t.o -o %t2 --compress-debug-sections zlib
# RUN: llvm-objdump -s %t2 | FileCheck %s --check-prefix=ZLIBCONTENT
# RUN: llvm-readobj -S %t2 | FileCheck %s --check-prefix=ZLIBFLAGS

# RUN: not ld.lld %t.o -o /dev/null --compress-debug-sections=zlib-gabi 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR %s
# ERR: unknown --compress-debug-sections value: zlib-gabi

.section .debug_str,"MS",@progbits,1
.Linfo_string0:
  .asciz "AAAAAAAAAAAAAAAAAAAAAAAAAAA"
.Linfo_string1:
  .asciz "BBBBBBBBBBBBBBBBBBBBBBBBBBB"
