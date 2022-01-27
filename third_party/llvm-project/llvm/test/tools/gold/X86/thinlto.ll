; First ensure that the ThinLTO handling in the gold plugin handles
; bitcode without summary sections gracefully.
; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/thinlto.ll -o %t2.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    -shared %t.o %t2.o -o %t3
; RUN: not test -e %t3
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM

; Next generate summary sections and test gold handling.
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o

; Ensure gold generates an index and not a binary if requested.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    -shared %t.o %t2.o -o %t3
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump %t2.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND2
; RUN: llvm-dis %t.o.thinlto.bc -o - | FileCheck %s --check-prefix=DIS1
; RUN: llvm-dis %t2.o.thinlto.bc -o - | FileCheck %s --check-prefix=DIS2
; RUN: not test -e %t3

; Ensure gold generates an index as well as a binary with save-temps in ThinLTO mode.
; First force single-threaded mode
; RUN: rm -f %t4*
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=jobs=1 \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-bcanalyzer -dump %t4.index.bc | FileCheck %s --check-prefix=COMBINED
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM
; Ensure ld does not emit empty combined module in default.
; RUN: ls %t4.lto.o* | count 2

; Check with --no-map-whole-files
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=jobs=1 \
; RUN:    --no-map-whole-files \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-bcanalyzer -dump %t4.index.bc | FileCheck %s --check-prefix=COMBINED
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM

; Next force multi-threaded mode
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=jobs=2 \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-bcanalyzer -dump %t4.index.bc | FileCheck %s --check-prefix=COMBINED
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM

; Test --plugin-opt=obj-path to ensure unique object files generated.
; RUN: rm -f %t5.o %t5.o1
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=jobs=2 \
; RUN:    --plugin-opt=obj-path=%t5.o \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-nm %t5.o1 | FileCheck %s --check-prefix=NM2
; RUN: llvm-nm %t5.o2 | FileCheck %s --check-prefix=NM2
; Ensure ld emits empty combined module if specific obj-path.
; RUN: ls %t5.o* | count 3

; Test to ensure that thinlto-index-only with obj-path creates the file.
; RUN: rm -f %t5.o %t5.o1
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=jobs=2 \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    --plugin-opt=obj-path=%t5.o \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-readobj -h %t5.o | FileCheck %s --check-prefix=FORMAT
; RUN: llvm-nm %t5.o 2>&1 | count 0

; NM: T f
; NM2: T {{f|g}}
; FORMAT: Format: elf64-x86-64

; The backend index for this module contains summaries from itself and
; Inputs/thinlto.ll, as it imports from the latter.
; BACKEND1: <MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; BACKEND1-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND1-NEXT: <VERSION
; BACKEND1-NEXT: <FLAGS
; BACKEND1-NEXT: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1-NEXT: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1-NEXT: <COMBINED
; BACKEND1-NEXT: <COMBINED
; BACKEND1-NEXT: <BLOCK_COUNT
; BACKEND1-NEXT: </GLOBALVAL_SUMMARY_BLOCK

; The backend index for Input/thinlto.ll contains summaries from itself only,
; as it does not import anything.
; BACKEND2: <MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp2.o'
; BACKEND2-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VERSION
; BACKEND2-NEXT: <FLAGS
; BACKEND2-NEXT: <VALUE_GUID op0=1 op1=-5300342847281564238
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: <BLOCK_COUNT
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK

; DIS1: ^0 = module: (path: "{{.*}}thinlto.ll.tmp.o", hash: (0, 0, 0, 0, 0))
; DIS1: ^1 = module: (path: "{{.*}}thinlto.ll.tmp2.o", hash: (0, 0, 0, 0, 0))
; DIS1: ^2 = gv: (guid: 13146401226427987378, summaries: (function: (module: ^1, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 0, canAutoHide: 0), insts: 1)))
; DIS1: ^3 = gv: (guid: 14740650423002898831, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 0, canAutoHide: 0), insts: 2, calls: ((callee: ^2)))))

; DIS2: ^0 = module: (path: "{{.*}}thinlto.ll.tmp2.o", hash: (0, 0, 0, 0, 0))
; DIS2: ^1 = gv: (guid: 13146401226427987378, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 0, canAutoHide: 0), insts: 1)))

; COMBINED: <MODULE_STRTAB_BLOCK
; COMBINED-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; COMBINED-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; COMBINED-NEXT: </MODULE_STRTAB_BLOCK
; COMBINED-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT: <VERSION
; COMBINED-NEXT: <FLAGS
; COMBINED-NEXT: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; COMBINED-NEXT: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <BLOCK_COUNT
; COMBINED-NEXT: </GLOBALVAL_SUMMARY_BLOCK

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
