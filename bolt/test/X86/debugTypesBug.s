# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %p/Inputs/main.s -o %tmain.o
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %p/Inputs/file1.s -o %tfile1.o
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %p/Inputs/file2.s -o %tfile2.o
# RUN: %host_cxx %tmain.o %tfile1.o %tfile2.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --reorder-blocks=reverse -update-debug-sections -o %t.out
# RUN: llvm-dwarfdump --debug-types=0x000005f7 %t.out | grep DW_AT_stmt_list | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-OUTPUT: DW_AT_stmt_list	(0x000004d4)
