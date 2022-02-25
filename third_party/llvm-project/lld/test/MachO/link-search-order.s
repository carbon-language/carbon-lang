# REQUIRES: x86

################ Place dynlib in %tD, and archive in %tA
# RUN: rm -rf %t %tA %tD
# RUN: mkdir -p %t %tA %tD
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %p/Inputs/libhello.s -o %t/hello.o
# RUN: %lld -dylib -install_name @executable_path/libhello.dylib %t/hello.o -o %t/libhello.dylib
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %p/Inputs/libgoodbye.s -o %t/goodbye.o
# RUN: %lld -dylib -install_name @executable_path/libgoodbye.dylib %t/goodbye.o -o %tD/libgoodbye.dylib
# RUN: llvm-ar --format=darwin crs %tA/libgoodbye.a %t/goodbye.o
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o

################ default, which is the same as -search_paths_first
# RUN: %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tA -L%tD -L%t -lhello -lgoodbye -lSystem %t/test.o \
# RUN:     --print-dylib-search | FileCheck --check-prefix=ARCHIVESEARCH -DPATH=%t %s
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=ARCHIVE %s

################ Test all permutations of -L%t{A,D} with -search_paths_first
# RUN: %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tA -L%tD -L%t -lhello -lgoodbye -lSystem %t/test.o -search_paths_first
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=ARCHIVE %s
# RUN: %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tD -L%tA -L%t -lhello -lgoodbye -lSystem %t/test.o -search_paths_first
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=DYLIB %s
# RUN: %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tA       -L%t -lhello -lgoodbye -lSystem %t/test.o -search_paths_first
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=ARCHIVE %s
# RUN: %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tD      -L%t -lhello -lgoodbye -lSystem %t/test.o -search_paths_first
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=DYLIB %s

################ Test all permutations of -L%t{A,D} with -search_dylibs_first
# RUN: env RC_TRACE_DYLIB_SEARCHING=1 %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tA -L%tD -L%t -lhello -lgoodbye -lSystem %t/test.o -search_dylibs_first \
# RUN:     | FileCheck --check-prefix=DYLIBSEARCH -DPATH=%t %s
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=DYLIB %s
# RUN: %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tD -L%tA -L%t -lhello -lgoodbye -lSystem %t/test.o -search_dylibs_first
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=DYLIB %s
# RUN: %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tA       -L%t -lhello -lgoodbye -lSystem %t/test.o -search_dylibs_first
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=ARCHIVE %s
# RUN: %lld -L%S/Inputs/MacOSX.sdk/usr/lib -o %t/test -Z \
# RUN:     -L%tD       -L%t -lhello -lgoodbye -lSystem %t/test.o -search_dylibs_first
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck --check-prefix=DYLIB %s

# DYLIB: @executable_path/libhello.dylib
# DYLIB: @executable_path/libgoodbye.dylib
# DYLIB: /usr/lib/libSystem.dylib

# DYLIBSEARCH:      searched {{.*}}/MacOSX.sdk/usr/lib{{[/\\]}}libhello.tbd, not found
# DYLIBSEARCH-NEXT: searched {{.*}}/MacOSX.sdk/usr/lib{{[/\\]}}libhello.dylib, not found
# DYLIBSEARCH-NEXT: searched [[PATH]]A{{[/\\]}}libhello.tbd, not found
# DYLIBSEARCH-NEXT: searched [[PATH]]A{{[/\\]}}libhello.dylib, not found
# DYLIBSEARCH:      searched [[PATH]]{{[/\\]}}libhello.dylib, found
# DYLIBSEARCH:      searched [[PATH]]D{{[/\\]}}libgoodbye.dylib, found

# ARCHIVE:     @executable_path/libhello.dylib
# ARCHIVE-NOT: @executable_path/libgoodbye.dylib
# ARCHIVE:     /usr/lib/libSystem.dylib

# ARCHIVESEARCH:      searched {{.*}}/MacOSX.sdk/usr/lib{{[/\\]}}libhello.tbd, not found
# ARCHIVESEARCH-NEXT: searched {{.*}}/MacOSX.sdk/usr/lib{{[/\\]}}libhello.dylib, not found
# ARCHIVESEARCH-NEXT: searched {{.*}}/MacOSX.sdk/usr/lib{{[/\\]}}libhello.a, not found
# ARCHIVESEARCH-NEXT: searched [[PATH]]A{{[/\\]}}libhello.tbd, not found
# ARCHIVESEARCH-NEXT: searched [[PATH]]A{{[/\\]}}libhello.dylib, not found
# ARCHIVESEARCH-NEXT: searched [[PATH]]A{{[/\\]}}libhello.a, not found
# ARCHIVESEARCH:      searched [[PATH]]{{[/\\]}}libhello.dylib, found
# ARCHIVESEARCH:      searched [[PATH]]A{{[/\\]}}libgoodbye.a, found

.section __TEXT,__text
.global _main

_main:
  movl $0x2000004, %eax                         # write()
  mov $1, %rdi                                  # stdout
  movq _hello_world@GOTPCREL(%rip), %rsi
  mov $13, %rdx                                 # length
  syscall

  movl $0x2000004, %eax                         # write()
  mov $1, %rdi                                  # stdout
  movq _hello_its_me@GOTPCREL(%rip), %rsi
  mov $15, %rdx                                 # length
  syscall

  movl $0x2000004, %eax                         # write()
  mov $1, %rdi                                  # stdout
  movq _goodbye_world@GOTPCREL(%rip), %rsi
  mov $15, %rdx                                 # length
  syscall
  mov $0, %rax
  ret
