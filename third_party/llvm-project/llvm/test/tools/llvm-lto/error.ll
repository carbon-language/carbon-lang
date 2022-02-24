; RUN: not llvm-lto foobar 2>&1 | FileCheck -DMSG=%errc_ENOENT %s
; CHECK: llvm-lto: error loading file 'foobar': [[MSG]]

; RUN: not llvm-lto --list-symbols-only %S/Inputs/empty.bc 2>&1 | FileCheck %s --check-prefix=CHECK-LIST
; CHECK-LIST: llvm-lto: error loading file '{{.*}}/Inputs/empty.bc': The file was not recognized as a valid object file

; RUN: not llvm-lto --list-dependent-libraries-only %S/Inputs/empty.bc 2>&1 | FileCheck %s --check-prefix=CHECK-LIBS
; CHECK-LIBS: llvm-lto: {{.*}}/Inputs/empty.bc: Could not read LTO input file: The file was not recognized as a valid object file

; RUN: not llvm-lto --print-macho-cpu-only %S/Inputs/empty.bc 2>&1 | FileCheck %s --check-prefix=CHECK-MACHO
; CHECK-MACHO: llvm-lto: error: The file was not recognized as a valid object file

; RUN: not llvm-lto --thinlto %S/Inputs/empty.bc 2>&1 | FileCheck %s --check-prefix=CHECK-THIN
; CHECK-THIN: llvm-lto: error loading file '{{.*}}/Inputs/empty.bc': file too small to contain bitcode header
