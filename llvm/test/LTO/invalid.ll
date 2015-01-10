; RUN: not llvm-lto %S/../Bitcode/invalid.ll.bc 2>&1 | FileCheck %s


; CHECK: llvm-lto{{.*}}: error loading file '{{.*}}/../Bitcode/invalid.ll.bc': Unknown attribute kind (48)
