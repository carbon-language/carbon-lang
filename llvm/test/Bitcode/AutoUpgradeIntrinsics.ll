; This isn't really an assembly file. It just runs test on bitcode to ensure
; it is auto-upgraded.
; RUN: llvm-dis < %s.bc | FileCheck %s
; CHECK-NOT: {i32 @llvm\\.ct}
; CHECK-NOT: {llvm\\.part\\.set\\.i\[0-9\]*\\.i\[0-9\]*\\.i\[0-9\]*}
; CHECK-NOT: {llvm\\.part\\.select\\.i\[0-9\]*\\.i\[0-9\]*}
; CHECK-NOT: {llvm\\.bswap\\.i\[0-9\]*\\.i\[0-9\]*}

