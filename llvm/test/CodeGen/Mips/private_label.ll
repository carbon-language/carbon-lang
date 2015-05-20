; RUN: llc -march=mips < %s | FileCheck -check-prefix=O32 %s
; RUN: llc -march=mips64 -target-abi n32 < %s | FileCheck -check-prefix=N32 %s
; RUN: llc -march=mips64 -target-abi n64 < %s | FileCheck -check-prefix=N64 %s

define void @labels() nounwind  {
entry:
  ; O32: $func_end
  ; N32: .Lfunc_end
  ; N64: .Lfunc_end
  ret void
}
