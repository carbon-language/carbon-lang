; RUN: llc -mcpu=mips32 -march=mipsel -relocation-model=static -o - %s | FileCheck %s
; XFAIL: *

define void @isr_sw0() #0 {
  call void bitcast (void (...)* @write to void ()*)()
}

declare void @write(...)

attributes #0 = { "interrupt"="sw0" }

