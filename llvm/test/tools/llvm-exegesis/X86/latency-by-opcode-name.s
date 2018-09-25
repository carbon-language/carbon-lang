# RUN: llvm-exegesis -mode=latency -opcode-name=ADD32rr | FileCheck %s

CHECK:      mode:            latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD32rr
