// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// PR8283

// CHECK: pavgusb %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xbf]
pavgusb	%mm2, %mm1

// CHECK: pavgusb 9(%esi,%edx), %mm3 # encoding: [0x0f,0x0f,0x5c,0x16,0x09,0
pavgusb	9(%esi,%edx), %mm3

        
