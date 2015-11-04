@ RUN: not llvm-mc < %s -triple thumbv5-linux-gnueabi -filetype=obj -o %t 2>&1 | FileCheck %s

        bl      end
        .space 0x400000
end:

@ CHECK: out of range for branch
