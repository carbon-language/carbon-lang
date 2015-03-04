// Check if -cc1as knows about the 'target-abi' argument.

// RUN: %clang -cc1as -triple mips--linux-gnu -filetype obj -target-cpu mips32 -target-abi o32 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=ABI-O32 %s
// ABI-O32-NOT: clang -cc1as: error: unknown argument: '-target-abi'
