; RUN: llvm-upgrade < %s | llvm-as | llc | grep {foo bar":}

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.2.0"

"foo bar" = global int 4

