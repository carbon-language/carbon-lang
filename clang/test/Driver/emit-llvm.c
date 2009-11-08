// RUN: not clang -ccc-host-triple i386-pc-linux-gnu -emit-llvm -o %t %s 2> %t.log
// RUN: grep 'unable to pass LLVM bit-code files to linker' %t.log

