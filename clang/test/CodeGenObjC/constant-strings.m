// RUN: clang-cc -fnext-runtime -emit-llvm -o %t %s
// RUN: clang-cc -fgnu-runtime -emit-llvm -o %t %s && grep NXConstantString %t | count 1
// RUN: clang-cc -fgnu-runtime -fconstant-string-class=NSConstantString -emit-llvm -o %t %s && grep NSConstantString %t | count 1

id a = @"Hello World!";

