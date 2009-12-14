// RUN: clang -cc1 -emit-llvm -o %t %s
// RUN: clang -cc1 -fgnu-runtime -emit-llvm -o %t %s && grep NXConstantString %t | count 1
// RUN: clang -cc1 -fgnu-runtime -fconstant-string-class NSConstantString -emit-llvm -o %t %s && grep NSConstantString %t | count 1

id a = @"Hello World!";

