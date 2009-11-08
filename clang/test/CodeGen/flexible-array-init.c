// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm -o - %s | grep 7 | count 1
// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm -o - %s | grep 11 | count 1
// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm -o - %s | grep 13 | count 1
// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm -o - %s | grep 15 | count 1

struct { int x; int y[]; } a = { 1, 7, 11 };

struct { int x; int y[]; } b = { 1, { 13, 15 } };
