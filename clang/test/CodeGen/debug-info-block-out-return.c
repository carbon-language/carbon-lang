// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -g -fblocks -emit-llvm -o - %s | FileCheck %s

// Check that arg numbering is not affected by LLVM IR argument numbering -
// since the latter is affected by return-by-out-parameter ABI requirements

// 1 for the argument number (1 indexed), 2 for the line number
// 16777218 == 1 << 24 | 2
// 33554434 == 2 << 24 | 2
// This explains the two magic numbers below, testing that these two arguments
// are numbered correctly. If they are not numbered correctly they may appear
// out of order or not at all (the latter would occur if they were both assigned
// the same argument number by mistake).

// CHECK: metadata !"0x101\00.block_descriptor\0016777218\00{{[0-9]+}}", {{.*}} ; [ DW_TAG_arg_variable ] [.block_descriptor]
// CHECK: metadata !"0x101\00param\0033554434\00{{[0-9]+}}", {{.*}} ; [ DW_TAG_arg_variable ] [param]

// Line directive so we don't have to worry about how many lines preceed the
// test code (as the line number is mangled in with the argument number as shown
// above)
#line 1
typedef struct { int array[12]; } BigStruct_t;
BigStruct_t (^a)() = ^(int param) {
    BigStruct_t b;
    return b;
};
