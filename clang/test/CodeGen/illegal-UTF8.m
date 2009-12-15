// RUN: %clang %s -S -m64 -o -

@class NSString;

// FIXME: GCC emits the following warning:
// CodeGen/illegal-UTF8.m:4: warning: input conversion stopped due to an input byte that does not belong to the input codeset UTF-8

NSString *S = @"\xff\xff___WAIT___";
