// RUN: %llvmgcc %s -S -g -o - | grep -v DW_TAG_member
// Interface P should not be a member of interface I in debug info.
@interface P 
@end

@interface I : P 
@end

void fn(I *iptr) {}
