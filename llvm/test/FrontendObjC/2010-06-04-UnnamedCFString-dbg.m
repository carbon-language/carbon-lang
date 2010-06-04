// RUN: %llvmgcc -S -O0 -g %s -o - | grep DW_TAG_variable | count 1

// Do not emit debug info for unnamed builtin CFString variable.
@interface Foo
@end
Foo *FooName = @"FooBar";
