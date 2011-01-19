// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fobjc-default-synthesize-properties -emit-llvm -g %s -o %t 
// RUN: grep DW_TAG_lexical_block %t | count 5
// rdar://8757124

@class NSArray;

void f(NSArray *a) {
  id keys;
  for (id thisKey in keys) {
  }
  for (id thisKey in keys) {
  }
}
