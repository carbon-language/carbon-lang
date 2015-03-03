// RUN: %clang_cc1 -fblocks -g -emit-llvm -o - %s | FileCheck %s
// Verify that the desired debugging type is generated for a structure
//  member that is a pointer to a block. 

// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "__block_literal_generic"
// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "__block_descriptor"
struct inStruct {
  void (^genericBlockPtr)();
} is;

