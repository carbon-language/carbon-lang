//===- CodeGen/ValueTypes.h - Low-Level Target independ. types --*- C++ -*-===//
//
// This file defines the set of low-level target independent types which various
// values in the code generator are.  This allows the target specific behavior
// of instructions to be described to target independent passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_VALUETYPES_H
#define LLVM_CODEGEN_VALUETYPES_H

/// MVT namespace - This namespace defines the ValueType enum, which contains
/// the various low-level value types.
///
namespace MVT {  // MRF = Machine Register Flags
  enum ValueType {
    Other          =   0 << 0,   // This is a non-standard value
    i1             =   1 << 0,   // This is a 1 bit integer value
    i8             =   1 << 1,   // This is an 8 bit integer value
    i16            =   1 << 2,   // This is a 16 bit integer value
    i32            =   1 << 3,   // This is a 32 bit integer value
    i64            =   1 << 4,   // This is a 64 bit integer value
    i128           =   1 << 5,   // This is a 128 bit integer value

    f32             =   1 << 6,   // This is a 32 bit floating point value
    f64             =   1 << 7,   // This is a 64 bit floating point value
    f80             =   1 << 8,   // This is a 80 bit floating point value
    f128            =   1 << 9,   // This is a 128 bit floating point value
  };
};

#endif

