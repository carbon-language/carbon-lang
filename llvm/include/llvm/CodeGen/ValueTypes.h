//===- CodeGen/ValueTypes.h - Low-Level Target independ. types --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
namespace MVT {  // MVT = Machine Value Types
  enum ValueType {
    // If you change this numbering, you must change the values in Target.td as
    // well!
    Other          =   0,   // This is a non-standard value
    i1             =   1,   // This is a 1 bit integer value
    i8             =   2,   // This is an 8 bit integer value
    i16            =   3,   // This is a 16 bit integer value
    i32            =   4,   // This is a 32 bit integer value
    i64            =   5,   // This is a 64 bit integer value
    i128           =   6,   // This is a 128 bit integer value

    f32             =  7,   // This is a 32 bit floating point value
    f64             =  8,   // This is a 64 bit floating point value
    f80             =  9,   // This is a 80 bit floating point value
    f128            = 10,   // This is a 128 bit floating point value

    isVoid          = 11,   // This has no value
  };
};

#endif

