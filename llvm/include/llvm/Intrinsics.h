//===-- llvm/Instrinsics.h - LLVM Intrinsic Function Handling ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a set of enums which allow processing of intrinsic
// functions.  Values of these enum types are returned by
// Function::getIntrinsicID.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTRINSICS_H
#define LLVM_INTRINSICS_H

namespace llvm {

/// Intrinsic Namespace - This namespace contains an enum with a value for
/// every intrinsic/builtin function known by LLVM.  These enum values are
/// returned by Function::getIntrinsicID().
///
namespace Intrinsic {
  enum ID {
    not_intrinsic = 0,   // Must be zero

    // Varargs handling intrinsics...
    va_start,       // Used to implement the va_start macro in C
    va_end,         // Used to implement the va_end macro in C
    va_copy,        // Used to implement the va_copy macro in C

    // Setjmp/Longjmp intrinsics...
    setjmp,         // Used to represent a setjmp call in C
    longjmp,        // Used to represent a longjmp call in C
    sigsetjmp,      // Used to represent a sigsetjmp call in C
    siglongjmp,     // Used to represent a siglongjmp call in C

    // Debugging intrinsics...
    dbg_stoppoint,    // Represents source lines and breakpointable places
    dbg_region_start, // Start of a region
    dbg_region_end,   // End of a region
    dbg_func_start,   // Start of a function
    dbg_declare,      // Declare a local object

    // Standard libc functions...
    memcpy,         // Used to copy non-overlapping memory blocks
    memmove,        // Used to copy overlapping memory blocks


    // Standard libm functions...
    

    //===------------------------------------------------------------------===//
    // This section defines intrinsic functions used to represent Alpha
    // instructions...
    //
    alpha_ctlz,     // CTLZ (count leading zero): counts the number of leading
                    // zeros in the given ulong value

    alpha_cttz,     // CTTZ (count trailing zero): counts the number of trailing
                    // zeros in the given ulong value 

    alpha_ctpop,    // CTPOP (count population): counts the number of ones in
                    // the given ulong value 

    alpha_umulh,    // UMULH (unsigned multiply quadword high): Takes two 64-bit
                    // (ulong) values, and returns the upper 64 bits of their
                    // 128 bit product as a ulong

    alpha_vecop,    // A generic vector operation. This function is used to
                    // represent various Alpha vector/multimedia instructions.
                    // It takes 4 parameters:
                    //  - the first two are 2 ulong vectors
                    //  - the third (uint) is the size (in bytes) of each 
                    //    vector element. Thus a value of 1 means that the two
                    //    input vectors consist of 8 bytes
                    //  - the fourth (uint) is the operation to be performed on
                    //    the vectors. Its possible values are defined in the
                    //    enumeration AlphaVecOps.

    alpha_pup,      // A pack/unpack operation. This function is used to
                    // represent Alpha pack/unpack operations. 
                    // It takes 3 parameters:
                    //  - the first is an ulong to pack/unpack
                    //  - the second (uint) is the size of each component
                    //    Valid values are 2 (word) or 4 (longword)
                    //  - the third (uint) is the operation to be performed.
                    //    Possible values defined in the enumeration 
                    //    AlphaPupOps

    alpha_bytezap,  // This intrinsic function takes two parameters: a ulong 
                    // (64-bit) value and a ubyte value, and returns a ulong.
                    // Each bit in the ubyte corresponds to a byte in the 
                    // ulong. If the bit is 0, the byte in the output equals
                    // the corresponding byte in the input, else the byte in
                    // the output is zero.

    alpha_bytemanip,// This intrinsic function represents all Alpha byte
                    // manipulation instructions. It takes 3 parameters:
                    //  - The first two are ulong inputs to operate on
                    //  - The third (uint) is the operation to perform. 
                    //    Possible values defined in the enumeration
                    //    AlphaByteManipOps

    alpha_dfpbop,   // This intrinsic function represents Alpha instructions
                    // that operate on two doubles and return a double. The
                    // first two parameters are the two double values to
                    // operate on, and the third is a uint that specifies the
                    // operation to perform. Its possible values are defined in
                    // the enumeration AlphaFloatingBinaryOps

    alpha_dfpuop,   // This intrinsic function represents operation on a single
                    // double precision floating point value. The first 
                    // paramters is the value and the second is the operation.
                    // The possible values for the operations are defined in the
                    // enumeration AlphaFloatingUnaryOps

    alpha_unordered,// This intrinsic function tests if two double precision
                    // floating point values are unordered. It has two
                    // parameters: the two values to be tested. It return a
                    // boolean true if the two are unordered, else false.

    alpha_uqtodfp,  // A generic function that converts a ulong to a double.
                    // How the conversion is performed is specified by the
                    // second parameter, the possible values for which are
                    // defined in the AlphaUqToDfpOps enumeration

    alpha_uqtosfp,  // A generic function that converts a ulong to a float.
                    // How the conversion is performed is specified by the
                    // second parameter, the possible values for which are
                    // defined in the AlphaUqToSfpOps enumeration

    alpha_dfptosq,  // A generic function that converts double to a long.
                    // How the conversion is performed is specified by the
                    // second parameter, the possible values for which are
                    // defined in the AlphaDfpToSqOps enumeration

    alpha_sfptosq,  // A generic function that converts a float to a long.
                    // How the conversion is performed is specified by the
                    // second parameter, the possible values for which are
                    // defined in the AlphaSfpToSq enumeration
  };

} // End Intrinsic namespace

} // End llvm namespace

#endif
