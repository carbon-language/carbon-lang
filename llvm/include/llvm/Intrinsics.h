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

    // Varargs handling intrinsics.
    vastart,        // Used to implement the va_start macro in C
    vaend,          // Used to implement the va_end macro in C
    vacopy,         // Used to implement the va_copy macro in C

    // Code generator intrinsics.
    returnaddress,    // Yields the return address of a dynamic call frame
    frameaddress,     // Yields the frame address of a dynamic call frame
    stacksave,        // Save the stack pointer
    stackrestore,     // Restore the stack pointer
    prefetch,         // Prefetch a value into the cache
    pcmarker,         // Export a PC from near the marker
    readcyclecounter, // Read cycle counter register

    // setjmp/longjmp intrinsics.
    setjmp,         // Used to represent a setjmp call in C
    longjmp,        // Used to represent a longjmp call in C
    sigsetjmp,      // Used to represent a sigsetjmp call in C
    siglongjmp,     // Used to represent a siglongjmp call in C

    // Garbage Collection intrinsics.
    gcroot,         // Defines a new GC root on the stack
    gcread,         // Defines a read of a heap object  (for read barriers)
    gcwrite,        // Defines a write to a heap object (for write barriers)

    // Debugging intrinsics.
    dbg_stoppoint,    // Represents source lines and breakpointable places
    dbg_region_start, // Start of a region
    dbg_region_end,   // End of a region
    dbg_func_start,   // Start of a function
    dbg_declare,      // Declare a local object

    // Standard C library intrinsics.
    memcpy_i32,      // Copy non-overlapping memory blocks.  i32 size.
    memcpy_i64,      // Copy non-overlapping memory blocks.  i64 size.
    memmove_i32,     // Copy potentially overlapping memory blocks.  i32 size.
    memmove_i64,     // Copy potentially overlapping memory blocks.  i64 size.
    memset_i32,      // Fill memory with a byte value.  i32 size.
    memset_i64,      // Fill memory with a byte value.  i64 size.
    isunordered_f32, // Return true if either float argument is a NaN
    isunordered_f64, // Return true if either double argument is a NaN
    sqrt_f32,        // Square root of float
    sqrt_f64,        // Square root of double

    // Bit manipulation instrinsics.
    bswap_i16,      // Byteswap 16 bits
    bswap_i32,      // Byteswap 32 bits
    bswap_i64,      // Byteswap 64 bits
    ctpop_i8,       // Count population of sbyte
    ctpop_i16,      // Count population of short
    ctpop_i32,      // Count population of int
    ctpop_i64,      // Count population of long
    ctlz_i8,        // Count leading zeros of sbyte
    ctlz_i16,       // Count leading zeros of short
    ctlz_i32,       // Count leading zeros of int
    ctlz_i64,       // Count leading zeros of long
    cttz_i8,        // Count trailing zeros of sbyte
    cttz_i16,       // Count trailing zeros of short
    cttz_i32,       // Count trailing zeros of int
    cttz_i64,       // Count trailing zeros of long
  };

} // End Intrinsic namespace

} // End llvm namespace

#endif
