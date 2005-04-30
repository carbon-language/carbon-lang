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
    returnaddress,  // Yields the return address of a dynamic call frame
    frameaddress,   // Yields the frame address of a dynamic call frame
    prefetch,       // Prefetch a value into the cache
    pcmarker,       // Export a PC from near the marker

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


    // Standard libc functions.
    memcpy,         // Copy non-overlapping memory blocks
    memmove,        // Copy potentially overlapping memory blocks
    memset,         // Fill memory with a byte value

    // libm related functions.
    isunordered,    // Return true if either argument is a NaN
    sqrt,

    // Input/Output intrinsics.
    readport,
    writeport,
    readio,
    writeio
  };

} // End Intrinsic namespace

} // End llvm namespace

#endif
