//===- Debug.h - An easy way to add debug output to your code ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a handle way of adding debugging information to your
// code, without it being enabled all of the time, and without having to add
// command line options to enable it.
//
// In particular, just wrap your code with the DEBUG() macro, and it will be
// enabled automatically if you specify '-debug' on the command-line.
// Alternatively, you can also use the SET_DEBUG_TYPE("foo") macro to specify
// that your debug code belongs to class "foo".  Then, on the command line, you
// can specify '-debug-only=foo' to enable JUST the debug information for the
// foo class.
//
// When compiling in release mode, the -debug-* options and all code in DEBUG()
// statements disappears, so it does not effect the runtime of the code.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUG_H
#define SUPPORT_DEBUG_H

namespace llvm {

// DebugFlag - This boolean is set to true if the '-debug' command line option
// is specified.  This should probably not be referenced directly, instead, use
// the DEBUG macro below.
//
extern bool DebugFlag;

// isCurrentDebugType - Return true if the specified string is the debug type
// specified on the command line, or if none was specified on the command line
// with the -debug-only=X option.
//
bool isCurrentDebugType(const char *Type);

// DEBUG macro - This macro should be used by passes to emit debug information.
// In the '-debug' option is specified on the commandline, and if this is a
// debug build, then the code specified as the option to the macro will be
// executed.  Otherwise it will not be.  Example:
//
// DEBUG(cerr << "Bitset contains: " << Bitset << "\n");
//

#ifndef DEBUG_TYPE
#define DEBUG_TYPE ""
#endif

#ifdef NDEBUG
#define DEBUG(X)
#else
#define DEBUG(X) \
  do { if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) { X; } } while (0)
#endif

} // End llvm namespace

#endif
