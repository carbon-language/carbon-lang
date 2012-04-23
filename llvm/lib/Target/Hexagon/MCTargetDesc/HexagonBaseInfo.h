//===-- HexagonBaseInfo.h - Top level definitions for Hexagon --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Hexagon target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBASEINFO_H
#define HEXAGONBASEINFO_H

namespace llvm {

/// HexagonII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace HexagonII {

  // *** The code below must match HexagonInstrFormat*.td *** //

  // MCInstrDesc TSFlags
  enum {

    // Predicated instructions.
    PredicatedPos  = 1,
    PredicatedMask = 0x1
  };

  // *** The code above must match HexagonInstrFormat*.td *** //

} // End namespace HexagonII.

} // End namespace llvm.

#endif
