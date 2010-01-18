//===- SMLoc.h - Manager for Source Buffers & Diagnostics -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SMLoc class.  This class encapsulates a location in
// source code for use in diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SMLOC_H
#define SUPPORT_SMLOC_H

namespace llvm {

// SMLoc - Represents a location in source code.
class SMLoc {    
  const char *Ptr;
public:
  SMLoc() : Ptr(0) {}
  SMLoc(const SMLoc &RHS) : Ptr(RHS.Ptr) {}
    
  bool isValid() const { return Ptr != 0; }
    
  bool operator==(const SMLoc &RHS) const { return RHS.Ptr == Ptr; }
  bool operator!=(const SMLoc &RHS) const { return RHS.Ptr != Ptr; }
    
  const char *getPointer() const { return Ptr; }
    
  static SMLoc getFromPointer(const char *Ptr) {
    SMLoc L;
    L.Ptr = Ptr;
    return L;
  }
};

}

#endif