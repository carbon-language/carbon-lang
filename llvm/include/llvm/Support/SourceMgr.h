//===- SourceMgr.h - Manager for Source Buffers & Diagnostics ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SourceMgr class.  This class is used as a simple
// substrate for diagnostics, #include handling, and other low level things for
// simple parsers.
//
//===----------------------------------------------------------------------===//

#ifndef TGSOURCEMGR_H
#define TGSOURCEMGR_H

#include <string>
#include <vector>
#include <cassert>

namespace llvm {
  class MemoryBuffer;
  class TGSourceMgr;
  
class SMLoc {
  const char *Ptr;
public:
  SMLoc() : Ptr(0) {}
  SMLoc(const SMLoc &RHS) : Ptr(RHS.Ptr) {}
  
  bool operator==(const SMLoc &RHS) const { return RHS.Ptr == Ptr; }
  bool operator!=(const SMLoc &RHS) const { return RHS.Ptr != Ptr; }

  const char *getPointer() const { return Ptr; }
  
  static SMLoc getFromPointer(const char *Ptr) {
    SMLoc L;
    L.Ptr = Ptr;
    return L;
  }
};

/// TGSourceMgr - This owns the files read by tblgen, handles include stacks,
/// and handles printing of diagnostics.
class TGSourceMgr {
  struct SrcBuffer {
    /// Buffer - The memory buffer for the file.
    MemoryBuffer *Buffer;
    
    /// IncludeLoc - This is the location of the parent include, or null if at
    /// the top level.
    SMLoc IncludeLoc;
  };
  
  /// Buffers - This is all of the buffers that we are reading from.
  std::vector<SrcBuffer> Buffers;
  
  TGSourceMgr(const TGSourceMgr&);    // DO NOT IMPLEMENT
  void operator=(const TGSourceMgr&); // DO NOT IMPLEMENT
public:
  TGSourceMgr() {}
  ~TGSourceMgr();
  
  const SrcBuffer &getBufferInfo(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i];
  }

  const MemoryBuffer *getMemoryBuffer(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i].Buffer;
  }
  
  SMLoc getParentIncludeLoc(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i].IncludeLoc;
  }
  
  unsigned AddNewSourceBuffer(MemoryBuffer *F, SMLoc IncludeLoc) {
    SrcBuffer NB;
    NB.Buffer = F;
    NB.IncludeLoc = IncludeLoc;
    Buffers.push_back(NB);
    return Buffers.size()-1;
  }
  
  /// FindBufferContainingLoc - Return the ID of the buffer containing the
  /// specified location, returning -1 if not found.
  int FindBufferContainingLoc(SMLoc Loc) const;
  
  /// FindLineNumber - Find the line number for the specified location in the
  /// specified file.  This is not a fast method.
  unsigned FindLineNumber(SMLoc Loc, int BufferID = -1) const;
  
  
  /// PrintError - Emit an error message about the specified location with the
  /// specified string.
  void PrintError(SMLoc ErrorLoc, const std::string &Msg) const;
  
private:
  void PrintIncludeStack(SMLoc IncludeLoc) const;
};
  
}  // end llvm namespace

#endif
