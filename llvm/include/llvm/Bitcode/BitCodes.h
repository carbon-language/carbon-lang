//===- BitCodes.h - Enum values for the bitcode format ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header Bitcode enum values.
//
// The enum values defined in this file should be considered permanent.  If
// new features are added, they should have values added at the end of the
// respective lists.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_BITCODES_H
#define LLVM_BITCODE_BITCODES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {
namespace bitc {
  enum StandardWidths {
    BlockIDWidth = 8,  // We use VBR-8 for block IDs.
    CodeLenWidth = 4,  // Codelen are VBR-4.
    BlockSizeWidth = 32  // BlockSize up to 2^32 32-bit words = 32GB per block.
  };
  
  // The standard code namespace always has a way to exit a block, enter a
  // nested block, define abbrevs, and define an unabbreviated record.
  enum FixedCodes {
    END_BLOCK = 0,  // Must be zero to guarantee termination for broken bitcode.
    ENTER_SUBBLOCK = 1,

    /// DEFINE_ABBREV - Defines an abbrev for the current block.  It consists
    /// of a vbr5 for # operand infos.  Each operand info is emitted with a
    /// single bit to indicate if it is a literal encoding.  If so, the value is
    /// emitted with a vbr8.  If not, the encoding is emitted as 3 bits followed
    /// by the info value as a vbr5 if needed.
    DEFINE_ABBREV = 2, 
    
    // UNABBREV_RECORDs are emitted with a vbr6 for the record code, followed by
    // a vbr6 for the # operands, followed by vbr6's for each operand.
    UNABBREV_RECORD = 3,
    
    // This is not a code, this is a marker for the first abbrev assignment.
    FIRST_ABBREV = 4
  };
} // End bitc namespace

/// BitCodeAbbrevOp - This describes one or more operands in an abbreviation.
/// This is actually a union of two different things:
///   1. It could be a literal integer value ("the operand is always 17").
///   2. It could be an encoding specification ("this operand encoded like so").
///
class BitCodeAbbrevOp {
  uint64_t Val;           // A literal value or data for an encoding.
  bool IsLiteral : 1;     // Indicate whether this is a literal value or not.
  unsigned Enc   : 3;     // The encoding to use.
public:
  enum Encoding {
    FixedWidth = 1,   // A fixed with field, Val specifies number of bits.
    VBR        = 2   // A VBR field where Val specifies the width of each chunk.
  };
    
  BitCodeAbbrevOp(uint64_t V) :  Val(V), IsLiteral(true) {}
  BitCodeAbbrevOp(Encoding E, uint64_t Data)
    : Val(Data), IsLiteral(false), Enc(E) {}
  
  bool isLiteral() const { return IsLiteral; }
  bool isEncoding() const { return !IsLiteral; }

  // Accessors for literals.
  uint64_t getLiteralValue() const { assert(isLiteral()); return Val; }
  
  // Accessors for encoding info.
  Encoding getEncoding() const { assert(isEncoding()); return (Encoding)Enc; }
  uint64_t getEncodingData() const { assert(isEncoding()); return Val; }
  
  bool hasEncodingData() const { return hasEncodingData(getEncoding()); }
  static bool hasEncodingData(Encoding E) {
    return true; 
  }
};

class BitCodeAbbrev {
  SmallVector<BitCodeAbbrevOp, 8> OperandList;
  unsigned char RefCount; // Number of things using this.
  ~BitCodeAbbrev() {}
public:
  BitCodeAbbrev() : RefCount(1) {}
  
  void addRef() { ++RefCount; }
  void dropRef() { if (--RefCount == 0) delete this; }

  unsigned getNumOperandInfos() const { return OperandList.size(); }
  const BitCodeAbbrevOp &getOperandInfo(unsigned N) const {
    return OperandList[N];
  }
  
  void Add(const BitCodeAbbrevOp &OpInfo) {
    OperandList.push_back(OpInfo);
  }
};
} // End llvm namespace

#endif
