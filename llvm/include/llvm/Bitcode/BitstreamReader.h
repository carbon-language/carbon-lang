//===- BitstreamReader.h - Low-level bitstream reader interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License.  See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitstreamReader class.  This class can be used to
// read an arbitrary bitstream, regardless of its contents.
//
//===----------------------------------------------------------------------===//

#ifndef BITSTREAM_READER_H
#define BITSTREAM_READER_H

#include "llvm/Bitcode/BitCodes.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace llvm {
  
class BitstreamReader {
  const unsigned char *NextChar;
  const unsigned char *LastChar;
  
  /// CurWord - This is the current data we have pulled from the stream but have
  /// not returned to the client.
  uint32_t CurWord;
  
  /// BitsInCurWord - This is the number of bits in CurWord that are valid. This
  /// is always from [0...31] inclusive.
  unsigned BitsInCurWord;
  
  // CurCodeSize - This is the declared size of code values used for the current
  // block, in bits.
  unsigned CurCodeSize;
  
  /// BlockScope - This tracks the codesize of parent blocks.
  SmallVector<unsigned, 8> BlockScope;
  
public:
  BitstreamReader(const unsigned char *Start, const unsigned char *End)
    : NextChar(Start), LastChar(End) {
    assert(((End-Start) & 3) == 0 &&"Bitcode stream not a multiple of 4 bytes");
    CurWord = 0;
    BitsInCurWord = 0;
    CurCodeSize = 2;
  }
  
  bool AtEndOfStream() const { return NextChar == LastChar; }
  
  uint32_t Read(unsigned NumBits) {
    // If the field is fully contained by CurWord, return it quickly.
    if (BitsInCurWord >= NumBits) {
      uint32_t R = CurWord & ((1U << NumBits)-1);
      CurWord >>= NumBits;
      BitsInCurWord -= NumBits;
      return R;
    }

    // If we run out of data, stop at the end of the stream.
    if (LastChar == NextChar) {
      CurWord = 0;
      BitsInCurWord = 0;
      return 0;
    }
    
    unsigned R = CurWord;

    // Read the next word from the stream.
    CurWord = (NextChar[0] <<  0) | (NextChar[1] << 8) |
              (NextChar[2] << 16) | (NextChar[3] << 24);
    NextChar += 4;
    
    // Extract NumBits-BitsInCurWord from what we just read.
    unsigned BitsLeft = NumBits-BitsInCurWord;
    
    // Be careful here, BitsLeft is in the range [1..32] inclusive.
    R |= (CurWord & (~0U >> (32-BitsLeft))) << BitsInCurWord;
    
    // BitsLeft bits have just been used up from CurWord.
    if (BitsLeft != 32)
      CurWord >>= BitsLeft;
    else
      CurWord = 0;
    BitsInCurWord = 32-BitsLeft;
    return R;
  }
  
  uint32_t ReadVBR(unsigned NumBits) {
    uint32_t Piece = Read(NumBits);
    if ((Piece & (1U << NumBits-1)) == 0)
      return Piece;

    uint32_t Result = 0;
    unsigned NextBit = 0;
    while (1) {
      Result |= (Piece & ((1U << (NumBits-1))-1)) << NextBit;

      if ((Piece & (1U << NumBits-1)) == 0)
        return Result;
      
      NextBit += NumBits-1;
      Piece = Read(NumBits);
    }
  }
  
  uint64_t ReadVBR64(unsigned NumBits) {
    uint64_t Piece = Read(NumBits);
    if ((Piece & (1U << NumBits-1)) == 0)
      return Piece;
    
    uint64_t Result = 0;
    unsigned NextBit = 0;
    while (1) {
      Result |= (Piece & ((1U << (NumBits-1))-1)) << NextBit;
      
      if ((Piece & (1U << NumBits-1)) == 0)
        return Result;
      
      NextBit += NumBits-1;
      Piece = Read(NumBits);
    }
  }

  void SkipToWord() {
    BitsInCurWord = 0;
    CurWord = 0;
  }

  
  unsigned ReadCode() {
    return Read(CurCodeSize);
  }

  //===--------------------------------------------------------------------===//
  // Block Manipulation
  //===--------------------------------------------------------------------===//
  
  // Block header:
  //    [ENTER_SUBBLOCK, blockid, newcodelen, <align4bytes>, blocklen]

  /// ReadSubBlockID - Having read the ENTER_SUBBLOCK code, read the BlockID for
  /// the block.
  unsigned ReadSubBlockID() {
    return ReadVBR(bitc::BlockIDWidth);
  }
  
  /// SkipBlock - Having read the ENTER_SUBBLOCK abbrevid and a BlockID, skip
  /// over the body of this block.  If the block record is malformed, return
  /// true.
  bool SkipBlock() {
    // Read and ignore the codelen value.  Since we are skipping this block, we
    // don't care what code widths are used inside of it.
    ReadVBR(bitc::CodeLenWidth);
    SkipToWord();
    unsigned NumWords = Read(bitc::BlockSizeWidth);
    
    // Check that the block wasn't partially defined, and that the offset isn't
    // bogus.
    if (AtEndOfStream() || NextChar+NumWords*4 > LastChar)
      return true;
    
    NextChar += NumWords*4;
    return false;
  }
  
  /// EnterSubBlock - Having read the ENTER_SUBBLOCK abbrevid, read and enter
  /// the block, returning the BlockID of the block we just entered.
  bool EnterSubBlock() {
    BlockScope.push_back(CurCodeSize);
    
    // Get the codesize of this block.
    CurCodeSize = ReadVBR(bitc::CodeLenWidth);
    SkipToWord();
    unsigned NumWords = Read(bitc::BlockSizeWidth);
    
    // Validate that this block is sane.
    if (CurCodeSize == 0 || AtEndOfStream() || NextChar+NumWords*4 > LastChar)
      return true;
    
    return false;
  }
  
  bool ReadBlockEnd() {
    if (BlockScope.empty()) return true;
    
    // Block tail:
    //    [END_BLOCK, <align4bytes>]
    SkipToWord();
    CurCodeSize = BlockScope.back();
    BlockScope.pop_back();
    return false;
  }
  
  //===--------------------------------------------------------------------===//
  // Record Processing
  //===--------------------------------------------------------------------===//
  
  unsigned ReadRecord(unsigned AbbrevID, SmallVectorImpl<uint64_t> &Vals) {
    if (AbbrevID == bitc::UNABBREV_RECORD) {
      unsigned Code = ReadVBR(6);
      unsigned NumElts = ReadVBR(6);
      for (unsigned i = 0; i != NumElts; ++i)
        Vals.push_back(ReadVBR64(6));
      return Code;
    }
    
    assert(0 && "Reading with abbrevs not implemented!");
  }
  
};

} // End llvm namespace

#endif

    