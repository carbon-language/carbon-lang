//===- BitstreamWriter.h - Low-level bitstream writer interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License.  See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitstreamWriter class.  This class can be used to
// write an arbitrary bitstream, regardless of its contents.
//
//===----------------------------------------------------------------------===//

#ifndef BITSTREAM_WRITER_H
#define BITSTREAM_WRITER_H

#include "llvm/Bitcode/BitCodes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <vector>

namespace llvm {

class BitstreamWriter {
  std::vector<unsigned char> &Out;

  /// CurBit - Always between 0 and 31 inclusive, specifies the next bit to use.
  unsigned CurBit;
  
  /// CurValue - The current value.  Only bits < CurBit are valid.
  uint32_t CurValue;
  
  // CurCodeSize - This is the declared size of code values used for the current
  // block, in bits.
  unsigned CurCodeSize;
  
  struct Block {
    unsigned PrevCodeSize;
    unsigned StartSizeWord;
    Block(unsigned PCS, unsigned SSW) : PrevCodeSize(PCS), StartSizeWord(SSW) {}
  };
  
  /// BlockScope - This tracks the current blocks that we have entered.
  std::vector<Block> BlockScope;
public:
  BitstreamWriter(std::vector<unsigned char> &O) 
    : Out(O), CurBit(0), CurValue(0), CurCodeSize(2) {}

  ~BitstreamWriter() {
    assert(CurBit == 0 && "Unflused data remaining");
    assert(BlockScope.empty() && "Block imbalance");
  }
  //===--------------------------------------------------------------------===//
  // Basic Primitives for emitting bits to the stream.
  //===--------------------------------------------------------------------===//
  
  void Emit(uint32_t Val, unsigned NumBits) {
    assert(NumBits <= 32 && "Invalid value size!");
    assert((Val & ~(~0U >> (32-NumBits))) == 0 && "High bits set!");
    CurValue |= Val << CurBit;
    if (CurBit + NumBits < 32) {
      CurBit += NumBits;
      return;
    }
    
    // Add the current word.
    unsigned V = CurValue;
    Out.push_back((unsigned char)(V >>  0));
    Out.push_back((unsigned char)(V >>  8));
    Out.push_back((unsigned char)(V >> 16));
    Out.push_back((unsigned char)(V >> 24));
    
    if (CurBit)
      CurValue = Val >> (32-CurBit);
    else
      CurValue = 0;
    CurBit = (CurBit+NumBits) & 31;
  }
  
  void Emit64(uint64_t Val, unsigned NumBits) {
    if (NumBits <= 32)
      Emit((uint32_t)Val, NumBits);
    else {
      Emit((uint32_t)Val, 32);
      Emit((uint32_t)(Val >> 32), NumBits-32);
    }
  }
  
  void FlushToWord() {
    if (CurBit) {
      unsigned V = CurValue;
      Out.push_back((unsigned char)(V >>  0));
      Out.push_back((unsigned char)(V >>  8));
      Out.push_back((unsigned char)(V >> 16));
      Out.push_back((unsigned char)(V >> 24));
      CurBit = 0;
      CurValue = 0;
    }
  }
  
  void EmitVBR(uint32_t Val, unsigned NumBits) {
    uint32_t Threshold = 1U << (NumBits-1);
    
    // Emit the bits with VBR encoding, NumBits-1 bits at a time.
    while (Val >= Threshold) {
      Emit((Val & ((1 << (NumBits-1))-1)) | (1 << (NumBits-1)), NumBits);
      Val >>= NumBits-1;
    }
    
    Emit(Val, NumBits);
  }
  
  void EmitVBR64(uint64_t Val, unsigned NumBits) {
    if ((uint32_t)Val == Val)
      return EmitVBR((uint32_t)Val, NumBits);
    
    uint64_t Threshold = 1U << (NumBits-1);
    
    // Emit the bits with VBR encoding, NumBits-1 bits at a time.
    while (Val >= Threshold) {
      Emit(((uint32_t)Val & ((1 << (NumBits-1))-1)) |
           (1 << (NumBits-1)), NumBits);
      Val >>= NumBits-1;
    }
    
    Emit((uint32_t)Val, NumBits);
  }
  
  /// EmitCode - Emit the specified code.
  void EmitCode(unsigned Val) {
    Emit(Val, CurCodeSize);
  }
  
  //===--------------------------------------------------------------------===//
  // Block Manipulation
  //===--------------------------------------------------------------------===//
  
  void EnterSubblock(unsigned BlockID, unsigned CodeLen) {
    // Block header:
    //    [ENTER_SUBBLOCK, blockid, newcodelen, <align4bytes>, blocklen]
    EmitCode(bitc::ENTER_SUBBLOCK);
    EmitVBR(BlockID, bitc::BlockIDWidth);
    EmitVBR(CodeLen, bitc::CodeLenWidth);
    FlushToWord();
    BlockScope.push_back(Block(CurCodeSize, Out.size()/4));
    // Emit a placeholder, which will be replaced when the block is popped.
    Emit(0, bitc::BlockSizeWidth);
    
    CurCodeSize = CodeLen;
  }
  
  void ExitBlock() {
    assert(!BlockScope.empty() && "Block scope imbalance!");
    Block B = BlockScope.back();
    BlockScope.pop_back();
    
    // Block tail:
    //    [END_BLOCK, <align4bytes>]
    EmitCode(bitc::END_BLOCK);
    FlushToWord();

    // Compute the size of the block, in words, not counting the size field.
    unsigned SizeInWords = Out.size()/4-B.StartSizeWord - 1;
    unsigned ByteNo = B.StartSizeWord*4;
    
    // Update the block size field in the header of this sub-block.
    Out[ByteNo++] = (unsigned char)(SizeInWords >>  0);
    Out[ByteNo++] = (unsigned char)(SizeInWords >>  8);
    Out[ByteNo++] = (unsigned char)(SizeInWords >> 16);
    Out[ByteNo++] = (unsigned char)(SizeInWords >> 24);
    
    // Restore the outer block's code size.
    CurCodeSize = B.PrevCodeSize;
  }
  
  //===--------------------------------------------------------------------===//
  // Record Emission
  //===--------------------------------------------------------------------===//
  
  /// EmitRecord - Emit the specified record to the stream, using an abbrev if
  /// we have one to compress the output.
  void EmitRecord(unsigned Code, SmallVectorImpl<uint64_t> &Vals,
                  unsigned Abbrev = 0) {
    if (Abbrev) {
      assert(0 && "abbrevs not implemented yet!");
    } else {
      // If we don't have an abbrev to use, emit this in its fully unabbreviated
      // form.
      EmitCode(bitc::UNABBREV_RECORD);
      EmitVBR(Code, 6);
      EmitVBR(Vals.size(), 6);
      for (unsigned i = 0, e = Vals.size(); i != e; ++i)
        EmitVBR64(Vals[i], 6);
    }
  }
  
  /// EmitRecord - Emit the specified record to the stream, using an abbrev if
  /// we have one to compress the output.
  void EmitRecord(unsigned Code, SmallVectorImpl<unsigned> &Vals,
                  unsigned Abbrev = 0) {
    if (Abbrev) {
      assert(0 && "abbrevs not implemented yet!");
    } else {
      // If we don't have an abbrev to use, emit this in its fully unabbreviated
      // form.
      EmitCode(bitc::UNABBREV_RECORD);
      EmitVBR(Code, 6);
      EmitVBR(Vals.size(), 6);
      for (unsigned i = 0, e = Vals.size(); i != e; ++i)
        EmitVBR(Vals[i], 6);
    }
  }
};


} // End llvm namespace

#endif
