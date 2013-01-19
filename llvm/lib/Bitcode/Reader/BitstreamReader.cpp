//===- BitstreamReader.cpp - BitstreamReader implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitstreamReader.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//  BitstreamCursor implementation
//===----------------------------------------------------------------------===//

void BitstreamCursor::operator=(const BitstreamCursor &RHS) {
  freeState();
  
  BitStream = RHS.BitStream;
  NextChar = RHS.NextChar;
  CurWord = RHS.CurWord;
  BitsInCurWord = RHS.BitsInCurWord;
  CurCodeSize = RHS.CurCodeSize;
  
  // Copy abbreviations, and bump ref counts.
  CurAbbrevs = RHS.CurAbbrevs;
  for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
       i != e; ++i)
    CurAbbrevs[i]->addRef();
  
  // Copy block scope and bump ref counts.
  BlockScope = RHS.BlockScope;
  for (unsigned S = 0, e = static_cast<unsigned>(BlockScope.size());
       S != e; ++S) {
    std::vector<BitCodeAbbrev*> &Abbrevs = BlockScope[S].PrevAbbrevs;
    for (unsigned i = 0, e = static_cast<unsigned>(Abbrevs.size());
         i != e; ++i)
      Abbrevs[i]->addRef();
  }
}

void BitstreamCursor::freeState() {
  // Free all the Abbrevs.
  for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
       i != e; ++i)
    CurAbbrevs[i]->dropRef();
  CurAbbrevs.clear();
  
  // Free all the Abbrevs in the block scope.
  for (unsigned S = 0, e = static_cast<unsigned>(BlockScope.size());
       S != e; ++S) {
    std::vector<BitCodeAbbrev*> &Abbrevs = BlockScope[S].PrevAbbrevs;
    for (unsigned i = 0, e = static_cast<unsigned>(Abbrevs.size());
         i != e; ++i)
      Abbrevs[i]->dropRef();
  }
  BlockScope.clear();
}

/// EnterSubBlock - Having read the ENTER_SUBBLOCK abbrevid, enter
/// the block, and return true if the block has an error.
bool BitstreamCursor::EnterSubBlock(unsigned BlockID, unsigned *NumWordsP) {
  // Save the current block's state on BlockScope.
  BlockScope.push_back(Block(CurCodeSize));
  BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
  
  // Add the abbrevs specific to this block to the CurAbbrevs list.
  if (const BitstreamReader::BlockInfo *Info =
      BitStream->getBlockInfo(BlockID)) {
    for (unsigned i = 0, e = static_cast<unsigned>(Info->Abbrevs.size());
         i != e; ++i) {
      CurAbbrevs.push_back(Info->Abbrevs[i]);
      CurAbbrevs.back()->addRef();
    }
  }
  
  // Get the codesize of this block.
  CurCodeSize = ReadVBR(bitc::CodeLenWidth);
  SkipToWord();
  unsigned NumWords = Read(bitc::BlockSizeWidth);
  if (NumWordsP) *NumWordsP = NumWords;
  
  // Validate that this block is sane.
  if (CurCodeSize == 0 || AtEndOfStream())
    return true;
  
  return false;
}


unsigned BitstreamCursor::ReadRecord(unsigned AbbrevID,
                                     SmallVectorImpl<uint64_t> &Vals,
                                     const char **BlobStart, unsigned *BlobLen){
  if (AbbrevID == bitc::UNABBREV_RECORD) {
    unsigned Code = ReadVBR(6);
    unsigned NumElts = ReadVBR(6);
    for (unsigned i = 0; i != NumElts; ++i)
      Vals.push_back(ReadVBR64(6));
    return Code;
  }
  
  const BitCodeAbbrev *Abbv = getAbbrev(AbbrevID);
  
  for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
    const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
    if (Op.isLiteral()) {
      ReadAbbreviatedLiteral(Op, Vals);
      continue;
    }
    
    if (Op.getEncoding() != BitCodeAbbrevOp::Array &&
        Op.getEncoding() != BitCodeAbbrevOp::Blob) {
      ReadAbbreviatedField(Op, Vals);
      continue;
    }
    
    if (Op.getEncoding() == BitCodeAbbrevOp::Array) {
      // Array case.  Read the number of elements as a vbr6.
      unsigned NumElts = ReadVBR(6);
      
      // Get the element encoding.
      assert(i+2 == e && "array op not second to last?");
      const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);
      
      // Read all the elements.
      for (; NumElts; --NumElts)
        ReadAbbreviatedField(EltEnc, Vals);
      continue;
    }
    
    assert(Op.getEncoding() == BitCodeAbbrevOp::Blob);
    // Blob case.  Read the number of bytes as a vbr6.
    unsigned NumElts = ReadVBR(6);
    SkipToWord();  // 32-bit alignment
    
    // Figure out where the end of this blob will be including tail padding.
    size_t NewEnd = NextChar+((NumElts+3)&~3);
    
    // If this would read off the end of the bitcode file, just set the
    // record to empty and return.
    if (!canSkipToPos(NewEnd)) {
      Vals.append(NumElts, 0);
      NextChar = BitStream->getBitcodeBytes().getExtent();
      break;
    }
    
    // Otherwise, read the number of bytes.  If we can return a reference to
    // the data, do so to avoid copying it.
    if (BlobStart) {
      *BlobStart = (const char*)BitStream->getBitcodeBytes().getPointer(
                                                            NextChar, NumElts);
      *BlobLen = NumElts;
    } else {
      for (; NumElts; ++NextChar, --NumElts)
        Vals.push_back(getByte(NextChar));
    }
    // Skip over tail padding.
    NextChar = NewEnd;
  }
  
  unsigned Code = (unsigned)Vals[0];
  Vals.erase(Vals.begin());
  return Code;
}


void BitstreamCursor::ReadAbbrevRecord() {
  BitCodeAbbrev *Abbv = new BitCodeAbbrev();
  unsigned NumOpInfo = ReadVBR(5);
  for (unsigned i = 0; i != NumOpInfo; ++i) {
    bool IsLiteral = Read(1) ? true : false;
    if (IsLiteral) {
      Abbv->Add(BitCodeAbbrevOp(ReadVBR64(8)));
      continue;
    }
    
    BitCodeAbbrevOp::Encoding E = (BitCodeAbbrevOp::Encoding)Read(3);
    if (BitCodeAbbrevOp::hasEncodingData(E))
      Abbv->Add(BitCodeAbbrevOp(E, ReadVBR64(5)));
    else
      Abbv->Add(BitCodeAbbrevOp(E));
  }
  CurAbbrevs.push_back(Abbv);
}

bool BitstreamCursor::ReadBlockInfoBlock() {
  // If this is the second stream to get to the block info block, skip it.
  if (BitStream->hasBlockInfoRecords())
    return SkipBlock();
  
  if (EnterSubBlock(bitc::BLOCKINFO_BLOCK_ID)) return true;
  
  SmallVector<uint64_t, 64> Record;
  BitstreamReader::BlockInfo *CurBlockInfo = 0;
  
  // Read all the records for this module.
  while (1) {
    unsigned Code = ReadCode();
    if (Code == bitc::END_BLOCK)
      return ReadBlockEnd();
    if (Code == bitc::ENTER_SUBBLOCK) {
      ReadSubBlockID();
      if (SkipBlock()) return true;
      continue;
    }
    
    // Read abbrev records, associate them with CurBID.
    if (Code == bitc::DEFINE_ABBREV) {
      if (!CurBlockInfo) return true;
      ReadAbbrevRecord();
      
      // ReadAbbrevRecord installs the abbrev in CurAbbrevs.  Move it to the
      // appropriate BlockInfo.
      BitCodeAbbrev *Abbv = CurAbbrevs.back();
      CurAbbrevs.pop_back();
      CurBlockInfo->Abbrevs.push_back(Abbv);
      continue;
    }
    
    // Read a record.
    Record.clear();
    switch (ReadRecord(Code, Record)) {
      default: break;  // Default behavior, ignore unknown content.
      case bitc::BLOCKINFO_CODE_SETBID:
        if (Record.size() < 1) return true;
        CurBlockInfo = &BitStream->getOrCreateBlockInfo((unsigned)Record[0]);
        break;
      case bitc::BLOCKINFO_CODE_BLOCKNAME: {
        if (!CurBlockInfo) return true;
        if (BitStream->isIgnoringBlockInfoNames()) break;  // Ignore name.
        std::string Name;
        for (unsigned i = 0, e = Record.size(); i != e; ++i)
          Name += (char)Record[i];
        CurBlockInfo->Name = Name;
        break;
      }
      case bitc::BLOCKINFO_CODE_SETRECORDNAME: {
        if (!CurBlockInfo) return true;
        if (BitStream->isIgnoringBlockInfoNames()) break;  // Ignore name.
        std::string Name;
        for (unsigned i = 1, e = Record.size(); i != e; ++i)
          Name += (char)Record[i];
        CurBlockInfo->RecordNames.push_back(std::make_pair((unsigned)Record[0],
                                                           Name));
        break;
      }
    }
  }
}


