//==- Serialize.cpp - Generic Object Serialization to Bitcode ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the internal methods used for object serialization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/Serialization.h"

using namespace llvm;

Serializer::Serializer(BitstreamWriter& stream, unsigned BlockID)
  : Stream(stream), inBlock(BlockID >= 8) {
    
  if (inBlock) Stream.EnterSubblock(8,3);
}

Serializer::~Serializer() {
  if (inRecord())
    EmitRecord();

  if (inBlock)
    Stream.ExitBlock();
  
  Stream.FlushToWord();
}

void Serializer::EmitRecord() {
  assert(Record.size() > 0 && "Cannot emit empty record.");
  Stream.EmitRecord(8,Record);
  Record.clear();
}

void Serializer::EmitInt(unsigned X, unsigned bits) {
  Record.push_back(X);
}

void Serializer::EmitCString(const char* cstr) {
  unsigned l = strlen(cstr);
  Record.push_back(l);
  
  for (unsigned i = 0; i < l; i++)
    Record.push_back(cstr[i]);

  EmitRecord();
}
