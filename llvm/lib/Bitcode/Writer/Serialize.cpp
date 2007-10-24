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

#include "llvm/Bitcode/Serialize.h"
#include "string.h"

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

void Serializer::EmitInt(unsigned X) {
  Record.push_back(X);
}

void Serializer::EmitCStr(const char* s, const char* end) {
  Record.push_back(end - s);
  
  while(s != end) {
    Record.push_back(*s);
    ++s;
  }

  EmitRecord();
}

void Serializer::EmitCStr(const char* s) {
  EmitCStr(s,s+strlen(s));
}

#define INT_EMIT(TYPE)\
void SerializeTrait<TYPE>::Emit(Serializer&S, TYPE X) { S.EmitInt(X); }

INT_EMIT(bool)
INT_EMIT(unsigned char)
INT_EMIT(unsigned short)
INT_EMIT(unsigned int)
INT_EMIT(unsigned long)
INT_EMIT(unsigned long long)
