//==- Deserialize.cpp - Generic Object Serialization to Bitcode --*- C++ -*-==//
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

Deserializer::Deserializer(BitstreamReader& stream)
  : Stream(stream), RecIdx(0) {
}

Deserializer::~Deserializer() {
  assert (RecIdx >= Record.size() && 
          "Still scanning bitcode record when deserialization completed.");
}

void Deserializer::ReadRecord() {
  // FIXME: Check if we haven't run off the edge of the stream.
  // FIXME: Handle abbreviations.
  unsigned Code = Stream.ReadCode();
  // FIXME: Check for the correct code.
  assert (Record.size() == 0);
  
  Stream.ReadRecord(Code,Record);
  
  assert (Record.size() > 0);
}

uint64_t Deserializer::ReadInt(unsigned Bits) {
  // FIXME: Any error recovery/handling with incomplete or bad files?
  if (!inRecord())
    ReadRecord();

  // FIXME: check for loss of precision in read (compare to Bits)
  return Record[RecIdx++];
}

char* Deserializer::ReadCString(char* cstr, unsigned MaxLen, bool isNullTerm) {
  if (cstr == NULL)
    MaxLen = 0; // Zero this just in case someone does something funny.
  
  unsigned len = ReadInt(32);

  // FIXME: perform dynamic checking of lengths?
  assert (MaxLen == 0 || (len + (isNullTerm ? 1 : 0)) <= MaxLen);

  if (!cstr)
    cstr = new char[len + (isNullTerm ? 1 : 0)];
  
  assert (cstr != NULL);
  
  for (unsigned i = 0; i < len; ++i)
    cstr[i] = ReadInt(8);
  
  if (isNullTerm)
    cstr[len+1] = '\0';
  
  return cstr;
}

void Deserializer::ReadCString(std::vector<char>& buff, bool isNullTerm) {
  buff.clear();

  unsigned len = ReadInt(32);
  
  buff.reserve(len);
  
  for (unsigned i = 0; i < len; ++i)
    buff.push_back(ReadInt(8));
  
  if (isNullTerm)
    buff.push_back('\0');
}
