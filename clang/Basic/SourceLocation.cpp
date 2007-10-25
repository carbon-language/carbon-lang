//==--- SourceLocation.cpp - Compact identifier for Source Files -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines serialization methods for the SourceLocation class.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceLocation.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using llvm::Serializer;
using llvm::Deserializer;
using llvm::SerializeTrait;
using namespace clang;

void SerializeTrait<SourceLocation>::Emit(Serializer& S, SourceLocation L) {
  // FIXME: Add code for abbreviation.
  S.EmitInt(L.getRawEncoding());  
}

SourceLocation SerializeTrait<SourceLocation>::ReadVal(Deserializer& D) {
  return SourceLocation::getFromRawEncoding(D.ReadInt());   
}
