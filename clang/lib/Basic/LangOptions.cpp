//===--- LangOptions.cpp - Language feature info --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the methods for LangOptions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/LangOptions.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace clang;

void LangOptions::Emit(llvm::Serializer& S) const {
  S.EmitBool((bool) Trigraphs);
  S.EmitBool((bool) BCPLComment);
  S.EmitBool((bool) DollarIdents);
  S.EmitBool((bool) Digraphs);
  S.EmitBool((bool) HexFloats);
  S.EmitBool((bool) C99);
  S.EmitBool((bool) Microsoft);
  S.EmitBool((bool) CPlusPlus);
  S.EmitBool((bool) CPlusPlus0x);
  S.EmitBool((bool) NoExtensions);
  S.EmitBool((bool) CXXOperatorNames);             
  S.EmitBool((bool) ObjC1);
  S.EmitBool((bool) ObjC2);
  S.EmitBool((unsigned) GC);
  S.EmitBool((bool) PascalStrings);
  S.EmitBool((bool) Boolean);
  S.EmitBool((bool) WritableStrings);
  S.EmitBool((bool) LaxVectorConversions);
}

void LangOptions::Read(llvm::Deserializer& D) {
  Trigraphs = D.ReadBool() ? 1 : 0;
  BCPLComment = D.ReadBool() ? 1 : 0;
  DollarIdents = D.ReadBool() ? 1 : 0;
  Digraphs = D.ReadBool() ? 1 : 0;
  HexFloats = D.ReadBool() ? 1 : 0;
  C99 = D.ReadBool() ? 1 : 0;
  Microsoft = D.ReadBool() ? 1 : 0;
  CPlusPlus = D.ReadBool() ? 1 : 0;
  CPlusPlus0x = D.ReadBool() ? 1 : 0;
  NoExtensions = D.ReadBool() ? 1 : 0;
  CXXOperatorNames = D.ReadBool() ? 1 : 0;
  ObjC1 = D.ReadBool() ? 1 : 0;
  ObjC2 = D.ReadBool() ? 1 : 0;
  GC = D.ReadInt();
  PascalStrings = D.ReadBool() ? 1 : 0;
  Boolean = D.ReadBool() ? 1 : 0;
  WritableStrings = D.ReadBool() ? 1 : 0;
  LaxVectorConversions = D.ReadBool() ? 1 : 0;
}
