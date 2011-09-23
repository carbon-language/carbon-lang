//===--- APValue.cpp - Union class for APFloat/APSInt/Complex -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the APValue class.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"
#include "clang/AST/CharUnits.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

namespace {
  struct LV {
    const Expr* Base;
    CharUnits Offset;
  };
}

APValue::APValue(const Expr* B) : Kind(Uninitialized) {
  MakeLValue(); setLValue(B, CharUnits::Zero());
}

const APValue &APValue::operator=(const APValue &RHS) {
  if (Kind != RHS.Kind) {
    MakeUninit();
    if (RHS.isInt())
      MakeInt();
    else if (RHS.isFloat())
      MakeFloat();
    else if (RHS.isVector())
      MakeVector();
    else if (RHS.isComplexInt())
      MakeComplexInt();
    else if (RHS.isComplexFloat())
      MakeComplexFloat();
    else if (RHS.isLValue())
      MakeLValue();
  }
  if (isInt())
    setInt(RHS.getInt());
  else if (isFloat())
    setFloat(RHS.getFloat());
  else if (isVector())
    setVector(((const Vec *)(const char *)RHS.Data)->Elts,
              RHS.getVectorLength());
  else if (isComplexInt())
    setComplexInt(RHS.getComplexIntReal(), RHS.getComplexIntImag());
  else if (isComplexFloat())
    setComplexFloat(RHS.getComplexFloatReal(), RHS.getComplexFloatImag());
  else if (isLValue())
    setLValue(RHS.getLValueBase(), RHS.getLValueOffset());
  return *this;
}

void APValue::MakeUninit() {
  if (Kind == Int)
    ((APSInt*)(char*)Data)->~APSInt();
  else if (Kind == Float)
    ((APFloat*)(char*)Data)->~APFloat();
  else if (Kind == Vector)
    ((Vec*)(char*)Data)->~Vec();
  else if (Kind == ComplexInt)
    ((ComplexAPSInt*)(char*)Data)->~ComplexAPSInt();
  else if (Kind == ComplexFloat)
    ((ComplexAPFloat*)(char*)Data)->~ComplexAPFloat();
  else if (Kind == LValue) {
    ((LV*)(char*)Data)->~LV();
  }
  Kind = Uninitialized;
}

void APValue::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}

static double GetApproxValue(const llvm::APFloat &F) {
  llvm::APFloat V = F;
  bool ignored;
  V.convert(llvm::APFloat::IEEEdouble, llvm::APFloat::rmNearestTiesToEven,
            &ignored);
  return V.convertToDouble();
}

void APValue::print(raw_ostream &OS) const {
  switch (getKind()) {
  default: llvm_unreachable("Unknown APValue kind!");
  case Uninitialized:
    OS << "Uninitialized";
    return;
  case Int:
    OS << "Int: " << getInt();
    return;
  case Float:
    OS << "Float: " << GetApproxValue(getFloat());
    return;
  case Vector:
    OS << "Vector: " << getVectorElt(0);
    for (unsigned i = 1; i != getVectorLength(); ++i)
      OS << ", " << getVectorElt(i);
    return;
  case ComplexInt:
    OS << "ComplexInt: " << getComplexIntReal() << ", " << getComplexIntImag();
    return;
  case ComplexFloat:
    OS << "ComplexFloat: " << GetApproxValue(getComplexFloatReal())
       << ", " << GetApproxValue(getComplexFloatImag());
  case LValue:
    OS << "LValue: <todo>";
    return;
  }
}

static void WriteShortAPValueToStream(raw_ostream& Out,
                                      const APValue& V) {
  switch (V.getKind()) {
  default: llvm_unreachable("Unknown APValue kind!");
  case APValue::Uninitialized:
    Out << "Uninitialized";
    break;
  case APValue::Int:
    Out << V.getInt();
    break;
  case APValue::Float:
    Out << GetApproxValue(V.getFloat());
    break;
  case APValue::Vector:
    Out << '[';
    WriteShortAPValueToStream(Out, V.getVectorElt(0));
    for (unsigned i = 1; i != V.getVectorLength(); ++i) {
      Out << ", ";
      WriteShortAPValueToStream(Out, V.getVectorElt(i));
    }
    Out << ']';
    break;
  case APValue::ComplexInt:
    Out << V.getComplexIntReal() << "+" << V.getComplexIntImag() << "i";
    break;
  case APValue::ComplexFloat:
    Out << GetApproxValue(V.getComplexFloatReal()) << "+"
        << GetApproxValue(V.getComplexFloatImag()) << "i";
    break;
  case APValue::LValue:
    Out << "LValue: <todo>";
    break;
  }
}

const DiagnosticBuilder &clang::operator<<(const DiagnosticBuilder &DB,
                                           const APValue &V) {
  llvm::SmallString<64> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  WriteShortAPValueToStream(Out, V);
  return DB << Out.str();
}

const Expr* APValue::getLValueBase() const {
  assert(isLValue() && "Invalid accessor");
  return ((const LV*)(const void*)Data)->Base;
}

CharUnits APValue::getLValueOffset() const {
    assert(isLValue() && "Invalid accessor");
    return ((const LV*)(const void*)Data)->Offset;
}

void APValue::setLValue(const Expr *B, const CharUnits &O) {
  assert(isLValue() && "Invalid accessor");
  ((LV*)(char*)Data)->Base = B;
  ((LV*)(char*)Data)->Offset = O;
}

void APValue::MakeLValue() {
  assert(isUninit() && "Bad state change");
  new ((void*)(char*)Data) LV();
  Kind = LValue;
}

