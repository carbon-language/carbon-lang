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
#include "llvm/Support/raw_ostream.h"
using namespace clang;


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
    setVector(((Vec*)(char*)RHS.Data)->Elts, RHS.getVectorLength());
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

void APValue::print(llvm::raw_ostream &OS) const {
  switch (getKind()) {
  default: assert(0 && "Unknown APValue kind!");
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

