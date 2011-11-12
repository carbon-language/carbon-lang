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
  struct LVBase {
    APValue::LValueBase Base;
    CharUnits Offset;
    unsigned PathLength;
  };
}

struct APValue::LV : LVBase {
  static const unsigned InlinePathSpace =
      (MaxSize - sizeof(LVBase)) / sizeof(LValuePathEntry);

  /// Path - The sequence of base classes, fields and array indices to follow to
  /// walk from Base to the subobject. When performing GCC-style folding, there
  /// may not be such a path.
  union {
    LValuePathEntry Path[InlinePathSpace];
    LValuePathEntry *PathPtr;
  };

  LV() { PathLength = (unsigned)-1; }
  ~LV() { if (hasPathPtr()) delete [] PathPtr; }

  void allocPath() {
    if (hasPathPtr()) PathPtr = new LValuePathEntry[PathLength];
  }
  void freePath() { if (hasPathPtr()) delete [] PathPtr; }

  bool hasPath() const { return PathLength != (unsigned)-1; }
  bool hasPathPtr() const { return hasPath() && PathLength > InlinePathSpace; }

  LValuePathEntry *getPath() { return hasPathPtr() ? PathPtr : Path; }
  const LValuePathEntry *getPath() const {
    return hasPathPtr() ? PathPtr : Path;
  }
};

// FIXME: Reduce the malloc traffic here.

APValue::Arr::Arr(unsigned NumElts, unsigned Size) :
  Elts(new APValue[NumElts + (NumElts != Size ? 1 : 0)]),
  NumElts(NumElts), ArrSize(Size) {}
APValue::Arr::~Arr() { delete [] Elts; }

APValue::StructData::StructData(unsigned NumBases, unsigned NumFields) :
  Elts(new APValue[NumBases+NumFields]),
  NumBases(NumBases), NumFields(NumFields) {}
APValue::StructData::~StructData() {
  delete [] Elts;
}

APValue::UnionData::UnionData() : Field(0), Value(new APValue) {}
APValue::UnionData::~UnionData () {
  delete Value;
}

const APValue &APValue::operator=(const APValue &RHS) {
  if (this == &RHS)
    return *this;
  if (Kind != RHS.Kind || Kind == Array || Kind == Struct) {
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
    else if (RHS.isArray())
      MakeArray(RHS.getArrayInitializedElts(), RHS.getArraySize());
    else if (RHS.isStruct())
      MakeStruct(RHS.getStructNumBases(), RHS.getStructNumFields());
    else if (RHS.isUnion())
      MakeUnion();
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
  else if (isLValue()) {
    if (RHS.hasLValuePath())
      setLValue(RHS.getLValueBase(), RHS.getLValueOffset(),RHS.getLValuePath());
    else
      setLValue(RHS.getLValueBase(), RHS.getLValueOffset(), NoLValuePath());
  } else if (isArray()) {
    for (unsigned I = 0, N = RHS.getArrayInitializedElts(); I != N; ++I)
      getArrayInitializedElt(I) = RHS.getArrayInitializedElt(I);
    if (RHS.hasArrayFiller())
      getArrayFiller() = RHS.getArrayFiller();
  } else if (isStruct()) {
    for (unsigned I = 0, N = RHS.getStructNumBases(); I != N; ++I)
      getStructBase(I) = RHS.getStructBase(I);
    for (unsigned I = 0, N = RHS.getStructNumFields(); I != N; ++I)
      getStructField(I) = RHS.getStructField(I);
  } else if (isUnion())
    setUnion(RHS.getUnionField(), RHS.getUnionValue());
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
  else if (Kind == LValue)
    ((LV*)(char*)Data)->~LV();
  else if (Kind == Array)
    ((Arr*)(char*)Data)->~Arr();
  else if (Kind == Struct)
    ((StructData*)(char*)Data)->~StructData();
  else if (Kind == Union)
    ((UnionData*)(char*)Data)->~UnionData();
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
    return;
  case LValue:
    OS << "LValue: <todo>";
    return;
  case Array:
    OS << "Array: ";
    for (unsigned I = 0, N = getArrayInitializedElts(); I != N; ++I) {
      OS << getArrayInitializedElt(I);
      if (I != getArraySize() - 1) OS << ", ";
    }
    if (hasArrayFiller())
      OS << getArraySize() - getArrayInitializedElts() << " x "
         << getArrayFiller();
    return;
  case Struct:
    OS << "Struct ";
    if (unsigned N = getStructNumBases()) {
      OS << " bases: " << getStructBase(0);
      for (unsigned I = 1; I != N; ++I)
        OS << ", " << getStructBase(I);
    }
    if (unsigned N = getStructNumFields()) {
      OS << " fields: " << getStructField(0);
      for (unsigned I = 1; I != N; ++I)
        OS << ", " << getStructField(I);
    }
    return;
  case Union:
    OS << "Union: " << getUnionValue();
    return;
  }
  llvm_unreachable("Unknown APValue kind!");
}

static void WriteShortAPValueToStream(raw_ostream& Out,
                                      const APValue& V) {
  switch (V.getKind()) {
  case APValue::Uninitialized:
    Out << "Uninitialized";
    return;
  case APValue::Int:
    Out << V.getInt();
    return;
  case APValue::Float:
    Out << GetApproxValue(V.getFloat());
    return;
  case APValue::Vector:
    Out << '[';
    WriteShortAPValueToStream(Out, V.getVectorElt(0));
    for (unsigned i = 1; i != V.getVectorLength(); ++i) {
      Out << ", ";
      WriteShortAPValueToStream(Out, V.getVectorElt(i));
    }
    Out << ']';
    return;
  case APValue::ComplexInt:
    Out << V.getComplexIntReal() << "+" << V.getComplexIntImag() << "i";
    return;
  case APValue::ComplexFloat:
    Out << GetApproxValue(V.getComplexFloatReal()) << "+"
        << GetApproxValue(V.getComplexFloatImag()) << "i";
    return;
  case APValue::LValue:
    Out << "LValue: <todo>";
    return;
  case APValue::Array:
    Out << '{';
    if (unsigned N = V.getArrayInitializedElts()) {
      Out << V.getArrayInitializedElt(0);
      for (unsigned I = 1; I != N; ++I)
        Out << ", " << V.getArrayInitializedElt(I);
    }
    Out << '}';
    return;
  case APValue::Struct:
    Out << '{';
    if (unsigned N = V.getStructNumBases()) {
      Out << V.getStructBase(0);
      for (unsigned I = 1; I != N; ++I)
        Out << ", " << V.getStructBase(I);
      if (V.getStructNumFields())
        Out << ", ";
    }
    if (unsigned N = V.getStructNumFields()) {
      Out << V.getStructField(0);
      for (unsigned I = 1; I != N; ++I)
        Out << ", " << V.getStructField(I);
    }
    Out << '}';
    return;
  case APValue::Union:
    Out << '{' << V.getUnionValue() << '}';
    return;
  }
  llvm_unreachable("Unknown APValue kind!");
}

const DiagnosticBuilder &clang::operator<<(const DiagnosticBuilder &DB,
                                           const APValue &V) {
  llvm::SmallString<64> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  WriteShortAPValueToStream(Out, V);
  return DB << Out.str();
}

const APValue::LValueBase APValue::getLValueBase() const {
  assert(isLValue() && "Invalid accessor");
  return ((const LV*)(const void*)Data)->Base;
}

CharUnits &APValue::getLValueOffset() {
  assert(isLValue() && "Invalid accessor");
  return ((LV*)(void*)Data)->Offset;
}

bool APValue::hasLValuePath() const {
  assert(isLValue() && "Invalid accessor");
  return ((const LV*)(const char*)Data)->hasPath();
}

ArrayRef<APValue::LValuePathEntry> APValue::getLValuePath() const {
  assert(isLValue() && hasLValuePath() && "Invalid accessor");
  const LV &LVal = *((const LV*)(const char*)Data);
  return ArrayRef<LValuePathEntry>(LVal.getPath(), LVal.PathLength);
}

void APValue::setLValue(LValueBase B, const CharUnits &O, NoLValuePath) {
  assert(isLValue() && "Invalid accessor");
  LV &LVal = *((LV*)(char*)Data);
  LVal.freePath();
  LVal.Base = B;
  LVal.Offset = O;
  LVal.PathLength = (unsigned)-1;
}

void APValue::setLValue(LValueBase B, const CharUnits &O,
                        ArrayRef<LValuePathEntry> Path) {
  assert(isLValue() && "Invalid accessor");
  LV &LVal = *((LV*)(char*)Data);
  LVal.freePath();
  LVal.Base = B;
  LVal.Offset = O;
  LVal.PathLength = Path.size();
  LVal.allocPath();
  memcpy(LVal.getPath(), Path.data(), Path.size() * sizeof(LValuePathEntry));
}

void APValue::MakeLValue() {
  assert(isUninit() && "Bad state change");
  assert(sizeof(LV) <= MaxSize && "LV too big");
  new ((void*)(char*)Data) LV();
  Kind = LValue;
}

void APValue::MakeArray(unsigned InitElts, unsigned Size) {
  assert(isUninit() && "Bad state change");
  new ((void*)(char*)Data) Arr(InitElts, Size);
  Kind = Array;
}
