//===--- APValue.h - Union class for APFloat/APSInt/Complex -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the APValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_APVALUE_H
#define LLVM_CLANG_AST_APVALUE_H

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/APFloat.h"

namespace clang {
  class Expr;

/// APValue - This class implements a discriminated union of [uninitialized]
/// [APSInt] [APFloat], [Complex APSInt] [Complex APFloat], [Expr + Offset].
class APValue {
  typedef llvm::APSInt APSInt;
  typedef llvm::APFloat APFloat;
public:
  enum ValueKind {
    Uninitialized,
    SInt,
    Float,
    ComplexSInt,
    ComplexFloat,
    LValue
  };
private:
  ValueKind Kind;
  
  struct ComplexAPSInt { 
    APSInt Real, Imag; 
    ComplexAPSInt() : Real(1), Imag(1) {}
  };
  struct ComplexAPFloat {
    APFloat Real, Imag;
    ComplexAPFloat() : Real(0.0), Imag(0.0) {}
  };
  
  struct LValue {
    Expr* Base;
    uint64_t Offset;
  };
  
  enum {
    MaxSize = (sizeof(ComplexAPSInt) > sizeof(ComplexAPFloat) ? 
               sizeof(ComplexAPSInt) : sizeof(ComplexAPFloat))
  };
  
  /// Data - space for the largest member in units of void*.  This is an effort
  /// to ensure that the APSInt/APFloat values have proper alignment.
  void *Data[(MaxSize+sizeof(void*)-1)/sizeof(void*)];
  
public:
  APValue() : Kind(Uninitialized) {}
  explicit APValue(const APSInt &I) : Kind(Uninitialized) {
    MakeSInt(); setSInt(I);
  }
  explicit APValue(const APFloat &F) : Kind(Uninitialized) {
    MakeFloat(); setFloat(F);
  }
  APValue(const APSInt &R, const APSInt &I) : Kind(Uninitialized) {
    MakeComplexSInt(); setComplexSInt(R, I);
  }
  APValue(const APFloat &R, const APFloat &I) : Kind(Uninitialized) {
    MakeComplexFloat(); setComplexFloat(R, I);
  }
  APValue(const APValue &RHS) : Kind(Uninitialized) {
    *this = RHS;
  }
  ~APValue() {
    MakeUninit();
  }
  
  ValueKind getKind() const { return Kind; }
  bool isUninit() const { return Kind == Uninitialized; }
  bool isSInt() const { return Kind == SInt; }
  bool isFloat() const { return Kind == Float; }
  bool isComplexSInt() const { return Kind == ComplexSInt; }
  bool isComplexFloat() const { return Kind == ComplexFloat; }
  
  const APSInt &getSInt() const {
    assert(isSInt() && "Invalid accessor");
    return *(const APSInt*)(const void*)Data;
  }
  const APFloat &getFloat() const {
    assert(isFloat() && "Invalid accessor");
    return *(const APFloat*)(const void*)Data;
  }
  const APSInt &getComplexSIntReal() const {
    assert(isComplexSInt() && "Invalid accessor");
    return ((const ComplexAPSInt*)(const void*)Data)->Real;
  }
  const APSInt &getComplexSIntImag() const {
    assert(isComplexSInt() && "Invalid accessor");
    return ((const ComplexAPSInt*)(const void*)Data)->Imag;
  }
  const APFloat &getComplexFloatReal() const {
    assert(isComplexFloat() && "Invalid accessor");
    return ((const ComplexAPFloat*)(const void*)Data)->Real;
  }
  const APFloat &getComplexFloatImag() const {
    assert(isComplexFloat() && "Invalid accessor");
    return ((const ComplexAPFloat*)(const void*)Data)->Imag;
  }
  
  void setSInt(const APSInt &I) {
    assert(isSInt() && "Invalid accessor");
    *(APSInt*)(void*)Data = I;
  }
  void setFloat(const APFloat &F) {
    assert(isFloat() && "Invalid accessor");
    *(APFloat*)(void*)Data = F;
  }
  void setComplexSInt(const APSInt &R, const APSInt &I) {
    assert(isComplexSInt() && "Invalid accessor");
    ((ComplexAPSInt*)(void*)Data)->Real = R;
    ((ComplexAPSInt*)(void*)Data)->Imag = I;
  }
  void setComplexFloat(const APFloat &R, const APFloat &I) {
    assert(isComplexFloat() && "Invalid accessor");
    ((ComplexAPFloat*)(void*)Data)->Real = R;
    ((ComplexAPFloat*)(void*)Data)->Imag = I;
  }
  
  const APValue &operator=(const APValue &RHS) {
    if (Kind != RHS.Kind) {
      MakeUninit();
      if (RHS.isSInt())
        MakeSInt();
      else if (RHS.isFloat())
        MakeFloat();
      else if (RHS.isComplexSInt())
        MakeComplexSInt();
      else if (RHS.isComplexFloat())
        MakeComplexFloat();
    }
    if (isSInt())
      setSInt(RHS.getSInt());
    else if (isFloat())
      setFloat(RHS.getFloat());
    else if (isComplexSInt())
      setComplexSInt(RHS.getComplexSIntReal(), RHS.getComplexSIntImag());
    else if (isComplexFloat())
      setComplexFloat(RHS.getComplexFloatReal(), RHS.getComplexFloatImag());
    return *this;
  }
  
private:
  void MakeUninit() {
    if (Kind == SInt)
      ((APSInt*)(void*)Data)->~APSInt();
    else if (Kind == Float)
      ((APFloat*)(void*)Data)->~APFloat();
    else if (Kind == ComplexSInt)
      ((ComplexAPSInt*)(void*)Data)->~ComplexAPSInt();
    else if (Kind == ComplexFloat)
      ((ComplexAPFloat*)(void*)Data)->~ComplexAPFloat();
  }
  void MakeSInt() {
    assert(isUninit() && "Bad state change");
    new ((void*)Data) APSInt(1);
    Kind = SInt;
  }
  void MakeFloat() {
    assert(isUninit() && "Bad state change");
    new ((APFloat*)(void*)Data) APFloat(0.0);
    Kind = Float;
  }
  void MakeComplexSInt() {
    assert(isUninit() && "Bad state change");
    new ((ComplexAPSInt*)(void*)Data) ComplexAPSInt();
    Kind = ComplexSInt;
  }
  void MakeComplexFloat() {
    assert(isUninit() && "Bad state change");
    new ((ComplexAPFloat*)(void*)Data) ComplexAPFloat();
    Kind = ComplexFloat;
  }
};

}

#endif
