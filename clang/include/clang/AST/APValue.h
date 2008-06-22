//===--- APValue.h - Union class for APFloat/APInt/Complex ------*- C++ -*-===//
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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"

namespace clang {

/// APValue - This class implements a discriminated union of [uninitialized]
/// [APInt] [APFloat], [Complex APInt] [Complex APFloat].
class APValue {
  typedef llvm::APInt APInt;
  typedef llvm::APFloat APFloat;
public:
  enum ValueKind {
    Uninitialized,
    Int,
    Float,
    ComplexInt,
    ComplexFloat
  };
private:
  ValueKind Kind;
  
  struct ComplexAPInt { APInt Real, Imag; };
  struct ComplexAPFloat {
    APFloat Real, Imag;
    ComplexAPFloat() : Real(0.0), Imag(0.0) {}
  };
  
  enum {
    MaxSize = (sizeof(ComplexAPInt) > sizeof(ComplexAPFloat) ? 
               sizeof(ComplexAPInt) : sizeof(ComplexAPFloat))
  };
  
  /// Data - space for the largest member in units of void*.  This is an effort
  /// to ensure that the APInt/APFloat values have proper alignment.
  void *Data[(MaxSize+sizeof(void*)-1)/sizeof(void*)];
  
public:
  APValue() : Kind(Uninitialized) {}
  explicit APValue(const APInt &I) : Kind(Uninitialized) {
    MakeInt(); setInt(I);
  }
  explicit APValue(const APFloat &F) : Kind(Uninitialized) {
    MakeFloat(); setFloat(F);
  }
  APValue(const APInt &R, const APInt &I) : Kind(Uninitialized) {
    MakeComplexInt(); setComplexInt(R, I);
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
  bool isInt() const { return Kind == Int; }
  bool isFloat() const { return Kind == Float; }
  bool isComplexInt() const { return Kind == ComplexInt; }
  bool isComplexFloat() const { return Kind == ComplexFloat; }
  
  const APInt &getInt() const {
    assert(isInt() && "Invalid accessor");
    return *(const APInt*)(const void*)Data;
  }
  const APFloat &getFloat() const {
    assert(isFloat() && "Invalid accessor");
    return *(const APFloat*)(const void*)Data;
  }
  const APInt &getComplexIntReal() const {
    assert(isComplexInt() && "Invalid accessor");
    return ((const ComplexAPInt*)(const void*)Data)->Real;
  }
  const APInt &getComplexIntImag() const {
    assert(isComplexInt() && "Invalid accessor");
    return ((const ComplexAPInt*)(const void*)Data)->Imag;
  }
  const APFloat &getComplexFloatReal() const {
    assert(isComplexFloat() && "Invalid accessor");
    return ((const ComplexAPFloat*)(const void*)Data)->Real;
  }
  const APFloat &getComplexFloatImag() const {
    assert(isComplexFloat() && "Invalid accessor");
    return ((const ComplexAPFloat*)(const void*)Data)->Imag;
  }
  
  void setInt(const APInt &I) {
    assert(isInt() && "Invalid accessor");
    *(APInt*)(void*)Data = I;
  }
  void setFloat(const APFloat &F) {
    assert(isFloat() && "Invalid accessor");
    *(APFloat*)(void*)Data = F;
  }
  void setComplexInt(const APInt &R, const APInt &I) {
    assert(isComplexInt() && "Invalid accessor");
    ((ComplexAPInt*)(void*)Data)->Real = R;
    ((ComplexAPInt*)(void*)Data)->Imag = I;
  }
  void setComplexFloat(const APFloat &R, const APFloat &I) {
    assert(isComplexFloat() && "Invalid accessor");
    ((ComplexAPFloat*)(void*)Data)->Real = R;
    ((ComplexAPFloat*)(void*)Data)->Imag = I;
  }
  
  const APValue &operator=(const APValue &RHS) {
    if (Kind != RHS.Kind) {
      MakeUninit();
      if (RHS.isInt())
        MakeInt();
      else if (RHS.isFloat())
        MakeFloat();
      else if (RHS.isComplexInt())
        MakeComplexInt();
      else if (RHS.isComplexFloat())
        MakeComplexFloat();
    }
    if (isInt())
      setInt(RHS.getInt());
    else if (isFloat())
      setFloat(RHS.getFloat());
    else if (isComplexInt())
      setComplexInt(RHS.getComplexIntReal(), RHS.getComplexIntImag());
    else if (isComplexFloat())
      setComplexFloat(RHS.getComplexFloatReal(), RHS.getComplexFloatImag());
    return *this;
  }
  
private:
  void MakeUninit() {
    if (Kind == Int)
      ((APInt*)(void*)Data)->~APInt();
    else if (Kind == Float)
      ((APFloat*)(void*)Data)->~APFloat();
    else if (Kind == ComplexInt)
      ((ComplexAPInt*)(void*)Data)->~ComplexAPInt();
    else if (Kind == ComplexFloat)
      ((ComplexAPFloat*)(void*)Data)->~ComplexAPFloat();
  }
  void MakeInt() {
    assert(isUninit() && "Bad state change");
    new ((void*)Data) APInt();
    Kind = Int;
  }
  void MakeFloat() {
    assert(isUninit() && "Bad state change");
    new ((APFloat*)(void*)Data) APFloat(0.0);
    Kind = Float;
  }
  void MakeComplexInt() {
    assert(isUninit() && "Bad state change");
    new ((ComplexAPInt*)(void*)Data) ComplexAPInt();
    Kind = ComplexInt;
  }
  void MakeComplexFloat() {
    assert(isUninit() && "Bad state change");
    new ((ComplexAPFloat*)(void*)Data) ComplexAPFloat();
    Kind = ComplexFloat;
  }
};

}

#endif
