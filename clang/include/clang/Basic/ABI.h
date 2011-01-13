//===----- ABI.h - ABI related declarations ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These enums/classes describe ABI related information about constructors,
// destructors and thunks.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_BASIC_ABI_H
#define CLANG_BASIC_ABI_H

#include <stdint.h>

namespace clang {

/// CXXCtorType - C++ constructor types
enum CXXCtorType {
    Ctor_Complete,          // Complete object ctor
    Ctor_Base,              // Base object ctor
    Ctor_CompleteAllocating // Complete object allocating ctor
};

/// CXXDtorType - C++ destructor types
enum CXXDtorType {
    Dtor_Deleting, // Deleting dtor
    Dtor_Complete, // Complete object dtor
    Dtor_Base      // Base object dtor
};

/// ReturnAdjustment - A return adjustment.
struct ReturnAdjustment {
  /// NonVirtual - The non-virtual adjustment from the derived object to its
  /// nearest virtual base.
  int64_t NonVirtual;
  
  /// VBaseOffsetOffset - The offset (in bytes), relative to the address point 
  /// of the virtual base class offset.
  int64_t VBaseOffsetOffset;
  
  ReturnAdjustment() : NonVirtual(0), VBaseOffsetOffset(0) { }
  
  bool isEmpty() const { return !NonVirtual && !VBaseOffsetOffset; }

  friend bool operator==(const ReturnAdjustment &LHS, 
                         const ReturnAdjustment &RHS) {
    return LHS.NonVirtual == RHS.NonVirtual && 
      LHS.VBaseOffsetOffset == RHS.VBaseOffsetOffset;
  }

  friend bool operator<(const ReturnAdjustment &LHS,
                        const ReturnAdjustment &RHS) {
    if (LHS.NonVirtual < RHS.NonVirtual)
      return true;
    
    return LHS.NonVirtual == RHS.NonVirtual && 
      LHS.VBaseOffsetOffset < RHS.VBaseOffsetOffset;
  }
};
  
/// ThisAdjustment - A 'this' pointer adjustment.
struct ThisAdjustment {
  /// NonVirtual - The non-virtual adjustment from the derived object to its
  /// nearest virtual base.
  int64_t NonVirtual;

  /// VCallOffsetOffset - The offset (in bytes), relative to the address point,
  /// of the virtual call offset.
  int64_t VCallOffsetOffset;
  
  ThisAdjustment() : NonVirtual(0), VCallOffsetOffset(0) { }

  bool isEmpty() const { return !NonVirtual && !VCallOffsetOffset; }

  friend bool operator==(const ThisAdjustment &LHS, 
                         const ThisAdjustment &RHS) {
    return LHS.NonVirtual == RHS.NonVirtual && 
      LHS.VCallOffsetOffset == RHS.VCallOffsetOffset;
  }
  
  friend bool operator<(const ThisAdjustment &LHS,
                        const ThisAdjustment &RHS) {
    if (LHS.NonVirtual < RHS.NonVirtual)
      return true;
    
    return LHS.NonVirtual == RHS.NonVirtual && 
      LHS.VCallOffsetOffset < RHS.VCallOffsetOffset;
  }
};

/// ThunkInfo - The 'this' pointer adjustment as well as an optional return
/// adjustment for a thunk.
struct ThunkInfo {
  /// This - The 'this' pointer adjustment.
  ThisAdjustment This;
    
  /// Return - The return adjustment.
  ReturnAdjustment Return;

  ThunkInfo() { }

  ThunkInfo(const ThisAdjustment &This, const ReturnAdjustment &Return)
    : This(This), Return(Return) { }

  friend bool operator==(const ThunkInfo &LHS, const ThunkInfo &RHS) {
    return LHS.This == RHS.This && LHS.Return == RHS.Return;
  }

  friend bool operator<(const ThunkInfo &LHS, const ThunkInfo &RHS) {
    if (LHS.This < RHS.This)
      return true;
      
    return LHS.This == RHS.This && LHS.Return < RHS.Return;
  }

  bool isEmpty() const { return This.isEmpty() && Return.isEmpty(); }
};  

} // end namespace clang

#endif // CLANG_BASIC_ABI_H
