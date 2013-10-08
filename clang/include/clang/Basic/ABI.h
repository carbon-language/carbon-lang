//===----- ABI.h - ABI related declarations ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Enums/classes describing ABI related information about constructors,
/// destructors and thunks.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_BASIC_ABI_H
#define CLANG_BASIC_ABI_H

#include "llvm/Support/DataTypes.h"

namespace clang {

/// \brief C++ constructor types.
enum CXXCtorType {
    Ctor_Complete,          ///< Complete object ctor
    Ctor_Base,              ///< Base object ctor
    Ctor_CompleteAllocating ///< Complete object allocating ctor
};

/// \brief C++ destructor types.
enum CXXDtorType {
    Dtor_Deleting, ///< Deleting dtor
    Dtor_Complete, ///< Complete object dtor
    Dtor_Base      ///< Base object dtor
};

/// \brief A return adjustment.
struct ReturnAdjustment {
  /// \brief The non-virtual adjustment from the derived object to its
  /// nearest virtual base.
  int64_t NonVirtual;
  
  /// \brief The offset (in bytes), relative to the address point 
  /// of the virtual base class offset.
  int64_t VBaseOffsetOffset;
  
  ReturnAdjustment() : NonVirtual(0), VBaseOffsetOffset(0) { }
  
  bool isEmpty() const { return !NonVirtual && !VBaseOffsetOffset; }

  friend bool operator==(const ReturnAdjustment &LHS, 
                         const ReturnAdjustment &RHS) {
    return LHS.NonVirtual == RHS.NonVirtual && 
      LHS.VBaseOffsetOffset == RHS.VBaseOffsetOffset;
  }

  friend bool operator!=(const ReturnAdjustment &LHS, const ReturnAdjustment &RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const ReturnAdjustment &LHS,
                        const ReturnAdjustment &RHS) {
    if (LHS.NonVirtual < RHS.NonVirtual)
      return true;
    
    return LHS.NonVirtual == RHS.NonVirtual && 
      LHS.VBaseOffsetOffset < RHS.VBaseOffsetOffset;
  }
};
  
/// \brief A \c this pointer adjustment.
struct ThisAdjustment {
  /// \brief The non-virtual adjustment from the derived object to its
  /// nearest virtual base.
  int64_t NonVirtual;

  /// \brief The offset (in bytes), relative to the address point,
  /// of the virtual call offset.
  int64_t VCallOffsetOffset;
  
  ThisAdjustment() : NonVirtual(0), VCallOffsetOffset(0) { }

  bool isEmpty() const { return !NonVirtual && !VCallOffsetOffset; }

  friend bool operator==(const ThisAdjustment &LHS, 
                         const ThisAdjustment &RHS) {
    return LHS.NonVirtual == RHS.NonVirtual && 
      LHS.VCallOffsetOffset == RHS.VCallOffsetOffset;
  }

  friend bool operator!=(const ThisAdjustment &LHS, const ThisAdjustment &RHS) {
    return !(LHS == RHS);
  }
  
  friend bool operator<(const ThisAdjustment &LHS,
                        const ThisAdjustment &RHS) {
    if (LHS.NonVirtual < RHS.NonVirtual)
      return true;
    
    return LHS.NonVirtual == RHS.NonVirtual && 
      LHS.VCallOffsetOffset < RHS.VCallOffsetOffset;
  }
};

class CXXMethodDecl;

/// \brief The \c this pointer adjustment as well as an optional return
/// adjustment for a thunk.
struct ThunkInfo {
  /// \brief The \c this pointer adjustment.
  ThisAdjustment This;
    
  /// \brief The return adjustment.
  ReturnAdjustment Return;

  /// \brief Holds a pointer to the overridden method this thunk is for,
  /// if needed by the ABI to distinguish different thunks with equal
  /// adjustments. Otherwise, null.
  /// CAUTION: In the unlikely event you need to sort ThunkInfos, consider using
  /// an ABI-specific comparator.
  const CXXMethodDecl *Method;

  ThunkInfo() : Method(0) { }

  ThunkInfo(const ThisAdjustment &This, const ReturnAdjustment &Return,
            const CXXMethodDecl *Method = 0)
      : This(This), Return(Return), Method(Method) {}

  friend bool operator==(const ThunkInfo &LHS, const ThunkInfo &RHS) {
    return LHS.This == RHS.This && LHS.Return == RHS.Return &&
           LHS.Method == RHS.Method;
  }

  bool isEmpty() const { return This.isEmpty() && Return.isEmpty() && Method == 0; }
};  

} // end namespace clang

#endif // CLANG_BASIC_ABI_H
