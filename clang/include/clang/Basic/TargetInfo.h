//===--- TargetInfo.h - Expose information about the target -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TargetInfo and TargetInfoImpl interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETINFO_H
#define LLVM_CLANG_BASIC_TARGETINFO_H

#include "clang/Basic/SourceLocation.h"
#include <vector>

namespace llvm {
namespace clang {

class TargetInfoImpl;
class Diagnostic;
  
/// TargetInfo - This class exposes information about the current target set.
/// A target set consists of a primary target and zero or more secondary targets
/// which are each represented by a TargetInfoImpl object.  TargetInfo responds
/// to various queries as though it were the primary target, but keeps track of,
/// and warns about, the first query made of it that are contradictary among the
/// targets it tracks.  For example, if it contains a "PPC32" and "PPC64"
/// target, it will warn the first time the size of the 'long' datatype is
/// queried.
///
/// Note that TargetInfo does not take ownership of the various targets, but
/// does expect them to be alive for as long as it is.
///
class TargetInfo {
  /// Primary - This tracks the primary target in the target set.
  ///
  TargetInfoImpl *PrimaryTarget;
  
  /// SecondaryTargets - This tracks the set of secondary targets.
  ///
  std::vector<TargetInfoImpl*> SecondaryTargets;
  
  /// Diag - If non-null, this object is used to report the first use of
  /// non-portable functionality in the translation unit.
  /// 
  Diagnostic *Diag;

  /// NonPortable - This instance variable keeps track of whether or not the
  /// current translation unit is portable across the set of targets tracked.
  bool NonPortable;

  /// These are all caches for target values.
  unsigned WCharWidth;
  
public:
  TargetInfo(TargetInfoImpl *Primary, Diagnostic *D = 0) {
    PrimaryTarget = Primary;
    Diag = D;
    NonPortable = false;
    
    // Initialize Cache values to uncomputed.
    WCharWidth = 0;
  }
  
  /// isNonPortable - Return true if the current translation unit has used a
  /// target property that is non-portable across the secondary targets.
  bool isNonPortable() const {
    return NonPortable;
  }
  
  /// isPortable - Return true if this translation unit is portable across the
  /// secondary targets so far.
  bool isPortable() const {
    return !NonPortable;
  }
  
  /// AddSecondaryTarget - Add a secondary target to the target set.
  void AddSecondaryTarget(TargetInfoImpl *Secondary) {
    SecondaryTargets.push_back(Secondary);
  }
  
  ///===---- Target property query methods --------------------------------===//
  
  /// getWCharWidth - Return the size of wchar_t in bytes.
  ///
  unsigned getWCharWidth(SourceLocation Loc) {
    if (!WCharWidth) ComputeWCharWidth(Loc);
    return WCharWidth;
  }
private:
  void DiagnoseNonPortability(SourceLocation Loc, unsigned DiagKind);
  void ComputeWCharWidth(SourceLocation Loc);
};




/// TargetInfoImpl - This class is implemented for specific targets and is used
/// by the TargetInfo class.  Target implementations should initialize instance
/// variables and implement various virtual methods if the default values are
/// not appropriate for the target.
class TargetInfoImpl {
protected:
  unsigned WCharWidth;    /// sizeof(wchar_t) in bytes.  Default value is 4.
public:
  TargetInfoImpl() : WCharWidth(4) {}
  
  /// getWCharWidth - Return the size of wchar_t in bytes.
  ///
  unsigned getWCharWidth() const { return WCharWidth; }
  
};

}  // end namespace clang
}  // end namespace llvm

#endif
