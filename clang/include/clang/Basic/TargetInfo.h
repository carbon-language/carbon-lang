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
#include <string>

namespace llvm {
namespace clang {

class TargetInfoImpl;
class Diagnostic;
namespace Builtin { struct Info; }
  
/// TargetInfo - This class exposes information about the current target set.
/// A target set consists of a primary target and zero or more secondary targets
/// which are each represented by a TargetInfoImpl object.  TargetInfo responds
/// to various queries as though it were the primary target, but keeps track of,
/// and warns about, the first query made of it that are contradictary among the
/// targets it tracks.  For example, if it contains a "PPC32" and "PPC64"
/// target, it will warn the first time the size of the 'long' datatype is
/// queried.
///
/// Note that TargetInfo does not take ownership of the various targets or the 
/// diagnostic info, but does expect them to be alive for as long as it is.
///
class TargetInfo {
  /// Primary - This tracks the primary target in the target set.
  ///
  const TargetInfoImpl *PrimaryTarget;
  
  /// SecondaryTargets - This tracks the set of secondary targets.
  ///
  std::vector<const TargetInfoImpl*> SecondaryTargets;
  
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
  TargetInfo(const TargetInfoImpl *Primary, Diagnostic *D = 0) {
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
  void AddSecondaryTarget(const TargetInfoImpl *Secondary) {
    SecondaryTargets.push_back(Secondary);
  }
  
  ///===---- Target property query methods --------------------------------===//

  /// DiagnoseNonPortability - Emit a diagnostic indicating that the current
  /// translation unit is non-portable due to a construct at the specified
  /// location.  DiagKind indicates what went wrong.
  void DiagnoseNonPortability(SourceLocation Loc, unsigned DiagKind);

  /// getTargetDefines - Appends the target-specific #define values for this
  /// target set to the specified buffer.
  void getTargetDefines(std::vector<char> &DefineBuffer);
  
  /// isCharSigned - Return true if 'char' is 'signed char' or false if it is
  /// treated as 'unsigned char'.  This is implementation defined according to
  /// C99 6.2.5p15.  In our implementation, this is target-specific.
  bool isCharSigned(SourceLocation Loc) {
    // FIXME: implement correctly.
    return true;
  }
  
  /// getPointerWidth - Return the width of pointers on this target, we
  /// currently assume one pointer type.
  unsigned getPointerWidth(SourceLocation Loc) {
    return 32;   // FIXME: implement correctly.
  }
  
  /// getBoolWidth - Return the size of '_Bool' and C++ 'bool' for this target,
  /// in bits.  
  unsigned getBoolWidth(SourceLocation Loc) {
    return 8;    // FIXME: implement correctly: wrong for ppc32.
  }
  
  /// getCharWidth - Return the size of 'char', 'signed char' and
  /// 'unsigned char' for this target, in bits.  
  unsigned getCharWidth(SourceLocation Loc) {
    return 8; // FIXME: implement correctly.
  }
  
  /// getShortWidth - Return the size of 'signed short' and 'unsigned short' for
  /// this target, in bits.  
  unsigned getShortWidth(SourceLocation Loc) {
    return 16; // FIXME: implement correctly.
  }
  
  /// getIntWidth - Return the size of 'signed int' and 'unsigned int' for this
  /// target, in bits.  
  unsigned getIntWidth(SourceLocation Loc) {
    return 32; // FIXME: implement correctly.
  }
  
  /// getLongWidth - Return the size of 'signed long' and 'unsigned long' for
  /// this target, in bits.  
  unsigned getLongWidth(SourceLocation Loc) {
    return 32;  // FIXME: implement correctly: wrong for ppc64/x86-64
  }

  /// getLongLongWidth - Return the size of 'signed long long' and
  /// 'unsigned long long' for this target, in bits.  
  unsigned getLongLongWidth(SourceLocation Loc) {
    return 64; // FIXME: implement correctly.
  }
  
  /// getWCharWidth - Return the size of wchar_t in bits.
  ///
  unsigned getWCharWidth(SourceLocation Loc) {
    if (!WCharWidth) ComputeWCharWidth(Loc);
    return WCharWidth;
  }
  
  /// getIntMaxTWidth - Return the size of intmax_t and uintmax_t for this
  /// target, in bits.  
  unsigned getIntMaxTWidth(SourceLocation Loc) {
    // FIXME: implement correctly.
    return 64;
  }
  
  /// getTargetBuiltins - Return information about target-specific builtins for
  /// the current primary target, and info about which builtins are non-portable
  /// across the current set of primary and secondary targets.
  void getTargetBuiltins(const Builtin::Info *&Records, unsigned &NumRecords,
                         std::vector<const char *> &NonPortableBuiltins) const;

private:
  void ComputeWCharWidth(SourceLocation Loc);
};




/// TargetInfoImpl - This class is implemented for specific targets and is used
/// by the TargetInfo class.  Target implementations should initialize instance
/// variables and implement various virtual methods if the default values are
/// not appropriate for the target.
class TargetInfoImpl {
protected:
  unsigned WCharWidth;    /// sizeof(wchar_t) in bits.  Default value is 32.
public:
  TargetInfoImpl() : WCharWidth(32) {}
  virtual ~TargetInfoImpl() {}
  
  /// getTargetDefines - Return a list of the target-specific #define values set
  /// when compiling to this target.  Each string should be of the form "X",
  /// which results in '#define X 1' or "X=Y" which results in "#define X Y"
  virtual void getTargetDefines(std::vector<std::string> &Defines) const = 0;

  /// getWCharWidth - Return the size of wchar_t in bits.
  ///
  unsigned getWCharWidth() const { return WCharWidth; }
  
  /// getTargetBuiltins - Return information about target-specific builtins for
  /// the target.
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = 0;
    NumRecords = 0;
  }
private:
  virtual void ANCHOR(); // out-of-line virtual method for class.
};

}  // end namespace clang
}  // end namespace llvm

#endif
