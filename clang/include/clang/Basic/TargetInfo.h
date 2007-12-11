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
#include "llvm/Support/DataTypes.h"
#include <vector>
#include <string>

namespace llvm { struct fltSemantics; }

namespace clang {

class TargetInfoImpl;
class Diagnostic;
class SourceManager;
  
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
  /// SrcMgr - The SourceManager associated with this TargetInfo.
  SourceManager& SrcMgr;
  
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
  unsigned WCharWidth, WCharAlign;
  
public:
  TargetInfo(SourceManager& SMgr, const TargetInfoImpl *Primary,
             Diagnostic *D = 0) : SrcMgr(SMgr) {
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
  void getPointerInfo(uint64_t &Size, unsigned &Align, SourceLocation Loc) {
    Size = 32;  // FIXME: implement correctly.
    Align = 32;
  }
  
  /// getBoolInfo - Return the size of '_Bool' and C++ 'bool' for this target,
  /// in bits.  
  void getBoolInfo(uint64_t &Size, unsigned &Align, SourceLocation Loc) {
    Size = Align = 8;    // FIXME: implement correctly: wrong for ppc32.
  }
  
  /// getCharInfo - Return the size of 'char', 'signed char' and
  /// 'unsigned char' for this target, in bits.  
  void getCharInfo(uint64_t &Size, unsigned &Align, SourceLocation Loc) {
    Size = Align = 8; // FIXME: implement correctly.
  }
  
  /// getShortInfo - Return the size of 'signed short' and 'unsigned short' for
  /// this target, in bits.  
  void getShortInfo(uint64_t &Size, unsigned &Align, SourceLocation Loc) {
    Size = Align = 16; // FIXME: implement correctly.
  }
  
  /// getIntInfo - Return the size of 'signed int' and 'unsigned int' for this
  /// target, in bits.  
  void getIntInfo(uint64_t &Size, unsigned &Align, SourceLocation Loc) {
    Size = Align = 32; // FIXME: implement correctly.
  }
  
  /// getLongInfo - Return the size of 'signed long' and 'unsigned long' for
  /// this target, in bits.  
  void getLongInfo(uint64_t &Size, unsigned &Align, SourceLocation Loc) {
    Size = Align = 32;  // FIXME: implement correctly: wrong for ppc64/x86-64
  }

  /// getLongLongInfo - Return the size of 'signed long long' and
  /// 'unsigned long long' for this target, in bits.  
  void getLongLongInfo(uint64_t &Size, unsigned &Align, 
                            SourceLocation Loc) {
    Size = Align = 64; // FIXME: implement correctly.
  }
  
  /// getFloatInfo - Characterize 'float' for this target.  
  void getFloatInfo(uint64_t &Size, unsigned &Align,
                    const llvm::fltSemantics *&Format, SourceLocation Loc);

  /// getDoubleInfo - Characterize 'double' for this target.
  void getDoubleInfo(uint64_t &Size, unsigned &Align,
                     const llvm::fltSemantics *&Format,  SourceLocation Loc);

  /// getLongDoubleInfo - Characterize 'long double' for this target.
  void getLongDoubleInfo(uint64_t &Size, unsigned &Align,
                         const llvm::fltSemantics *&Format, SourceLocation Loc);
  
  /// getWCharInfo - Return the size of wchar_t in bits.
  ///
  void getWCharInfo(uint64_t &Size, unsigned &Align, SourceLocation Loc) {
    if (!WCharWidth) ComputeWCharInfo(Loc);
    Size = WCharWidth;
    Align = WCharAlign;
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

  /// getVAListDeclaration - Return the declaration to use for
  /// __builtin_va_list, which is target-specific.
  const char *getVAListDeclaration() const;

  /// isValidGCCRegisterName - Returns whether the passed in string
  /// is a valid register name according to GCC. This is used by Sema for
  /// inline asm statements.
  bool isValidGCCRegisterName(const char *Name) const;

  // getNormalizedGCCRegisterName - Returns the "normalized" GCC register name.
  // For example, on x86 it will return "ax" when "eax" is passed in.
  const char *getNormalizedGCCRegisterName(const char *Name) const;
  
  enum ConstraintInfo {
    CI_None = 0x00,
    CI_AllowsMemory = 0x01,
    CI_AllowsRegister = 0x02,
    CI_ReadWrite = 0x03
  };

  // validateOutputConstraint, validateInputConstraint - Checks that
  // a constraint is valid and provides information about it.
  // FIXME: These should return a real error instead of just true/false.
  bool validateOutputConstraint(const char *Name, ConstraintInfo &Info) const;
  bool validateInputConstraint (const char *Name, unsigned NumOutputs,
                                ConstraintInfo &info) const;
  
  // Returns a string of target-specific clobbers, in LLVM format.
  const char *getClobbers() const;
  
  ///===---- Some helper methods ------------------------------------------===//

  unsigned getCharWidth(SourceLocation Loc) {
    uint64_t Size; unsigned Align;
    getCharInfo(Size, Align, Loc);
    return static_cast<unsigned>(Size);
  }
  
  unsigned getWCharWidth(SourceLocation Loc) {
    uint64_t Size; unsigned Align;
    getWCharInfo(Size, Align, Loc);
    return static_cast<unsigned>(Size);
  }
  
  unsigned getIntWidth(SourceLocation Loc) {
    uint64_t Size; unsigned Align;
    getIntInfo(Size, Align, Loc);
    return static_cast<unsigned>(Size);
  }
  
  unsigned getLongWidth(SourceLocation Loc) {
    uint64_t Size; unsigned Align;
    getLongInfo(Size, Align, Loc);
    return static_cast<unsigned>(Size);
  }

  unsigned getLongLongWidth(SourceLocation Loc) {
    uint64_t Size; unsigned Align;
    getLongLongInfo(Size, Align, Loc);
    return static_cast<unsigned>(Size);
  }

  /// getTargetPrefix - Return the target prefix used for identifying
  /// llvm intrinsics.
  const char *getTargetPrefix() const;
    
  /// getTargetTriple - Return the target triple of the primary target.
  const char *getTargetTriple() const;
  
  const char *getTargetDescription() const {
    // FIXME !
    // Hard code darwin-x86 for now.
    return "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:\
32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128";
  }
private:
  void ComputeWCharInfo(SourceLocation Loc);
};




/// TargetInfoImpl - This class is implemented for specific targets and is used
/// by the TargetInfo class.  Target implementations should initialize instance
/// variables and implement various virtual methods if the default values are
/// not appropriate for the target.
class TargetInfoImpl {
protected:
  unsigned WCharWidth;    /// sizeof(wchar_t) in bits.  Default value is 32.
  unsigned WCharAlign;    /// alignof(wchar_t) in bits.  Default value is 32.
  std::string Triple;
public:
  TargetInfoImpl(const std::string& triple) 
    : WCharWidth(32), WCharAlign(32), Triple(triple) {}
  
  virtual ~TargetInfoImpl() {}
  
  /// getTargetTriple - Return the string representing the target triple this
  ///  TargetInfoImpl object was created from.
  const char* getTargetTriple() const { return Triple.c_str(); }
  
  virtual const char *getTargetPrefix() const = 0;

  /// getTargetDefines - Return a list of the target-specific #define values set
  /// when compiling to this target.  Each string should be of the form
  /// "#define X Y\n".
  virtual void getTargetDefines(std::vector<char> &Defines) const = 0;

  /// getVAListDeclaration - Return the declaration to use for
  /// __builtin_va_list, which is target-specific.
  virtual const char *getVAListDeclaration() const = 0;
    
  /// getWCharWidth - Return the size of wchar_t in bits.
  ///
  void getWCharInfo(unsigned &Size, unsigned &Align) const {
    Size = WCharWidth;
    Align = WCharAlign;
  }
    
  /// getTargetBuiltins - Return information about target-specific builtins for
  /// the target.
  virtual void getTargetBuiltins(const Builtin::Info *&Records,
                                 unsigned &NumRecords) const {
    Records = 0;
    NumRecords = 0;
  }
  
  virtual void getGCCRegNames(const char * const *&Names, 
                              unsigned &NumNames) const = 0;

  struct GCCRegAlias {
    const char * const Aliases[5];
    const char * const Register;
  };
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                unsigned &NumAliases) const = 0;
  
  virtual bool validateAsmConstraint(char c, 
                                     TargetInfo::ConstraintInfo &info) const= 0;
  
  virtual const char *getClobbers() const = 0;
private:
  virtual void ANCHOR(); // out-of-line virtual method for class.
};

}  // end namespace clang

#endif
