//===--- IdentifierTable.h - Hash table for identifier lookup ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IdentifierInfo, IdentifierVisitor, and
// IdentifierTable interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_IDENTIFIERTABLE_H
#define LLVM_CLANG_IDENTIFIERTABLE_H

#include "clang/Basic/TokenKinds.h"
#include <string> 

namespace llvm {
namespace clang {
  class IdentifierTable;
  class MacroInfo;
  class LangOptions;
  
/// IdentifierInfo - One of these records is kept for each identifier that
/// is lexed.  This contains information about whether the token was #define'd,
/// is a language keyword, or if it is a front-end token of some sort (e.g. a
/// variable or function name).  The preprocessor keeps this information in a
/// set, and all tok::identifier tokens have a pointer to one of these.  
class IdentifierInfo {
  unsigned NameLen;                // String that is the identifier.
  MacroInfo *Macro;                // Set if this identifier is #define'd.
  tok::TokenKind TokenID      : 8; // Front-end token ID or tok::identifier.
  tok::PPKeywordKind PPID     : 5; // ID for preprocessor command like 'ifdef'.
  tok::ObjCKeywordKind ObjCID : 5; // ID for preprocessor command like 'ifdef'.
  bool IsExtension            : 1; // True if identifier is a lang extension.
  bool IsPoisoned             : 1; // True if identifier is poisoned.
  bool IsOtherTargetMacro     : 1; // True if ident is macro on another target.
  void *FETokenInfo;               // Managed by the language front-end.
  friend class IdentifierTable;
public:
  /// getName - Return the actual string for this identifier.  The length of
  /// this string is stored in NameLen, and the returned string is properly null
  /// terminated.
  ///
  const char *getName() const {
    // String data is stored immediately after the IdentifierInfo object.
    return (const char*)(this+1);
  }
  
  /// getNameLength - Return the length of the identifier string.
  ///
  unsigned getNameLength() const {
    return NameLen;
  }
  
  /// getMacroInfo - Return macro information about this identifier, or null if
  /// it is not a macro.
  MacroInfo *getMacroInfo() const { return Macro; }
  void setMacroInfo(MacroInfo *I) { Macro = I; }
  
  /// get/setTokenID - If this is a source-language token (e.g. 'for'), this API
  /// can be used to cause the lexer to map identifiers to source-language
  /// tokens.
  tok::TokenKind getTokenID() const { return TokenID; }
  void setTokenID(tok::TokenKind ID) { TokenID = ID; }
  
  /// getPPKeywordID - Return the preprocessor keyword ID for this identifier.
  /// For example, define will return tok::pp_define.
  tok::PPKeywordKind getPPKeywordID() const { return PPID; }
  void setPPKeywordID(tok::PPKeywordKind ID) { PPID = ID; }
  
  /// getObjCKeywordID - Return the Objective-C keyword ID for the this
  /// identifier.  For example, 'class' will return tok::objc_class if ObjC is
  /// enabled.
  tok::ObjCKeywordKind getObjCKeywordID() const { return ObjCID; }
  void setObjCKeywordID(tok::ObjCKeywordKind ID) { ObjCID = ID; }
  
  /// get/setExtension - Initialize information about whether or not this
  /// language token is an extension.  This controls extension warnings, and is
  /// only valid if a custom token ID is set.
  bool isExtensionToken() const { return IsExtension; }
  void setIsExtensionToken(bool Val) { IsExtension = Val; }
  
  /// setIsPoisoned - Mark this identifier as poisoned.  After poisoning, the
  /// Preprocessor will emit an error every time this token is used.
  void setIsPoisoned(bool Value = true) { IsPoisoned = Value; }
  
  /// isPoisoned - Return true if this token has been poisoned.
  bool isPoisoned() const { return IsPoisoned; }
  
  /// setIsOtherTargetMacro/isOtherTargetMacro control whether this identifier
  /// is seen as being a macro on some other target.
  void setIsOtherTargetMacro(bool Val = true) { IsOtherTargetMacro = Val; }
  bool isOtherTargetMacro() const { return IsOtherTargetMacro; }
  
  /// getFETokenInfo/setFETokenInfo - The language front-end is allowed to
  /// associate arbitrary metadata with this token.
  template<typename T>
  T *getFETokenInfo() const { return static_cast<T*>(FETokenInfo); }
  void setFETokenInfo(void *T) { FETokenInfo = T; }
private:
  void Destroy();
};

/// IdentifierVisitor - Subclasses of this class may be implemented to walk all
/// of the defined identifiers.
class IdentifierVisitor {
public:
  virtual ~IdentifierVisitor();
  virtual void VisitIdentifier(IdentifierInfo &II) const = 0;
};

/// IdentifierTable - This table implements an efficient mapping from strings to
/// IdentifierInfo nodes.  It has no other purpose, but this is an
/// extremely performance-critical piece of the code, as each occurrance of
/// every identifier goes through here when lexed.
class IdentifierTable {
  void *TheTable;
  void *TheMemory;
  unsigned NumIdentifiers;
public:
  /// IdentifierTable ctor - Create the identifier table, populating it with
  /// info about the language keywords for the language specified by LangOpts.
  IdentifierTable(const LangOptions &LangOpts);
  ~IdentifierTable();
  
  /// get - Return the identifier token info for the specified named identifier.
  ///
  IdentifierInfo &get(const char *NameStart, const char *NameEnd);
  IdentifierInfo &get(const std::string &Name);
  
  /// VisitIdentifiers - This method walks through all of the identifiers,
  /// invoking IV->VisitIdentifier for each of them.
  void VisitIdentifiers(const IdentifierVisitor &IV);
  
  /// PrintStats - Print some statistics to stderr that indicate how well the
  /// hashing is doing.
  void PrintStats() const;
private:
  void AddKeywords(const LangOptions &LangOpts);
};

}  // end namespace llvm
}  // end namespace clang

#endif
