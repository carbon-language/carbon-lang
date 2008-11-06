//===--- IdentifierTable.h - Hash table for identifier lookup ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IdentifierInfo, IdentifierTable, and Selector
// interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_IDENTIFIERTABLE_H
#define LLVM_CLANG_BASIC_IDENTIFIERTABLE_H

#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/SerializationFwd.h"
#include <string> 
#include <cassert> 

namespace llvm {
  template <typename T> struct DenseMapInfo;
}

namespace clang {
  struct LangOptions;
  class MultiKeywordSelector; // a private class used by Selector.
  class IdentifierInfo;
  class SourceLocation;
  
  /// IdentifierLocPair - A simple pair of identifier info and location.
  typedef std::pair<IdentifierInfo*, SourceLocation> IdentifierLocPair;
  
  
/// IdentifierInfo - One of these records is kept for each identifier that
/// is lexed.  This contains information about whether the token was #define'd,
/// is a language keyword, or if it is a front-end token of some sort (e.g. a
/// variable or function name).  The preprocessor keeps this information in a
/// set, and all tok::identifier tokens have a pointer to one of these.  
class IdentifierInfo {
  // Note: DON'T make TokenID a 'tok::TokenKind'; MSVC will treat it as a
  //       signed char and TokenKinds > 127 won't be handled correctly.
  unsigned TokenID            : 8; // Front-end token ID or tok::identifier. 
  // Objective-C keyword ('protocol' in '@protocol') or builtin (__builtin_inf).
  // First NUM_OBJC_KEYWORDS values are for Objective-C, the remaining values
  // are for builtins.
  unsigned ObjCOrBuiltinID    :10; 
  unsigned OperatorID         : 6; // C++ overloaded operator.
  bool HasMacro               : 1; // True if there is a #define for this.
  bool IsExtension            : 1; // True if identifier is a lang extension.
  bool IsPoisoned             : 1; // True if identifier is poisoned.
  bool IsCPPOperatorKeyword   : 1; // True if ident is a C++ operator keyword.
  // 4 bits left in 32-bit word.
  void *FETokenInfo;               // Managed by the language front-end.
  IdentifierInfo(const IdentifierInfo&);  // NONCOPYABLE.
  void operator=(const IdentifierInfo&);  // NONASSIGNABLE.
public:
  IdentifierInfo();

  /// getName - Return the actual string for this identifier.  The returned 
  /// string is properly null terminated.
  ///
  const char *getName() const {
    // We know that this is embedded into a StringMapEntry, and it knows how to
    // efficiently find the string.
    return llvm::StringMapEntry<IdentifierInfo>::
                  GetStringMapEntryFromValue(*this).getKeyData();
  }
  
  /// getLength - Efficiently return the length of this identifier info.
  ///
  unsigned getLength() const {
    return llvm::StringMapEntry<IdentifierInfo>::
                    GetStringMapEntryFromValue(*this).getKeyLength();
  }
  
  /// hasMacroDefinition - Return true if this identifier is #defined to some
  /// other value.
  bool hasMacroDefinition() const {
    return HasMacro;
  }
  void setHasMacroDefinition(bool Val) { HasMacro = Val; }
  
  /// get/setTokenID - If this is a source-language token (e.g. 'for'), this API
  /// can be used to cause the lexer to map identifiers to source-language
  /// tokens.
  tok::TokenKind getTokenID() const { return (tok::TokenKind)TokenID; }
  void setTokenID(tok::TokenKind ID) { TokenID = ID; }
  
  /// getPPKeywordID - Return the preprocessor keyword ID for this identifier.
  /// For example, "define" will return tok::pp_define.
  tok::PPKeywordKind getPPKeywordID() const;
  
  /// getObjCKeywordID - Return the Objective-C keyword ID for the this
  /// identifier.  For example, 'class' will return tok::objc_class if ObjC is
  /// enabled.
  tok::ObjCKeywordKind getObjCKeywordID() const {
    if (ObjCOrBuiltinID < tok::NUM_OBJC_KEYWORDS) 
      return tok::ObjCKeywordKind(ObjCOrBuiltinID);
    else
      return tok::objc_not_keyword;
  }
  void setObjCKeywordID(tok::ObjCKeywordKind ID) { ObjCOrBuiltinID = ID; }
  
  /// getBuiltinID - Return a value indicating whether this is a builtin
  /// function.  0 is not-built-in.  1 is builtin-for-some-nonprimary-target.
  /// 2+ are specific builtin functions.
  unsigned getBuiltinID() const { 
    if (ObjCOrBuiltinID >= tok::NUM_OBJC_KEYWORDS)
      return ObjCOrBuiltinID - tok::NUM_OBJC_KEYWORDS; 
    else
      return 0;
  }
  void setBuiltinID(unsigned ID) {
    ObjCOrBuiltinID = ID + tok::NUM_OBJC_KEYWORDS;
    assert(ObjCOrBuiltinID - tok::NUM_OBJC_KEYWORDS == ID 
           && "ID too large for field!");
  }
  
  /// getOverloadedOperatorID - Get the C++ overloaded operator that
  /// corresponds to this identifier.
  OverloadedOperatorKind getOverloadedOperatorID() const {
    return OverloadedOperatorKind(OperatorID);
  }
  void setOverloadedOperatorID(OverloadedOperatorKind ID) {
    OperatorID = ID;
    assert(OperatorID == (unsigned)ID && "ID too large for field!");
  }

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
  
  /// isCPlusPlusOperatorKeyword/setIsCPlusPlusOperatorKeyword controls whether
  /// this identifier is a C++ alternate representation of an operator.
  void setIsCPlusPlusOperatorKeyword(bool Val = true)
    { IsCPPOperatorKeyword = Val; }
  bool isCPlusPlusOperatorKeyword() const { return IsCPPOperatorKeyword; }

  /// getFETokenInfo/setFETokenInfo - The language front-end is allowed to
  /// associate arbitrary metadata with this token.
  template<typename T>
  T *getFETokenInfo() const { return static_cast<T*>(FETokenInfo); }
  void setFETokenInfo(void *T) { FETokenInfo = T; }
  
  /// Emit - Serialize this IdentifierInfo to a bitstream.
  void Emit(llvm::Serializer& S) const;
  
  /// Read - Deserialize an IdentifierInfo object from a bitstream.
  void Read(llvm::Deserializer& D);  
};

/// IdentifierTable - This table implements an efficient mapping from strings to
/// IdentifierInfo nodes.  It has no other purpose, but this is an
/// extremely performance-critical piece of the code, as each occurrance of
/// every identifier goes through here when lexed.
class IdentifierTable {
  // Shark shows that using MallocAllocator is *much* slower than using this
  // BumpPtrAllocator!
  typedef llvm::StringMap<IdentifierInfo, llvm::BumpPtrAllocator> HashTableTy;
  HashTableTy HashTable;

  /// OverloadedOperators - Identifiers corresponding to each of the
  /// overloadable operators in C++.
  IdentifierInfo *OverloadedOperators[NUM_OVERLOADED_OPERATORS];

public:
  /// IdentifierTable ctor - Create the identifier table, populating it with
  /// info about the language keywords for the language specified by LangOpts.
  IdentifierTable(const LangOptions &LangOpts);
  
  /// get - Return the identifier token info for the specified named identifier.
  ///
  IdentifierInfo &get(const char *NameStart, const char *NameEnd) {
    return HashTable.GetOrCreateValue(NameStart, NameEnd).getValue();
  }
  
  IdentifierInfo &get(const char *Name) {
    return get(Name, Name+strlen(Name));
  }
  IdentifierInfo &get(const std::string &Name) {
    // Don't use c_str() here: no need to be null terminated.
    const char *NameBytes = &Name[0];
    return get(NameBytes, NameBytes+Name.size());
  }

  /// getOverloadedOperator - Retrieve the identifier   
  IdentifierInfo &getOverloadedOperator(OverloadedOperatorKind Op) {
    return *OverloadedOperators[Op];
  }

  typedef HashTableTy::const_iterator iterator;
  typedef HashTableTy::const_iterator const_iterator;
  
  iterator begin() const { return HashTable.begin(); }
  iterator end() const   { return HashTable.end(); }
  
  unsigned size() const { return HashTable.size(); }
  
  /// PrintStats - Print some statistics to stderr that indicate how well the
  /// hashing is doing.
  void PrintStats() const;
  
  void AddKeywords(const LangOptions &LangOpts);
  void AddOverloadedOperators();

  /// Emit - Serialize this IdentifierTable to a bitstream.  This should
  ///  be called AFTER objects that externally reference the identifiers in the 
  ///  table have been serialized.  This is because only the identifiers that
  ///  are actually referenced are serialized.
  void Emit(llvm::Serializer& S) const;
  
  /// Create - Deserialize an IdentifierTable from a bitstream.
  static IdentifierTable* CreateAndRegister(llvm::Deserializer& D);
  
private:  
  /// This ctor is not intended to be used by anyone except for object
  /// serialization.
  IdentifierTable();  
};

/// Selector - This smart pointer class efficiently represents Objective-C
/// method names. This class will either point to an IdentifierInfo or a
/// MultiKeywordSelector (which is private). This enables us to optimize
/// selectors that take no arguments and selectors that take 1 argument, which 
/// accounts for 78% of all selectors in Cocoa.h.
class Selector {
  enum IdentifierInfoFlag {
    // MultiKeywordSelector = 0.
    ZeroArg  = 0x1,
    OneArg   = 0x2,
    ArgFlags = ZeroArg|OneArg
  };
  uintptr_t InfoPtr; // a pointer to the MultiKeywordSelector or IdentifierInfo.
  
  Selector(IdentifierInfo *II, unsigned nArgs) {
    InfoPtr = reinterpret_cast<uintptr_t>(II);
    assert((InfoPtr & ArgFlags) == 0 &&"Insufficiently aligned IdentifierInfo");
    assert(nArgs < 2 && "nArgs not equal to 0/1");
    InfoPtr |= nArgs+1;
  }
  Selector(MultiKeywordSelector *SI) {
    InfoPtr = reinterpret_cast<uintptr_t>(SI);
    assert((InfoPtr & ArgFlags) == 0 &&"Insufficiently aligned IdentifierInfo");
  }
  Selector(uintptr_t V) : InfoPtr(V) {}
public:
  friend class SelectorTable; // only the SelectorTable can create these.
  
  /// The default ctor should only be used when creating data structures that
  ///  will contain selectors.
  Selector() : InfoPtr(0) {}
  
  IdentifierInfo *getAsIdentifierInfo() const {
    if (getIdentifierInfoFlag())
      return reinterpret_cast<IdentifierInfo *>(InfoPtr & ~ArgFlags);
    return 0;
  }
  unsigned getIdentifierInfoFlag() const {
    return InfoPtr & ArgFlags;
  }
  /// operator==/!= - Indicate whether the specified selectors are identical.
  bool operator==(Selector RHS) const {
    return InfoPtr == RHS.InfoPtr;
  }
  bool operator!=(Selector RHS) const {
    return InfoPtr != RHS.InfoPtr;
  }
  void *getAsOpaquePtr() const {
    return reinterpret_cast<void*>(InfoPtr);
  }
  // Predicates to identify the selector type.
  bool isKeywordSelector() const { 
    return getIdentifierInfoFlag() != ZeroArg; 
  }
  bool isUnarySelector() const { 
    return getIdentifierInfoFlag() == ZeroArg;
  }
  unsigned getNumArgs() const;
  IdentifierInfo *getIdentifierInfoForSlot(unsigned argIndex) const;
  
  /// getName - Derive the full selector name (e.g. "foo:bar:") and return it.
  ///
  std::string getName() const;
  
  static Selector getEmptyMarker() {
    return Selector(uintptr_t(-1));
  }
  static Selector getTombstoneMarker() {
    return Selector(uintptr_t(-2));
  }
  
  // Emit - Emit a selector to bitcode.
  void Emit(llvm::Serializer& S) const;
  
  // ReadVal - Read a selector from bitcode.
  static Selector ReadVal(llvm::Deserializer& D);
};

/// SelectorTable - This table allows us to fully hide how we implement
/// multi-keyword caching.
class SelectorTable {
  void *Impl;  // Actually a FoldingSet<MultiKeywordSelector>*
  SelectorTable(const SelectorTable&); // DISABLED: DO NOT IMPLEMENT
  void operator=(const SelectorTable&); // DISABLED: DO NOT IMPLEMENT
public:
  SelectorTable();
  ~SelectorTable();

  /// getSelector - This can create any sort of selector.  NumArgs indicates
  /// whether this is a no argument selector "foo", a single argument selector
  /// "foo:" or multi-argument "foo:bar:".
  Selector getSelector(unsigned NumArgs, IdentifierInfo **IIV);
  
  Selector getUnarySelector(IdentifierInfo *ID) {
    return Selector(ID, 1);
  }
  Selector getNullarySelector(IdentifierInfo *ID) {
    return Selector(ID, 0);
  }
  
  // Emit - Emit a SelectorTable to bitcode.
  void Emit(llvm::Serializer& S) const;
  
  // Create - Reconstitute a SelectorTable from bitcode.
  static SelectorTable* CreateAndRegister(llvm::Deserializer& D);
};

}  // end namespace clang


/// Define DenseMapInfo so that Selectors can be used as keys in DenseMap and
/// DenseSets.
namespace llvm {
template <>
struct DenseMapInfo<clang::Selector> {
  static inline clang::Selector getEmptyKey() {
    return clang::Selector::getEmptyMarker();
  }
  static inline clang::Selector getTombstoneKey() {
    return clang::Selector::getTombstoneMarker(); 
  }
  
  static unsigned getHashValue(clang::Selector S);
  
  static bool isEqual(clang::Selector LHS, clang::Selector RHS) {
    return LHS == RHS;
  }
  
  static bool isPod() { return true; }
};
}  // end namespace llvm

#endif
