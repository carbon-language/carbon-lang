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
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <string>
#include <cassert>

namespace llvm {
  template <typename T> struct DenseMapInfo;
}

namespace clang {
  class LangOptions;
  class IdentifierInfo;
  class IdentifierTable;
  class SourceLocation;
  class MultiKeywordSelector; // private class used by Selector
  class DeclarationName;      // AST class that stores declaration names

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
  bool HasMacro               : 1; // True if there is a #define for this.
  bool IsExtension            : 1; // True if identifier is a lang extension.
  bool IsPoisoned             : 1; // True if identifier is poisoned.
  bool IsCPPOperatorKeyword   : 1; // True if ident is a C++ operator keyword.
  bool NeedsHandleIdentifier  : 1; // See "RecomputeNeedsHandleIdentifier".
  // 9 bits left in 32-bit word.
  void *FETokenInfo;               // Managed by the language front-end.
  llvm::StringMapEntry<IdentifierInfo*> *Entry;

  IdentifierInfo(const IdentifierInfo&);  // NONCOPYABLE.
  void operator=(const IdentifierInfo&);  // NONASSIGNABLE.

  friend class IdentifierTable;

public:
  IdentifierInfo();


  /// isStr - Return true if this is the identifier for the specified string.
  /// This is intended to be used for string literals only: II->isStr("foo").
  template <std::size_t StrLen>
  bool isStr(const char (&Str)[StrLen]) const {
    return getLength() == StrLen-1 && !memcmp(getName(), Str, StrLen-1);
  }

  /// getName - Return the actual string for this identifier.  The returned
  /// string is properly null terminated.
  ///
  const char *getName() const {
    if (Entry) return Entry->getKeyData();
    // FIXME: This is gross. It would be best not to embed specific details
    // of the PTH file format here.
    // The 'this' pointer really points to a
    // std::pair<IdentifierInfo, const char*>, where internal pointer
    // points to the external string data.
    return ((std::pair<IdentifierInfo, const char*>*) this)->second;
  }

  /// getLength - Efficiently return the length of this identifier info.
  ///
  unsigned getLength() const {
    if (Entry) return Entry->getKeyLength();
    // FIXME: This is gross. It would be best not to embed specific details
    // of the PTH file format here.
    // The 'this' pointer really points to a
    // std::pair<IdentifierInfo, const char*>, where internal pointer
    // points to the external string data.
    const char* p = ((std::pair<IdentifierInfo, const char*>*) this)->second-2;
    return (((unsigned) p[0])
        | (((unsigned) p[1]) << 8)) - 1;
  }

  /// hasMacroDefinition - Return true if this identifier is #defined to some
  /// other value.
  bool hasMacroDefinition() const {
    return HasMacro;
  }
  void setHasMacroDefinition(bool Val) {
    if (HasMacro == Val) return;

    HasMacro = Val;
    if (Val)
      NeedsHandleIdentifier = 1;
    else
      RecomputeNeedsHandleIdentifier();
  }

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
    assert(ObjCOrBuiltinID - unsigned(tok::NUM_OBJC_KEYWORDS) == ID
           && "ID too large for field!");
  }

  unsigned getObjCOrBuiltinID() const { return ObjCOrBuiltinID; }
  void setObjCOrBuiltinID(unsigned ID) { ObjCOrBuiltinID = ID; }

  /// get/setExtension - Initialize information about whether or not this
  /// language token is an extension.  This controls extension warnings, and is
  /// only valid if a custom token ID is set.
  bool isExtensionToken() const { return IsExtension; }
  void setIsExtensionToken(bool Val) {
    IsExtension = Val;
    if (Val)
      NeedsHandleIdentifier = 1;
    else
      RecomputeNeedsHandleIdentifier();
  }

  /// setIsPoisoned - Mark this identifier as poisoned.  After poisoning, the
  /// Preprocessor will emit an error every time this token is used.
  void setIsPoisoned(bool Value = true) {
    IsPoisoned = Value;
    if (Value)
      NeedsHandleIdentifier = 1;
    else
      RecomputeNeedsHandleIdentifier();
  }

  /// isPoisoned - Return true if this token has been poisoned.
  bool isPoisoned() const { return IsPoisoned; }

  /// isCPlusPlusOperatorKeyword/setIsCPlusPlusOperatorKeyword controls whether
  /// this identifier is a C++ alternate representation of an operator.
  void setIsCPlusPlusOperatorKeyword(bool Val = true) {
    IsCPPOperatorKeyword = Val;
    if (Val)
      NeedsHandleIdentifier = 1;
    else
      RecomputeNeedsHandleIdentifier();
  }
  bool isCPlusPlusOperatorKeyword() const { return IsCPPOperatorKeyword; }

  /// getFETokenInfo/setFETokenInfo - The language front-end is allowed to
  /// associate arbitrary metadata with this token.
  template<typename T>
  T *getFETokenInfo() const { return static_cast<T*>(FETokenInfo); }
  void setFETokenInfo(void *T) { FETokenInfo = T; }

  /// isHandleIdentifierCase - Return true if the Preprocessor::HandleIdentifier
  /// must be called on a token of this identifier.  If this returns false, we
  /// know that HandleIdentifier will not affect the token.
  bool isHandleIdentifierCase() const { return NeedsHandleIdentifier; }

private:
  /// RecomputeNeedsHandleIdentifier - The Preprocessor::HandleIdentifier does
  /// several special (but rare) things to identifiers of various sorts.  For
  /// example, it changes the "for" keyword token from tok::identifier to
  /// tok::for.
  ///
  /// This method is very tied to the definition of HandleIdentifier.  Any
  /// change to it should be reflected here.
  void RecomputeNeedsHandleIdentifier() {
    NeedsHandleIdentifier =
      (isPoisoned() | hasMacroDefinition() | isCPlusPlusOperatorKeyword() |
       isExtensionToken());
  }
};

/// IdentifierInfoLookup - An abstract class used by IdentifierTable that
///  provides an interface for performing lookups from strings
/// (const char *) to IdentiferInfo objects.
class IdentifierInfoLookup {
public:
  virtual ~IdentifierInfoLookup();

  /// get - Return the identifier token info for the specified named identifier.
  ///  Unlike the version in IdentifierTable, this returns a pointer instead
  ///  of a reference.  If the pointer is NULL then the IdentifierInfo cannot
  ///  be found.
  virtual IdentifierInfo* get(const char *NameStart, const char *NameEnd) = 0;
};

/// \brief An abstract class used to resolve numerical identifier
/// references (meaningful only to some external source) into
/// IdentifierInfo pointers.
class ExternalIdentifierLookup {
public:
  virtual ~ExternalIdentifierLookup();

  /// \brief Return the identifier associated with the given ID number.
  ///
  /// The ID 0 is associated with the NULL identifier.
  virtual IdentifierInfo *GetIdentifier(unsigned ID) = 0;
};

/// IdentifierTable - This table implements an efficient mapping from strings to
/// IdentifierInfo nodes.  It has no other purpose, but this is an
/// extremely performance-critical piece of the code, as each occurrance of
/// every identifier goes through here when lexed.
class IdentifierTable {
  // Shark shows that using MallocAllocator is *much* slower than using this
  // BumpPtrAllocator!
  typedef llvm::StringMap<IdentifierInfo*, llvm::BumpPtrAllocator> HashTableTy;
  HashTableTy HashTable;

  IdentifierInfoLookup* ExternalLookup;

public:
  /// IdentifierTable ctor - Create the identifier table, populating it with
  /// info about the language keywords for the language specified by LangOpts.
  IdentifierTable(const LangOptions &LangOpts,
                  IdentifierInfoLookup* externalLookup = 0);

  /// \brief Set the external identifier lookup mechanism.
  void setExternalIdentifierLookup(IdentifierInfoLookup *IILookup) {
    ExternalLookup = IILookup;
  }

  llvm::BumpPtrAllocator& getAllocator() {
    return HashTable.getAllocator();
  }

  /// get - Return the identifier token info for the specified named identifier.
  ///
  IdentifierInfo &get(const char *NameStart, const char *NameEnd) {
    llvm::StringMapEntry<IdentifierInfo*> &Entry =
      HashTable.GetOrCreateValue(NameStart, NameEnd);

    IdentifierInfo *II = Entry.getValue();
    if (II) return *II;

    // No entry; if we have an external lookup, look there first.
    if (ExternalLookup) {
      II = ExternalLookup->get(NameStart, NameEnd);
      if (II) {
        // Cache in the StringMap for subsequent lookups.
        Entry.setValue(II);
        return *II;
      }
    }

    // Lookups failed, make a new IdentifierInfo.
    void *Mem = getAllocator().Allocate<IdentifierInfo>();
    II = new (Mem) IdentifierInfo();
    Entry.setValue(II);

    // Make sure getName() knows how to find the IdentifierInfo
    // contents.
    II->Entry = &Entry;

    return *II;
  }

  /// \brief Creates a new IdentifierInfo from the given string.
  ///
  /// This is a lower-level version of get() that requires that this
  /// identifier not be known previously and that does not consult an
  /// external source for identifiers. In particular, external
  /// identifier sources can use this routine to build IdentifierInfo
  /// nodes and then introduce additional information about those
  /// identifiers.
  IdentifierInfo &CreateIdentifierInfo(const char *NameStart,
                                       const char *NameEnd) {
    llvm::StringMapEntry<IdentifierInfo*> &Entry =
      HashTable.GetOrCreateValue(NameStart, NameEnd);

    IdentifierInfo *II = Entry.getValue();
    assert(!II && "IdentifierInfo already exists");

    // Lookups failed, make a new IdentifierInfo.
    void *Mem = getAllocator().Allocate<IdentifierInfo>();
    II = new (Mem) IdentifierInfo();
    Entry.setValue(II);

    // Make sure getName() knows how to find the IdentifierInfo
    // contents.
    II->Entry = &Entry;

    return *II;
  }

  IdentifierInfo &get(const char *Name) {
    return get(Name, Name+strlen(Name));
  }
  IdentifierInfo &get(const std::string &Name) {
    // Don't use c_str() here: no need to be null terminated.
    const char *NameBytes = Name.data();
    return get(NameBytes, NameBytes+Name.size());
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
};

/// Selector - This smart pointer class efficiently represents Objective-C
/// method names. This class will either point to an IdentifierInfo or a
/// MultiKeywordSelector (which is private). This enables us to optimize
/// selectors that take no arguments and selectors that take 1 argument, which
/// accounts for 78% of all selectors in Cocoa.h.
class Selector {
  friend class DiagnosticInfo;

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

  IdentifierInfo *getAsIdentifierInfo() const {
    if (getIdentifierInfoFlag())
      return reinterpret_cast<IdentifierInfo *>(InfoPtr & ~ArgFlags);
    return 0;
  }
  unsigned getIdentifierInfoFlag() const {
    return InfoPtr & ArgFlags;
  }

public:
  friend class SelectorTable; // only the SelectorTable can create these
  friend class DeclarationName; // and the AST's DeclarationName.

  /// The default ctor should only be used when creating data structures that
  ///  will contain selectors.
  Selector() : InfoPtr(0) {}
  Selector(uintptr_t V) : InfoPtr(V) {}

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

  /// \brief Determine whether this is the empty selector.
  bool isNull() const { return InfoPtr == 0; }

  // Predicates to identify the selector type.
  bool isKeywordSelector() const {
    return getIdentifierInfoFlag() != ZeroArg;
  }
  bool isUnarySelector() const {
    return getIdentifierInfoFlag() == ZeroArg;
  }
  unsigned getNumArgs() const;
  IdentifierInfo *getIdentifierInfoForSlot(unsigned argIndex) const;

  /// getAsString - Derive the full selector name (e.g. "foo:bar:") and return
  /// it as an std::string.
  std::string getAsString() const;

  static Selector getEmptyMarker() {
    return Selector(uintptr_t(-1));
  }
  static Selector getTombstoneMarker() {
    return Selector(uintptr_t(-2));
  }
};

/// SelectorTable - This table allows us to fully hide how we implement
/// multi-keyword caching.
class SelectorTable {
  void *Impl;  // Actually a SelectorTableImpl
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

  /// constructSetterName - Return the setter name for the given
  /// identifier, i.e. "set" + Name where the initial character of Name
  /// has been capitalized.
  static Selector constructSetterName(IdentifierTable &Idents,
                                      SelectorTable &SelTable,
                                      const IdentifierInfo *Name) {
    llvm::SmallString<100> SelectorName;
    SelectorName = "set";
    SelectorName.append(Name->getName(), Name->getName()+Name->getLength());
    SelectorName[3] = toupper(SelectorName[3]);
    IdentifierInfo *SetterName =
      &Idents.get(SelectorName.data(),
                  SelectorName.data() + SelectorName.size());
    return SelTable.getUnarySelector(SetterName);
  }
};

/// DeclarationNameExtra - Common base of the MultiKeywordSelector,
/// CXXSpecialName, and CXXOperatorIdName classes, all of which are
/// private classes that describe different kinds of names.
class DeclarationNameExtra {
public:
  /// ExtraKind - The kind of "extra" information stored in the
  /// DeclarationName. See @c ExtraKindOrNumArgs for an explanation of
  /// how these enumerator values are used.
  enum ExtraKind {
    CXXConstructor = 0,
    CXXDestructor,
    CXXConversionFunction,
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
    CXXOperator##Name,
#include "clang/Basic/OperatorKinds.def"
    CXXUsingDirective,
    NUM_EXTRA_KINDS
  };

  /// ExtraKindOrNumArgs - Either the kind of C++ special name or
  /// operator-id (if the value is one of the CXX* enumerators of
  /// ExtraKind), in which case the DeclarationNameExtra is also a
  /// CXXSpecialName (for CXXConstructor, CXXDestructor, or
  /// CXXConversionFunction) or CXXOperatorIdName, it may be also
  /// name common to C++ using-directives (CXXUsingDirective), otherwise
  /// it is NUM_EXTRA_KINDS+NumArgs, where NumArgs is the number of
  /// arguments in the Objective-C selector, in which case the
  /// DeclarationNameExtra is also a MultiKeywordSelector.
  unsigned ExtraKindOrNumArgs;
};

}  // end namespace clang

namespace llvm {
/// Define DenseMapInfo so that Selectors can be used as keys in DenseMap and
/// DenseSets.
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

// Provide PointerLikeTypeTraits for IdentifierInfo pointers, which
// are not guaranteed to be 8-byte aligned.
template<>
class PointerLikeTypeTraits<clang::IdentifierInfo*> {
public:
  static inline void *getAsVoidPointer(clang::IdentifierInfo* P) {
    return P;
  }
  static inline clang::IdentifierInfo *getFromVoidPointer(void *P) {
    return static_cast<clang::IdentifierInfo*>(P);
  }
  enum { NumLowBitsAvailable = 1 };
};

template<>
class PointerLikeTypeTraits<const clang::IdentifierInfo*> {
public:
  static inline const void *getAsVoidPointer(const clang::IdentifierInfo* P) {
    return P;
  }
  static inline const clang::IdentifierInfo *getFromVoidPointer(const void *P) {
    return static_cast<const clang::IdentifierInfo*>(P);
  }
  enum { NumLowBitsAvailable = 1 };
};

}  // end namespace llvm
#endif
