//===--- IdentifierTable.cpp - Hash table for identifier lookup -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IdentifierInfo, IdentifierVisitor, and
// IdentifierTable interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

using namespace clang;

//===----------------------------------------------------------------------===//
// IdentifierInfo Implementation
//===----------------------------------------------------------------------===//

IdentifierInfo::IdentifierInfo() {
  TokenID = tok::identifier;
  ObjCOrBuiltinID = 0;
  HasMacro = false;
  IsExtension = false;
  IsPoisoned = false;
  IsCPPOperatorKeyword = false;
  NeedsHandleIdentifier = false;
  FETokenInfo = 0;
  Entry = 0;
}

//===----------------------------------------------------------------------===//
// IdentifierTable Implementation
//===----------------------------------------------------------------------===//

IdentifierInfoLookup::~IdentifierInfoLookup() {}

ExternalIdentifierLookup::~ExternalIdentifierLookup() {}

IdentifierTable::IdentifierTable(const LangOptions &LangOpts,
                                 IdentifierInfoLookup* externalLookup)
  : HashTable(8192), // Start with space for 8K identifiers.
    ExternalLookup(externalLookup) {

  // Populate the identifier table with info about keywords for the current
  // language.
  AddKeywords(LangOpts);
}

//===----------------------------------------------------------------------===//
// Language Keyword Implementation
//===----------------------------------------------------------------------===//

// Constants for TokenKinds.def
namespace {
  enum {
    KEYALL = 1,
    KEYC99 = 2,
    KEYCXX = 4,
    KEYCXX0X = 8,
    KEYGNU = 16,
    KEYMS = 32,
    BOOLSUPPORT = 64,
    KEYALTIVEC = 128
  };
}

/// AddKeyword - This method is used to associate a token ID with specific
/// identifiers because they are language keywords.  This causes the lexer to
/// automatically map matching identifiers to specialized token codes.
///
/// The C90/C99/CPP/CPP0x flags are set to 2 if the token should be
/// enabled in the specified langauge, set to 1 if it is an extension
/// in the specified language, and set to 0 if disabled in the
/// specified language.
static void AddKeyword(llvm::StringRef Keyword,
                       tok::TokenKind TokenCode, unsigned Flags,
                       const LangOptions &LangOpts, IdentifierTable &Table) {
  unsigned AddResult = 0;
  if (Flags & KEYALL) AddResult = 2;
  else if (LangOpts.CPlusPlus && (Flags & KEYCXX)) AddResult = 2;
  else if (LangOpts.CPlusPlus0x && (Flags & KEYCXX0X)) AddResult = 2;
  else if (LangOpts.C99 && (Flags & KEYC99)) AddResult = 2;
  else if (LangOpts.GNUKeywords && (Flags & KEYGNU)) AddResult = 1;
  else if (LangOpts.Microsoft && (Flags & KEYMS)) AddResult = 1;
  else if (LangOpts.Bool && (Flags & BOOLSUPPORT)) AddResult = 2;
  else if (LangOpts.AltiVec && (Flags & KEYALTIVEC)) AddResult = 2;

  // Don't add this keyword if disabled in this language.
  if (AddResult == 0) return;

  IdentifierInfo &Info = Table.get(Keyword);
  Info.setTokenID(TokenCode);
  Info.setIsExtensionToken(AddResult == 1);
}

/// AddCXXOperatorKeyword - Register a C++ operator keyword alternative
/// representations.
static void AddCXXOperatorKeyword(llvm::StringRef Keyword,
                                  tok::TokenKind TokenCode,
                                  IdentifierTable &Table) {
  IdentifierInfo &Info = Table.get(Keyword);
  Info.setTokenID(TokenCode);
  Info.setIsCPlusPlusOperatorKeyword();
}

/// AddObjCKeyword - Register an Objective-C @keyword like "class" "selector" or
/// "property".
static void AddObjCKeyword(llvm::StringRef Name,
                           tok::ObjCKeywordKind ObjCID,
                           IdentifierTable &Table) {
  Table.get(Name).setObjCKeywordID(ObjCID);
}

/// AddKeywords - Add all keywords to the symbol table.
///
void IdentifierTable::AddKeywords(const LangOptions &LangOpts) {
  // Add keywords and tokens for the current language.
#define KEYWORD(NAME, FLAGS) \
  AddKeyword(llvm::StringRef(#NAME), tok::kw_ ## NAME,  \
             FLAGS, LangOpts, *this);
#define ALIAS(NAME, TOK, FLAGS) \
  AddKeyword(llvm::StringRef(NAME), tok::kw_ ## TOK,  \
             FLAGS, LangOpts, *this);
#define CXX_KEYWORD_OPERATOR(NAME, ALIAS) \
  if (LangOpts.CXXOperatorNames)          \
    AddCXXOperatorKeyword(llvm::StringRef(#NAME), tok::ALIAS, *this);
#define OBJC1_AT_KEYWORD(NAME) \
  if (LangOpts.ObjC1)          \
    AddObjCKeyword(llvm::StringRef(#NAME), tok::objc_##NAME, *this);
#define OBJC2_AT_KEYWORD(NAME) \
  if (LangOpts.ObjC2)          \
    AddObjCKeyword(llvm::StringRef(#NAME), tok::objc_##NAME, *this);
#include "clang/Basic/TokenKinds.def"
}

tok::PPKeywordKind IdentifierInfo::getPPKeywordID() const {
  // We use a perfect hash function here involving the length of the keyword,
  // the first and third character.  For preprocessor ID's there are no
  // collisions (if there were, the switch below would complain about duplicate
  // case values).  Note that this depends on 'if' being null terminated.

#define HASH(LEN, FIRST, THIRD) \
  (LEN << 5) + (((FIRST-'a') + (THIRD-'a')) & 31)
#define CASE(LEN, FIRST, THIRD, NAME) \
  case HASH(LEN, FIRST, THIRD): \
    return memcmp(Name, #NAME, LEN) ? tok::pp_not_keyword : tok::pp_ ## NAME

  unsigned Len = getLength();
  if (Len < 2) return tok::pp_not_keyword;
  const char *Name = getNameStart();
  switch (HASH(Len, Name[0], Name[2])) {
  default: return tok::pp_not_keyword;
  CASE( 2, 'i', '\0', if);
  CASE( 4, 'e', 'i', elif);
  CASE( 4, 'e', 's', else);
  CASE( 4, 'l', 'n', line);
  CASE( 4, 's', 'c', sccs);
  CASE( 5, 'e', 'd', endif);
  CASE( 5, 'e', 'r', error);
  CASE( 5, 'i', 'e', ident);
  CASE( 5, 'i', 'd', ifdef);
  CASE( 5, 'u', 'd', undef);

  CASE( 6, 'a', 's', assert);
  CASE( 6, 'd', 'f', define);
  CASE( 6, 'i', 'n', ifndef);
  CASE( 6, 'i', 'p', import);
  CASE( 6, 'p', 'a', pragma);

  CASE( 7, 'd', 'f', defined);
  CASE( 7, 'i', 'c', include);
  CASE( 7, 'w', 'r', warning);

  CASE( 8, 'u', 'a', unassert);
  CASE(12, 'i', 'c', include_next);

  CASE(16, '_', 'i', __include_macros);
#undef CASE
#undef HASH
  }
}

//===----------------------------------------------------------------------===//
// Stats Implementation
//===----------------------------------------------------------------------===//

/// PrintStats - Print statistics about how well the identifier table is doing
/// at hashing identifiers.
void IdentifierTable::PrintStats() const {
  unsigned NumBuckets = HashTable.getNumBuckets();
  unsigned NumIdentifiers = HashTable.getNumItems();
  unsigned NumEmptyBuckets = NumBuckets-NumIdentifiers;
  unsigned AverageIdentifierSize = 0;
  unsigned MaxIdentifierLength = 0;

  // TODO: Figure out maximum times an identifier had to probe for -stats.
  for (llvm::StringMap<IdentifierInfo*, llvm::BumpPtrAllocator>::const_iterator
       I = HashTable.begin(), E = HashTable.end(); I != E; ++I) {
    unsigned IdLen = I->getKeyLength();
    AverageIdentifierSize += IdLen;
    if (MaxIdentifierLength < IdLen)
      MaxIdentifierLength = IdLen;
  }

  fprintf(stderr, "\n*** Identifier Table Stats:\n");
  fprintf(stderr, "# Identifiers:   %d\n", NumIdentifiers);
  fprintf(stderr, "# Empty Buckets: %d\n", NumEmptyBuckets);
  fprintf(stderr, "Hash density (#identifiers per bucket): %f\n",
          NumIdentifiers/(double)NumBuckets);
  fprintf(stderr, "Ave identifier length: %f\n",
          (AverageIdentifierSize/(double)NumIdentifiers));
  fprintf(stderr, "Max identifier length: %d\n", MaxIdentifierLength);

  // Compute statistics about the memory allocated for identifiers.
  HashTable.getAllocator().PrintStats();
}

//===----------------------------------------------------------------------===//
// SelectorTable Implementation
//===----------------------------------------------------------------------===//

unsigned llvm::DenseMapInfo<clang::Selector>::getHashValue(clang::Selector S) {
  return DenseMapInfo<void*>::getHashValue(S.getAsOpaquePtr());
}

namespace clang {
/// MultiKeywordSelector - One of these variable length records is kept for each
/// selector containing more than one keyword. We use a folding set
/// to unique aggregate names (keyword selectors in ObjC parlance). Access to
/// this class is provided strictly through Selector.
class MultiKeywordSelector
  : public DeclarationNameExtra, public llvm::FoldingSetNode {
  MultiKeywordSelector(unsigned nKeys) {
    ExtraKindOrNumArgs = NUM_EXTRA_KINDS + nKeys;
  }
public:
  // Constructor for keyword selectors.
  MultiKeywordSelector(unsigned nKeys, IdentifierInfo **IIV) {
    assert((nKeys > 1) && "not a multi-keyword selector");
    ExtraKindOrNumArgs = NUM_EXTRA_KINDS + nKeys;

    // Fill in the trailing keyword array.
    IdentifierInfo **KeyInfo = reinterpret_cast<IdentifierInfo **>(this+1);
    for (unsigned i = 0; i != nKeys; ++i)
      KeyInfo[i] = IIV[i];
  }

  // getName - Derive the full selector name and return it.
  std::string getName() const;

  unsigned getNumArgs() const { return ExtraKindOrNumArgs - NUM_EXTRA_KINDS; }

  typedef IdentifierInfo *const *keyword_iterator;
  keyword_iterator keyword_begin() const {
    return reinterpret_cast<keyword_iterator>(this+1);
  }
  keyword_iterator keyword_end() const {
    return keyword_begin()+getNumArgs();
  }
  IdentifierInfo *getIdentifierInfoForSlot(unsigned i) const {
    assert(i < getNumArgs() && "getIdentifierInfoForSlot(): illegal index");
    return keyword_begin()[i];
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      keyword_iterator ArgTys, unsigned NumArgs) {
    ID.AddInteger(NumArgs);
    for (unsigned i = 0; i != NumArgs; ++i)
      ID.AddPointer(ArgTys[i]);
  }
  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, keyword_begin(), getNumArgs());
  }
};
} // end namespace clang.

unsigned Selector::getNumArgs() const {
  unsigned IIF = getIdentifierInfoFlag();
  if (IIF == ZeroArg)
    return 0;
  if (IIF == OneArg)
    return 1;
  // We point to a MultiKeywordSelector (pointer doesn't contain any flags).
  MultiKeywordSelector *SI = reinterpret_cast<MultiKeywordSelector *>(InfoPtr);
  return SI->getNumArgs();
}

IdentifierInfo *Selector::getIdentifierInfoForSlot(unsigned argIndex) const {
  if (getIdentifierInfoFlag()) {
    assert(argIndex == 0 && "illegal keyword index");
    return getAsIdentifierInfo();
  }
  // We point to a MultiKeywordSelector (pointer doesn't contain any flags).
  MultiKeywordSelector *SI = reinterpret_cast<MultiKeywordSelector *>(InfoPtr);
  return SI->getIdentifierInfoForSlot(argIndex);
}

std::string MultiKeywordSelector::getName() const {
  llvm::SmallString<256> Str;
  llvm::raw_svector_ostream OS(Str);
  for (keyword_iterator I = keyword_begin(), E = keyword_end(); I != E; ++I) {
    if (*I)
      OS << (*I)->getName();
    OS << ':';
  }

  return OS.str();
}

std::string Selector::getAsString() const {
  if (InfoPtr == 0)
    return "<null selector>";

  if (InfoPtr & ArgFlags) {
    IdentifierInfo *II = getAsIdentifierInfo();

    // If the number of arguments is 0 then II is guaranteed to not be null.
    if (getNumArgs() == 0)
      return II->getName();

    if (!II)
      return ":";

    return II->getName().str() + ":";
  }

  // We have a multiple keyword selector (no embedded flags).
  return reinterpret_cast<MultiKeywordSelector *>(InfoPtr)->getName();
}


namespace {
  struct SelectorTableImpl {
    llvm::FoldingSet<MultiKeywordSelector> Table;
    llvm::BumpPtrAllocator Allocator;
  };
} // end anonymous namespace.

static SelectorTableImpl &getSelectorTableImpl(void *P) {
  return *static_cast<SelectorTableImpl*>(P);
}


Selector SelectorTable::getSelector(unsigned nKeys, IdentifierInfo **IIV) {
  if (nKeys < 2)
    return Selector(IIV[0], nKeys);

  SelectorTableImpl &SelTabImpl = getSelectorTableImpl(Impl);

  // Unique selector, to guarantee there is one per name.
  llvm::FoldingSetNodeID ID;
  MultiKeywordSelector::Profile(ID, IIV, nKeys);

  void *InsertPos = 0;
  if (MultiKeywordSelector *SI =
        SelTabImpl.Table.FindNodeOrInsertPos(ID, InsertPos))
    return Selector(SI);

  // MultiKeywordSelector objects are not allocated with new because they have a
  // variable size array (for parameter types) at the end of them.
  unsigned Size = sizeof(MultiKeywordSelector) + nKeys*sizeof(IdentifierInfo *);
  MultiKeywordSelector *SI =
    (MultiKeywordSelector*)SelTabImpl.Allocator.Allocate(Size,
                                         llvm::alignof<MultiKeywordSelector>());
  new (SI) MultiKeywordSelector(nKeys, IIV);
  SelTabImpl.Table.InsertNode(SI, InsertPos);
  return Selector(SI);
}

SelectorTable::SelectorTable() {
  Impl = new SelectorTableImpl();
}

SelectorTable::~SelectorTable() {
  delete &getSelectorTableImpl(Impl);
}

const char *clang::getOperatorSpelling(OverloadedOperatorKind Operator) {
  switch (Operator) {
  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    return 0;

#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
  case OO_##Name: return Spelling;
#include "clang/Basic/OperatorKinds.def"
  }

  return 0;
}

