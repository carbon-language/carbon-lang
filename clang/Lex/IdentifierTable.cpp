//===--- IdentifierTable.cpp - Hash table for identifier lookup -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IdentifierInfo, IdentifierVisitor, and
// IdentifierTable interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/IdentifierTable.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Basic/LangOptions.h"
#include <iostream>
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// IdentifierInfo Implementation
//===----------------------------------------------------------------------===//

IdentifierInfo::IdentifierInfo() {
  Macro = 0;
  TokenID = tok::identifier;
  PPID = tok::pp_not_keyword;
  ObjCID = tok::objc_not_keyword;
  IsExtension = false;
  IsPoisoned = false;
  IsOtherTargetMacro = false;
  FETokenInfo = 0;
}

IdentifierInfo::~IdentifierInfo() {
  delete Macro;
}

//===----------------------------------------------------------------------===//
// IdentifierVisitor Implementation
//===----------------------------------------------------------------------===//

IdentifierVisitor::~IdentifierVisitor() {
}


//===----------------------------------------------------------------------===//
// IdentifierTable Implementation
//===----------------------------------------------------------------------===//

/// IdentifierBucket - The hash table consists of an array of these.  If Info is
/// non-null, this is an extant entry, otherwise, it is a hole.
struct IdentifierBucket {
  /// FullHashValue - This remembers the full hash value of the identifier for
  /// easy scanning.
  unsigned FullHashValue;
  
  /// Info - This is a pointer to the actual identifier info object.
  IdentifierInfo *Info;
};

IdentifierTable::IdentifierTable(const LangOptions &LangOpts) {
  HashTableSize = 8192;   // Start with space for 8K identifiers.
  IdentifierBucket *TableArray = new IdentifierBucket[HashTableSize]();
  memset(TableArray, 0, HashTableSize*sizeof(IdentifierBucket));

  TheTable = TableArray;
  NumIdentifiers = 0;
  
  // Populate the identifier table with info about keywords for the current
  // language.
  AddKeywords(LangOpts);
}

IdentifierTable::~IdentifierTable() {
  IdentifierBucket *TableArray = (IdentifierBucket*)TheTable;
  for (unsigned i = 0, e = HashTableSize; i != e; ++i) {
    if (IdentifierInfo *Id = TableArray[i].Info) {
      // Free memory referenced by the identifier (e.g. macro info).
      Id->~IdentifierInfo();
      Allocator.Deallocate(Id);
    }
  }
  delete [] TableArray;
}

/// HashString - Compute a hash code for the specified string.
///
static unsigned HashString(const char *Start, const char *End) {
  unsigned int Result = 0;
  // Perl hash function.
  while (Start != End)
    Result = Result * 33 + *Start++;
  Result = Result + (Result >> 5);
  return Result;
}

IdentifierInfo &IdentifierTable::get(const char *NameStart,
                                     const char *NameEnd) {
  IdentifierBucket *TableArray = (IdentifierBucket*)TheTable;

  unsigned HTSize = HashTableSize;
  unsigned FullHashValue = HashString(NameStart, NameEnd);
  unsigned BucketNo = FullHashValue & (HTSize-1);
  unsigned Length = NameEnd-NameStart;
  
  unsigned ProbeAmt = 1;
  while (1) {
    IdentifierBucket &Bucket = TableArray[BucketNo];
    IdentifierInfo *BucketII = Bucket.Info;
    // If we found an empty bucket, this identifier isn't in the table yet.
    if (BucketII == 0) break;

    // If the full hash value matches, check deeply for a match.  The common
    // case here is that we are only looking at the buckets (for identifier info
    // being non-null and for the full hash value) not at the identifiers.  This
    // is important for cache locality.
    if (Bucket.FullHashValue == FullHashValue &&
        memcmp(BucketII->getName(), NameStart, Length) == 0)
      // We found a match!
      return *BucketII;
   
    // Okay, we didn't find the identifier.  Probe to the next bucket.
    BucketNo = (BucketNo+ProbeAmt) & (HashTableSize-1);
    
    // Use quadratic probing, it has fewer clumping artifacts than linear
    // probing and has good cache behavior in the common case.
    ++ProbeAmt;
  }
  
  // Okay, the identifier doesn't already exist, and BucketNo is the bucket to
  // fill in.  Allocate a new identifier with space for the null-terminated
  // string at the end.
  unsigned AllocSize = sizeof(IdentifierInfo)+Length+1;

  // FIXME: uses GCC extension.
  unsigned Alignment = __alignof__(IdentifierInfo);
  IdentifierInfo *Identifier =
    (IdentifierInfo*)Allocator.Allocate(AllocSize, Alignment);
  new (Identifier) IdentifierInfo();
  ++NumIdentifiers;

  // Copy the string information.
  char *StrBuffer = (char*)(Identifier+1);
  memcpy(StrBuffer, NameStart, Length);
  StrBuffer[Length] = 0;  // Null terminate string.
  
  // Fill in the bucket for the hash table.
  TableArray[BucketNo].Info = Identifier;
  TableArray[BucketNo].FullHashValue = FullHashValue;
  
  // If the hash table is now more than 3/4 full, rehash into a larger table.
  if (NumIdentifiers > HashTableSize*3/4)
    RehashTable();
  
  return *Identifier;
}

IdentifierInfo &IdentifierTable::get(const std::string &Name) {
  // Don't use c_str() here: no need to be null terminated.
  const char *NameBytes = &Name[0];
  unsigned Size = Name.size();
  return get(NameBytes, NameBytes+Size);
}

void IdentifierTable::RehashTable() {
  unsigned NewSize = HashTableSize*2;
  IdentifierBucket *NewTableArray = new IdentifierBucket[NewSize]();
  memset(NewTableArray, 0, NewSize*sizeof(IdentifierBucket));

  // Rehash all the identifier into their new buckets.  Luckily we already have
  // the hash values available :).
  IdentifierBucket *CurTable = (IdentifierBucket *)TheTable;
  for (IdentifierBucket *IB = CurTable, *E = CurTable+HashTableSize;
       IB != E; ++IB) {
    if (IB->Info) {
      // Fast case, bucket available.
      unsigned FullHash = IB->FullHashValue;
      unsigned NewBucket = FullHash & (NewSize-1);
      if (NewTableArray[NewBucket].Info == 0) {
        NewTableArray[FullHash & (NewSize-1)].Info = IB->Info;
        NewTableArray[FullHash & (NewSize-1)].FullHashValue = FullHash;
        continue;
      }
      
      unsigned ProbeSize = 1;
      do {
        NewBucket = (NewBucket + ProbeSize++) & (NewSize-1);
      } while (NewTableArray[NewBucket].Info);
        
      // Finally found a slot.  Fill it in.
      NewTableArray[NewBucket].Info = IB->Info;
      NewTableArray[NewBucket].FullHashValue = FullHash;
    }
  }

  delete[] CurTable;
  
  TheTable = NewTableArray;
  HashTableSize = NewSize;
}


/// VisitIdentifiers - This method walks through all of the identifiers,
/// invoking IV->VisitIdentifier for each of them.
void IdentifierTable::VisitIdentifiers(const IdentifierVisitor &IV) {
  IdentifierBucket *TableArray = (IdentifierBucket*)TheTable;
  for (unsigned i = 0, e = HashTableSize; i != e; ++i) {
    if (IdentifierInfo *Id = TableArray[i].Info)
      IV.VisitIdentifier(*Id);
  }
}

//===----------------------------------------------------------------------===//
// Language Keyword Implementation
//===----------------------------------------------------------------------===//

/// AddKeyword - This method is used to associate a token ID with specific
/// identifiers because they are language keywords.  This causes the lexer to
/// automatically map matching identifiers to specialized token codes.
///
/// The C90/C99/CPP flags are set to 0 if the token should be enabled in the
/// specified langauge, set to 1 if it is an extension in the specified
/// language, and set to 2 if disabled in the specified language.
static void AddKeyword(const std::string &Keyword, tok::TokenKind TokenCode,
                       int C90, int C99, int CXX,
                       const LangOptions &LangOpts, IdentifierTable &Table) {
  int Flags = LangOpts.CPlusPlus ? CXX : (LangOpts.C99 ? C99 : C90);
  
  // Don't add this keyword if disabled in this language or if an extension
  // and extensions are disabled.
  if (Flags + LangOpts.NoExtensions >= 2) return;
  
  const char *Str = &Keyword[0];
  IdentifierInfo &Info = Table.get(Str, Str+Keyword.size());
  Info.setTokenID(TokenCode);
  Info.setIsExtensionToken(Flags == 1);
}

/// AddPPKeyword - Register a preprocessor keyword like "define" "undef" or 
/// "elif".
static void AddPPKeyword(tok::PPKeywordKind PPID, 
                         const char *Name, unsigned NameLen,
                         IdentifierTable &Table) {
  Table.get(Name, Name+NameLen).setPPKeywordID(PPID);
}

/// AddObjCKeyword - Register an Objective-C @keyword like "class" "selector" or 
/// "property".
static void AddObjCKeyword(tok::ObjCKeywordKind ObjCID, 
                           const char *Name, unsigned NameLen,
                           IdentifierTable &Table) {
  Table.get(Name, Name+NameLen).setObjCKeywordID(ObjCID);
}

/// AddKeywords - Add all keywords to the symbol table.
///
void IdentifierTable::AddKeywords(const LangOptions &LangOpts) {
  enum {
    C90Shift = 0,
    EXTC90   = 1 << C90Shift,
    NOTC90   = 2 << C90Shift,
    C99Shift = 2,
    EXTC99   = 1 << C99Shift,
    NOTC99   = 2 << C99Shift,
    CPPShift = 4,
    EXTCPP   = 1 << CPPShift,
    NOTCPP   = 2 << CPPShift,
    Mask     = 3
  };
  
  // Add keywords and tokens for the current language.
#define KEYWORD(NAME, FLAGS) \
  AddKeyword(#NAME, tok::kw_ ## NAME,  \
             ((FLAGS) >> C90Shift) & Mask, \
             ((FLAGS) >> C99Shift) & Mask, \
             ((FLAGS) >> CPPShift) & Mask, LangOpts, *this);
#define ALIAS(NAME, TOK) \
  AddKeyword(NAME, tok::kw_ ## TOK, 0, 0, 0, LangOpts, *this);
#define PPKEYWORD(NAME) \
  AddPPKeyword(tok::pp_##NAME, #NAME, strlen(#NAME), *this);
#define OBJC1_AT_KEYWORD(NAME) \
  if (LangOpts.ObjC1)          \
    AddObjCKeyword(tok::objc_##NAME, #NAME, strlen(#NAME), *this);
#define OBJC2_AT_KEYWORD(NAME) \
  if (LangOpts.ObjC2)          \
    AddObjCKeyword(tok::objc_##NAME, #NAME, strlen(#NAME), *this);
#include "clang/Basic/TokenKinds.def"
}


//===----------------------------------------------------------------------===//
// Stats Implementation
//===----------------------------------------------------------------------===//

/// PrintStats - Print statistics about how well the identifier table is doing
/// at hashing identifiers.
void IdentifierTable::PrintStats() const {
  unsigned NumEmptyBuckets = 0;
  unsigned AverageIdentifierSize = 0;
  unsigned MaxIdentifierLength = 0;
  unsigned NumProbed = 0;
  
  IdentifierBucket *TableArray = (IdentifierBucket*)TheTable;
  for (unsigned i = 0, e = HashTableSize; i != e; ++i) {
    if (TableArray[i].Info == 0) {
      ++NumEmptyBuckets;
      continue;
    }
    IdentifierInfo *Id = TableArray[i].Info;
    unsigned IdLen = strlen(Id->getName());
    AverageIdentifierSize += IdLen;
    if (MaxIdentifierLength < IdLen)
      MaxIdentifierLength = IdLen;

    // Count the number of times something was probed.
    if ((TableArray[i].FullHashValue & (e-1)) != i)
      ++NumProbed;

    // TODO: Figure out maximum times an identifier had to probe for -stats.
  }
  
  std::cerr << "\n*** Identifier Table Stats:\n";
  std::cerr << "# Identifiers:   " << NumIdentifiers << "\n";
  std::cerr << "# Empty Buckets: " << NumEmptyBuckets << "\n";
  std::cerr << "Hash density (#identifiers per bucket): "
            << NumIdentifiers/(double)HashTableSize << "\n";
  std::cerr << "Num probed identifiers: " << NumProbed << " ("
            << NumProbed*100.0/NumIdentifiers << "%)\n";
  std::cerr << "Ave identifier length: "
            << (AverageIdentifierSize/(double)NumIdentifiers) << "\n";
  std::cerr << "Max identifier length: " << MaxIdentifierLength << "\n";
  
  // Compute statistics about the memory allocated for identifiers.
  Allocator.PrintStats();
}


