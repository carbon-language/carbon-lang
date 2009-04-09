//===--- PCHReader.cpp - Precompiled Headers Reader -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHReader class, which reads a precompiled header.
//
//===----------------------------------------------------------------------===//
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/PCHBitCodes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cstdio>

using namespace clang;

//===----------------------------------------------------------------------===//
// Declaration deserialization
//===----------------------------------------------------------------------===//
namespace {
  class VISIBILITY_HIDDEN PCHDeclReader {
    PCHReader &Reader;
    const PCHReader::RecordData &Record;
    unsigned &Idx;

  public:
    PCHDeclReader(PCHReader &Reader, const PCHReader::RecordData &Record,
                  unsigned &Idx)
      : Reader(Reader), Record(Record), Idx(Idx) { }

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
    void VisitNamedDecl(NamedDecl *ND);
    void VisitTypeDecl(TypeDecl *TD);
    void VisitTypedefDecl(TypedefDecl *TD);
    void VisitValueDecl(ValueDecl *VD);
    void VisitVarDecl(VarDecl *VD);

    std::pair<uint64_t, uint64_t> VisitDeclContext(DeclContext *DC);
  };
}

void PCHDeclReader::VisitDecl(Decl *D) {
  D->setDeclContext(cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  D->setLexicalDeclContext(
                     cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  D->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  D->setInvalidDecl(Record[Idx++]);
  // FIXME: hasAttrs
  D->setImplicit(Record[Idx++]);
  D->setAccess((AccessSpecifier)Record[Idx++]);
}

void PCHDeclReader::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  VisitDecl(TU);
}

void PCHDeclReader::VisitNamedDecl(NamedDecl *ND) {
  VisitDecl(ND);
  ND->setDeclName(Reader.ReadDeclarationName(Record, Idx));  
}

void PCHDeclReader::VisitTypeDecl(TypeDecl *TD) {
  VisitNamedDecl(TD);
  // FIXME: circular dependencies here?
  TD->setTypeForDecl(Reader.GetType(Record[Idx++]).getTypePtr());
}

void PCHDeclReader::VisitTypedefDecl(TypedefDecl *TD) {
  VisitTypeDecl(TD);
  TD->setUnderlyingType(Reader.GetType(Record[Idx++]));
}

void PCHDeclReader::VisitValueDecl(ValueDecl *VD) {
  VisitNamedDecl(VD);
  VD->setType(Reader.GetType(Record[Idx++]));
}

void PCHDeclReader::VisitVarDecl(VarDecl *VD) {
  VisitValueDecl(VD);
  VD->setStorageClass((VarDecl::StorageClass)Record[Idx++]);
  VD->setThreadSpecified(Record[Idx++]);
  VD->setCXXDirectInitializer(Record[Idx++]);
  VD->setDeclaredInCondition(Record[Idx++]);
  VD->setPreviousDeclaration(
                         cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  VD->setTypeSpecStartLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

std::pair<uint64_t, uint64_t> 
PCHDeclReader::VisitDeclContext(DeclContext *DC) {
  uint64_t LexicalOffset = Record[Idx++];
  uint64_t VisibleOffset = 0;
  if (DC->getPrimaryContext() == DC)
    VisibleOffset = Record[Idx++];
  return std::make_pair(LexicalOffset, VisibleOffset);
}

// FIXME: use the diagnostics machinery
static bool Error(const char *Str) {
  std::fprintf(stderr, "%s\n", Str);
  return true;
}

/// \brief Read the type-offsets block.
bool PCHReader::ReadTypeOffsets() {
  if (Stream.EnterSubBlock(pch::TYPE_OFFSETS_BLOCK_ID))
    return Error("Malformed block record");

  RecordData Record;
  while (true) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of TYPE_OFFSETS block");
      return false;
    }
    
    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }
    
    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    Record.clear();
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: ignore.
      break;
    case pch::TYPE_OFFSET:
      if (!TypeOffsets.empty())
        return Error("Duplicate TYPE_OFFSETS block");
      TypeOffsets.swap(Record);
      TypeAlreadyLoaded.resize(TypeOffsets.size(), false);
      break;
    }
  }
}

/// \brief Read the decl-offsets block.
bool PCHReader::ReadDeclOffsets() {
  if (Stream.EnterSubBlock(pch::DECL_OFFSETS_BLOCK_ID))
    return Error("Malformed block record");

  RecordData Record;
  while (true) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of DECL_OFFSETS block");
      return false;
    }
    
    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }
    
    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    Record.clear();
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: ignore.
      break;
    case pch::DECL_OFFSET:
      if (!DeclOffsets.empty())
        return Error("Duplicate DECL_OFFSETS block");
      DeclOffsets.swap(Record);
      DeclAlreadyLoaded.resize(DeclOffsets.size(), false);
      break;
    }
  }
}

bool PCHReader::ReadPCHBlock() {
  if (Stream.EnterSubBlock(pch::PCH_BLOCK_ID))
    return Error("Malformed block record");

  // Read all of the records and blocks for the PCH file.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of module block");
      return false;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      switch (Stream.ReadSubBlockID()) {
      case pch::DECLS_BLOCK_ID: // Skip decls block (lazily loaded)
      case pch::TYPES_BLOCK_ID: // Skip types block (lazily loaded)
      default:  // Skip unknown content.
        if (Stream.SkipBlock())
          return Error("Malformed block record");
        break;


      case pch::TYPE_OFFSETS_BLOCK_ID:
        if (ReadTypeOffsets())
          return Error("Malformed type-offsets block");
        break;

      case pch::DECL_OFFSETS_BLOCK_ID:
        if (ReadDeclOffsets())
          return Error("Malformed decl-offsets block");
        break;
      }
    }
  }

  return Error("Premature end of bitstream");
}

PCHReader::~PCHReader() { }

bool PCHReader::ReadPCH(const std::string &FileName) {
  // Open the PCH file.
  std::string ErrStr;
  Buffer.reset(llvm::MemoryBuffer::getFile(FileName.c_str(), &ErrStr));
  if (!Buffer)
    return Error(ErrStr.c_str());

  // Initialize the stream
  Stream.init((const unsigned char *)Buffer->getBufferStart(), 
              (const unsigned char *)Buffer->getBufferEnd());

  // Sniff for the signature.
  if (Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'P' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'H')
    return Error("Not a PCH file");

  // We expect a number of well-defined blocks, though we don't necessarily
  // need to understand them all.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    
    if (Code != llvm::bitc::ENTER_SUBBLOCK)
      return Error("Invalid record at top-level");

    unsigned BlockID = Stream.ReadSubBlockID();
    
    // We only know the PCH subblock ID.
    switch (BlockID) {
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (Stream.ReadBlockInfoBlock())
        return Error("Malformed BlockInfoBlock");
      break;
    case pch::PCH_BLOCK_ID:
      if (ReadPCHBlock())
        return true;
      break;
    default:
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      break;
    }
  }  

  // Load the translation unit declaration
  ReadDeclRecord(DeclOffsets[0], 0);

  return false;
}

/// \brief Read and return the type at the given offset.
///
/// This routine actually reads the record corresponding to the type
/// at the given offset in the bitstream. It is a helper routine for
/// GetType, which deals with reading type IDs.
QualType PCHReader::ReadTypeRecord(uint64_t Offset) {
  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  switch ((pch::TypeCode)Stream.ReadRecord(Code, Record)) {
  case pch::TYPE_FIXED_WIDTH_INT: {
    assert(Record.size() == 2 && "Incorrect encoding of fixed-width int type");
    return Context.getFixedWidthIntType(Record[0], Record[1]);
  }

  case pch::TYPE_COMPLEX: {
    assert(Record.size() == 1 && "Incorrect encoding of complex type");
    QualType ElemType = GetType(Record[0]);
    return Context.getComplexType(ElemType);
  }

  case pch::TYPE_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of pointer type");
    QualType PointeeType = GetType(Record[0]);
    return Context.getPointerType(PointeeType);
  }

  case pch::TYPE_BLOCK_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of block pointer type");
    QualType PointeeType = GetType(Record[0]);
    return Context.getBlockPointerType(PointeeType);
  }

  case pch::TYPE_LVALUE_REFERENCE: {
    assert(Record.size() == 1 && "Incorrect encoding of lvalue reference type");
    QualType PointeeType = GetType(Record[0]);
    return Context.getLValueReferenceType(PointeeType);
  }

  case pch::TYPE_RVALUE_REFERENCE: {
    assert(Record.size() == 1 && "Incorrect encoding of rvalue reference type");
    QualType PointeeType = GetType(Record[0]);
    return Context.getRValueReferenceType(PointeeType);
  }

  case pch::TYPE_MEMBER_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of member pointer type");
    QualType PointeeType = GetType(Record[0]);
    QualType ClassType = GetType(Record[1]);
    return Context.getMemberPointerType(PointeeType, ClassType.getTypePtr());
  }

    // FIXME: Several other kinds of types to deserialize here!
  default:
    assert("Unable to deserialize this type");
    break;
  }

  // Suppress a GCC warning
  return QualType();
}

/// \brief Note that we have loaded the declaration with the given
/// Index.
/// 
/// This routine notes that this declaration has already been loaded,
/// so that future GetDecl calls will return this declaration rather
/// than trying to load a new declaration.
inline void PCHReader::LoadedDecl(unsigned Index, Decl *D) {
  assert(!DeclAlreadyLoaded[Index] && "Decl loaded twice?");
  DeclAlreadyLoaded[Index] = true;
  DeclOffsets[Index] = reinterpret_cast<uint64_t>(D);
}

/// \brief Read the declaration at the given offset from the PCH file.
Decl *PCHReader::ReadDeclRecord(uint64_t Offset, unsigned Index) {
  Decl *D = 0;
  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  unsigned Idx = 0;
  PCHDeclReader Reader(*this, Record, Idx);
  switch ((pch::DeclCode)Stream.ReadRecord(Code, Record)) {
  case pch::DECL_TRANSLATION_UNIT:
    assert(Index == 0 && "Translation unit must be at index 0");
    Reader.VisitTranslationUnitDecl(Context.getTranslationUnitDecl());
    D = Context.getTranslationUnitDecl();
    LoadedDecl(Index, D);
    break;

  case pch::DECL_TYPEDEF: {
    TypedefDecl *Typedef = TypedefDecl::Create(Context, 0, SourceLocation(),
                                               0, QualType());
    LoadedDecl(Index, Typedef);
    Reader.VisitTypedefDecl(Typedef);
    D = Typedef;
    break;
  }

  case pch::DECL_VAR: {
    VarDecl *Var = VarDecl::Create(Context, 0, SourceLocation(), 0, QualType(),
                                   VarDecl::None, SourceLocation());
    LoadedDecl(Index, Var);
    Reader.VisitVarDecl(Var);
    D = Var;
    break;
  }

  default:
    assert(false && "Cannot de-serialize this kind of declaration");
    break;
  }

  // If this declaration is also a declaration context, get the
  // offsets for its tables of lexical and visible declarations.
  if (DeclContext *DC = dyn_cast<DeclContext>(D)) {
    std::pair<uint64_t, uint64_t> Offsets = Reader.VisitDeclContext(DC);
    if (Offsets.first || Offsets.second) {
      DC->setHasExternalLexicalStorage(Offsets.first != 0);
      DC->setHasExternalVisibleStorage(Offsets.second != 0);
      DeclContextOffsets[DC] = Offsets;
    }
  }
  assert(Idx == Record.size());

  return D;
}

QualType PCHReader::GetType(unsigned ID) {
  unsigned Quals = ID & 0x07; 
  unsigned Index = ID >> 3;

  if (Index < pch::NUM_PREDEF_TYPE_IDS) {
    QualType T;
    switch ((pch::PredefinedTypeIDs)Index) {
    case pch::PREDEF_TYPE_NULL_ID: return QualType();
    case pch::PREDEF_TYPE_VOID_ID: T = Context.VoidTy; break;
    case pch::PREDEF_TYPE_BOOL_ID: T = Context.BoolTy; break;

    case pch::PREDEF_TYPE_CHAR_U_ID:
    case pch::PREDEF_TYPE_CHAR_S_ID:
      // FIXME: Check that the signedness of CharTy is correct!
      T = Context.CharTy;
      break;

    case pch::PREDEF_TYPE_UCHAR_ID:      T = Context.UnsignedCharTy;     break;
    case pch::PREDEF_TYPE_USHORT_ID:     T = Context.UnsignedShortTy;    break;
    case pch::PREDEF_TYPE_UINT_ID:       T = Context.UnsignedIntTy;      break;
    case pch::PREDEF_TYPE_ULONG_ID:      T = Context.UnsignedLongTy;     break;
    case pch::PREDEF_TYPE_ULONGLONG_ID:  T = Context.UnsignedLongLongTy; break;
    case pch::PREDEF_TYPE_SCHAR_ID:      T = Context.SignedCharTy;       break;
    case pch::PREDEF_TYPE_WCHAR_ID:      T = Context.WCharTy;            break;
    case pch::PREDEF_TYPE_SHORT_ID:      T = Context.ShortTy;            break;
    case pch::PREDEF_TYPE_INT_ID:        T = Context.IntTy;              break;
    case pch::PREDEF_TYPE_LONG_ID:       T = Context.LongTy;             break;
    case pch::PREDEF_TYPE_LONGLONG_ID:   T = Context.LongLongTy;         break;
    case pch::PREDEF_TYPE_FLOAT_ID:      T = Context.FloatTy;            break;
    case pch::PREDEF_TYPE_DOUBLE_ID:     T = Context.DoubleTy;           break;
    case pch::PREDEF_TYPE_LONGDOUBLE_ID: T = Context.LongDoubleTy;       break;
    case pch::PREDEF_TYPE_OVERLOAD_ID:   T = Context.OverloadTy;         break;
    case pch::PREDEF_TYPE_DEPENDENT_ID:  T = Context.DependentTy;        break;
    }

    assert(!T.isNull() && "Unknown predefined type");
    return T.getQualifiedType(Quals);
  }

  Index -= pch::NUM_PREDEF_TYPE_IDS;
  if (!TypeAlreadyLoaded[Index]) {
    // Load the type from the PCH file.
    TypeOffsets[Index] = reinterpret_cast<uint64_t>(
                             ReadTypeRecord(TypeOffsets[Index]).getTypePtr());
    TypeAlreadyLoaded[Index] = true;
  }
    
  return QualType(reinterpret_cast<Type *>(TypeOffsets[Index]), Quals);
}

Decl *PCHReader::GetDecl(unsigned ID) {
  if (ID == 0)
    return 0;

  unsigned Index = ID - 1;
  if (DeclAlreadyLoaded[Index])
    return reinterpret_cast<Decl *>(DeclOffsets[Index]);

  // Load the declaration from the PCH file.
  return ReadDeclRecord(DeclOffsets[Index], Index);
}

bool PCHReader::ReadDeclsLexicallyInContext(DeclContext *DC,
                                     llvm::SmallVectorImpl<unsigned> &Decls) {
  assert(DC->hasExternalLexicalStorage() && 
         "DeclContext has no lexical decls in storage");
  uint64_t Offset = DeclContextOffsets[DC].first;
  assert(Offset && "DeclContext has no lexical decls in storage");

  // Load the record containing all of the declarations lexically in
  // this context.
  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  unsigned RecCode = Stream.ReadRecord(Code, Record);
  assert(RecCode == pch::DECL_CONTEXT_LEXICAL && "Expected lexical block");

  // Load all of the declaration IDs
  Decls.clear();
  Decls.insert(Decls.end(), Record.begin(), Record.end());
  return false;
}

bool PCHReader::ReadDeclsVisibleInContext(DeclContext *DC,
                           llvm::SmallVectorImpl<VisibleDeclaration> & Decls) {
  assert(DC->hasExternalVisibleStorage() && 
         "DeclContext has no visible decls in storage");
  uint64_t Offset = DeclContextOffsets[DC].second;
  assert(Offset && "DeclContext has no visible decls in storage");

  // Load the record containing all of the declarations visible in
  // this context.
  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  unsigned RecCode = Stream.ReadRecord(Code, Record);
  assert(RecCode == pch::DECL_CONTEXT_VISIBLE && "Expected visible block");
  if (Record.size() == 0)
    return false;  

  Decls.clear();

  unsigned Idx = 0;
  //  llvm::SmallVector<uintptr_t, 16> DeclIDs;
  while (Idx < Record.size()) {
    Decls.push_back(VisibleDeclaration());
    Decls.back().Name = ReadDeclarationName(Record, Idx);

    // FIXME: Don't actually read anything here!
    unsigned Size = Record[Idx++];
    llvm::SmallVector<unsigned, 4> & LoadedDecls
      = Decls.back().Declarations;
    LoadedDecls.reserve(Size);
    for (unsigned I = 0; I < Size; ++I)
      LoadedDecls.push_back(Record[Idx++]);
  }

  return false;
}

void PCHReader::PrintStats() {
  std::fprintf(stderr, "*** PCH Statistics:\n");

  unsigned NumTypesLoaded = std::count(TypeAlreadyLoaded.begin(),
                                       TypeAlreadyLoaded.end(),
                                       true);
  unsigned NumDeclsLoaded = std::count(DeclAlreadyLoaded.begin(),
                                       DeclAlreadyLoaded.end(),
                                       true);
  std::fprintf(stderr, "  %u/%u types read (%f%%)\n",
               NumTypesLoaded, (unsigned)TypeAlreadyLoaded.size(),
               ((float)NumTypesLoaded/(float)TypeAlreadyLoaded.size() * 100));
  std::fprintf(stderr, "  %u/%u declarations read (%f%%)\n",
               NumDeclsLoaded, (unsigned)DeclAlreadyLoaded.size(),
               ((float)NumDeclsLoaded/(float)DeclAlreadyLoaded.size() * 100));
  std::fprintf(stderr, "\n");
}

const IdentifierInfo *PCHReader::GetIdentifierInfo(const RecordData &Record, 
                                                   unsigned &Idx) {
  // FIXME: we need unique IDs for identifiers.
  std::string Str;
  unsigned Length = Record[Idx++];
  Str.resize(Length);
  for (unsigned I = 0; I != Length; ++I)
    Str[I] = Record[Idx++];
  return &Context.Idents.get(Str);
}

DeclarationName 
PCHReader::ReadDeclarationName(const RecordData &Record, unsigned &Idx) {
  DeclarationName::NameKind Kind = (DeclarationName::NameKind)Record[Idx++];
  switch (Kind) {
  case DeclarationName::Identifier:
    return DeclarationName(GetIdentifierInfo(Record, Idx));

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    assert(false && "Unable to de-serialize Objective-C selectors");
    break;

  case DeclarationName::CXXConstructorName:
    return Context.DeclarationNames.getCXXConstructorName(
                                                      GetType(Record[Idx++]));

  case DeclarationName::CXXDestructorName:
    return Context.DeclarationNames.getCXXDestructorName(
                                                      GetType(Record[Idx++]));

  case DeclarationName::CXXConversionFunctionName:
    return Context.DeclarationNames.getCXXConversionFunctionName(
                                                      GetType(Record[Idx++]));

  case DeclarationName::CXXOperatorName:
    return Context.DeclarationNames.getCXXOperatorName(
                                       (OverloadedOperatorKind)Record[Idx++]);

  case DeclarationName::CXXUsingDirective:
    return DeclarationName::getUsingDirectiveName();
  }

  // Required to silence GCC warning
  return DeclarationName();
}
