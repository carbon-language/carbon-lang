//===--- CGDebugInfo.cpp - Emit Debug Information for a Module ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the debug information generation while generating code.
//
//===----------------------------------------------------------------------===//

#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompileOptions.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/System/Path.h"
#include "llvm/Target/TargetMachine.h"
using namespace clang;
using namespace clang::CodeGen;

CGDebugInfo::CGDebugInfo(CodeGenModule *m)
  : M(m), isMainCompileUnitCreated(false), DebugFactory(M->getModule()),
    BlockLiteralGenericSet(false) {
}

CGDebugInfo::~CGDebugInfo() {
  assert(RegionStack.empty() && "Region stack mismatch, stack not empty!");
}

void CGDebugInfo::setLocation(SourceLocation Loc) {
  if (Loc.isValid())
    CurLoc = M->getContext().getSourceManager().getInstantiationLoc(Loc);
}

/// getOrCreateCompileUnit - Get the compile unit from the cache or create a new
/// one if necessary. This returns null for invalid source locations.
llvm::DICompileUnit CGDebugInfo::getOrCreateCompileUnit(SourceLocation Loc) {
  // Get source file information.
  const char *FileName =  "<unknown>";
  SourceManager &SM = M->getContext().getSourceManager();
  unsigned FID = 0;
  if (Loc.isValid()) {
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    FileName = PLoc.getFilename();
    FID = PLoc.getIncludeLoc().getRawEncoding();
  }

  // See if this compile unit has been used before.
  llvm::DICompileUnit &Unit = CompileUnitCache[FID];
  if (!Unit.isNull()) return Unit;

  // Get absolute path name.
  llvm::sys::Path AbsFileName(FileName);
  if (!AbsFileName.isAbsolute()) {
    llvm::sys::Path tmp = llvm::sys::Path::GetCurrentDirectory();
    tmp.appendComponent(FileName);
    AbsFileName = tmp;
  }

  // See if thie compile unit is representing main source file. Each source
  // file has corresponding compile unit. There is only one main source
  // file at a time.
  bool isMain = false;
  const LangOptions &LO = M->getLangOptions();
  const char *MainFileName = LO.getMainFileName();
  if (isMainCompileUnitCreated == false) {
    if (MainFileName) {
      if (!strcmp(AbsFileName.getLast().c_str(), MainFileName))
        isMain = true;
    } else {
      if (Loc.isValid() && SM.isFromMainFile(Loc))
        isMain = true;
    }
    if (isMain)
      isMainCompileUnitCreated = true;
  }

  unsigned LangTag;
  if (LO.CPlusPlus) {
    if (LO.ObjC1)
      LangTag = llvm::dwarf::DW_LANG_ObjC_plus_plus;
    else
      LangTag = llvm::dwarf::DW_LANG_C_plus_plus;
  } else if (LO.ObjC1) {
    LangTag = llvm::dwarf::DW_LANG_ObjC;
  } else if (LO.C99) {
    LangTag = llvm::dwarf::DW_LANG_C99;
  } else {
    LangTag = llvm::dwarf::DW_LANG_C89;
  }

  std::string Producer = "clang " CLANG_VERSION_STRING;
  bool isOptimized = LO.Optimize;
  const char *Flags = "";   // FIXME: Encode command line options.

  // Figure out which version of the ObjC runtime we have.
  unsigned RuntimeVers = 0;
  if (LO.ObjC1)
    RuntimeVers = LO.ObjCNonFragileABI ? 2 : 1;

  // Create new compile unit.
  return Unit = DebugFactory.CreateCompileUnit(LangTag, AbsFileName.getLast(),
                                               AbsFileName.getDirname(),
                                               Producer, isMain, isOptimized,
                                               Flags, RuntimeVers);
}

/// CreateType - Get the Basic type from the cache or create a new
/// one if necessary.
llvm::DIType CGDebugInfo::CreateType(const BuiltinType *BT,
                                     llvm::DICompileUnit Unit) {
  unsigned Encoding = 0;
  switch (BT->getKind()) {
  default:
  case BuiltinType::Void:
    return llvm::DIType();
  case BuiltinType::UChar:
  case BuiltinType::Char_U: Encoding = llvm::dwarf::DW_ATE_unsigned_char; break;
  case BuiltinType::Char_S:
  case BuiltinType::SChar: Encoding = llvm::dwarf::DW_ATE_signed_char; break;
  case BuiltinType::UShort:
  case BuiltinType::UInt:
  case BuiltinType::ULong:
  case BuiltinType::ULongLong: Encoding = llvm::dwarf::DW_ATE_unsigned; break;
  case BuiltinType::Short:
  case BuiltinType::Int:
  case BuiltinType::Long:
  case BuiltinType::LongLong:  Encoding = llvm::dwarf::DW_ATE_signed; break;
  case BuiltinType::Bool:      Encoding = llvm::dwarf::DW_ATE_boolean; break;
  case BuiltinType::Float:
  case BuiltinType::Double:    Encoding = llvm::dwarf::DW_ATE_float; break;
  }
  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(BT);
  uint64_t Align = M->getContext().getTypeAlign(BT);
  uint64_t Offset = 0;

  return DebugFactory.CreateBasicType(Unit,
                                  BT->getName(M->getContext().getLangOptions()),
                                      Unit, 0, Size, Align,
                                      Offset, /*flags*/ 0, Encoding);
}

llvm::DIType CGDebugInfo::CreateType(const ComplexType *Ty,
                                     llvm::DICompileUnit Unit) {
  // Bit size, align and offset of the type.
  unsigned Encoding = llvm::dwarf::DW_ATE_complex_float;
  if (Ty->isComplexIntegerType())
    Encoding = llvm::dwarf::DW_ATE_lo_user;

  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);
  uint64_t Offset = 0;

  return DebugFactory.CreateBasicType(Unit, "complex",
                                      Unit, 0, Size, Align,
                                      Offset, /*flags*/ 0, Encoding);
}

/// getOrCreateCVRType - Get the CVR qualified type from the cache or create
/// a new one if necessary.
llvm::DIType CGDebugInfo::CreateCVRType(QualType Ty, llvm::DICompileUnit Unit) {
  // We will create one Derived type for one qualifier and recurse to handle any
  // additional ones.
  llvm::DIType FromTy;
  unsigned Tag;
  if (Ty.isConstQualified()) {
    Tag = llvm::dwarf::DW_TAG_const_type;
    Ty.removeConst();
    FromTy = getOrCreateType(Ty, Unit);
  } else if (Ty.isVolatileQualified()) {
    Tag = llvm::dwarf::DW_TAG_volatile_type;
    Ty.removeVolatile();
    FromTy = getOrCreateType(Ty, Unit);
  } else {
    assert(Ty.isRestrictQualified() && "Unknown type qualifier for debug info");
    Tag = llvm::dwarf::DW_TAG_restrict_type;
    Ty.removeRestrict();
    FromTy = getOrCreateType(Ty, Unit);
  }

  // No need to fill in the Name, Line, Size, Alignment, Offset in case of
  // CVR derived types.
  return DebugFactory.CreateDerivedType(Tag, Unit, "", llvm::DICompileUnit(),
                                        0, 0, 0, 0, 0, FromTy);
}

llvm::DIType CGDebugInfo::CreateType(const ObjCObjectPointerType *Ty,
                                     llvm::DICompileUnit Unit) {
  llvm::DIType EltTy = getOrCreateType(Ty->getPointeeType(), Unit);

  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);

  return DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type, Unit,
                                        "", llvm::DICompileUnit(),
                                        0, Size, Align, 0, 0, EltTy);
}

llvm::DIType CGDebugInfo::CreateType(const PointerType *Ty,
                                     llvm::DICompileUnit Unit) {
  llvm::DIType EltTy = getOrCreateType(Ty->getPointeeType(), Unit);

  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);

  return DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type, Unit,
                                        "", llvm::DICompileUnit(),
                                        0, Size, Align, 0, 0, EltTy);
}

llvm::DIType CGDebugInfo::CreateType(const BlockPointerType *Ty,
                                     llvm::DICompileUnit Unit) {
  if (BlockLiteralGenericSet)
    return BlockLiteralGeneric;

  llvm::DICompileUnit DefUnit;
  unsigned Tag = llvm::dwarf::DW_TAG_structure_type;

  llvm::SmallVector<llvm::DIDescriptor, 5> EltTys;

  llvm::DIType FieldTy;

  QualType FType;
  uint64_t FieldSize, FieldOffset;
  unsigned FieldAlign;

  llvm::DIArray Elements;
  llvm::DIType EltTy, DescTy;

  FieldOffset = 0;
  FType = M->getContext().UnsignedLongTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = M->getContext().getTypeSize(FType);
  FieldAlign = M->getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "reserved", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = M->getContext().UnsignedLongTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = M->getContext().getTypeSize(FType);
  FieldAlign = M->getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "Size", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  Elements = DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());
  EltTys.clear();

  EltTy = DebugFactory.CreateCompositeType(Tag, Unit, "__block_descriptor",
                                           DefUnit, 0, FieldOffset, 0, 0, 0,
                                           llvm::DIType(), Elements);

  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);

  DescTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type,
                                          Unit, "", llvm::DICompileUnit(),
                                          0, Size, Align, 0, 0, EltTy);

  FieldOffset = 0;
  FType = M->getContext().getPointerType(M->getContext().VoidTy);
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = M->getContext().getTypeSize(FType);
  FieldAlign = M->getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__isa", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = M->getContext().IntTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = M->getContext().getTypeSize(FType);
  FieldAlign = M->getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__flags", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = M->getContext().IntTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = M->getContext().getTypeSize(FType);
  FieldAlign = M->getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__reserved", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = M->getContext().getPointerType(M->getContext().VoidTy);
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = M->getContext().getTypeSize(FType);
  FieldAlign = M->getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__FuncPtr", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = M->getContext().getPointerType(M->getContext().VoidTy);
  FieldTy = DescTy;
  FieldSize = M->getContext().getTypeSize(Ty);
  FieldAlign = M->getContext().getTypeAlign(Ty);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__descriptor", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  Elements = DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

  EltTy = DebugFactory.CreateCompositeType(Tag, Unit, "__block_literal_generic",
                                           DefUnit, 0, FieldOffset, 0, 0, 0,
                                           llvm::DIType(), Elements);

  BlockLiteralGenericSet = true;
  BlockLiteralGeneric
    = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type, Unit,
                                     "", llvm::DICompileUnit(),
                                     0, Size, Align, 0, 0, EltTy);
  return BlockLiteralGeneric;
}

llvm::DIType CGDebugInfo::CreateType(const TypedefType *Ty,
                                     llvm::DICompileUnit Unit) {
  // Typedefs are derived from some other type.  If we have a typedef of a
  // typedef, make sure to emit the whole chain.
  llvm::DIType Src = getOrCreateType(Ty->getDecl()->getUnderlyingType(), Unit);

  // We don't set size information, but do specify where the typedef was
  // declared.
  std::string TyName = Ty->getDecl()->getNameAsString();
  SourceLocation DefLoc = Ty->getDecl()->getLocation();
  llvm::DICompileUnit DefUnit = getOrCreateCompileUnit(DefLoc);

  SourceManager &SM = M->getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(DefLoc);
  unsigned Line = PLoc.isInvalid() ? 0 : PLoc.getLine();

  return DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_typedef, Unit,
                                        TyName, DefUnit, Line, 0, 0, 0, 0, Src);
}

llvm::DIType CGDebugInfo::CreateType(const FunctionType *Ty,
                                     llvm::DICompileUnit Unit) {
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  // Add the result type at least.
  EltTys.push_back(getOrCreateType(Ty->getResultType(), Unit));

  // Set up remainder of arguments if there is a prototype.
  // FIXME: IF NOT, HOW IS THIS REPRESENTED?  llvm-gcc doesn't represent '...'!
  if (const FunctionProtoType *FTP = dyn_cast<FunctionProtoType>(Ty)) {
    for (unsigned i = 0, e = FTP->getNumArgs(); i != e; ++i)
      EltTys.push_back(getOrCreateType(FTP->getArgType(i), Unit));
  } else {
    // FIXME: Handle () case in C.  llvm-gcc doesn't do it either.
  }

  llvm::DIArray EltTypeArray =
    DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_subroutine_type,
                                          Unit, "", llvm::DICompileUnit(),
                                          0, 0, 0, 0, 0,
                                          llvm::DIType(), EltTypeArray);
}

/// CreateType - get structure or union type.
llvm::DIType CGDebugInfo::CreateType(const RecordType *Ty,
                                     llvm::DICompileUnit Unit) {
  RecordDecl *Decl = Ty->getDecl();

  unsigned Tag;
  if (Decl->isStruct())
    Tag = llvm::dwarf::DW_TAG_structure_type;
  else if (Decl->isUnion())
    Tag = llvm::dwarf::DW_TAG_union_type;
  else {
    assert(Decl->isClass() && "Unknown RecordType!");
    Tag = llvm::dwarf::DW_TAG_class_type;
  }

  SourceManager &SM = M->getContext().getSourceManager();

  // Get overall information about the record type for the debug info.
  std::string Name = Decl->getNameAsString();

  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  llvm::DICompileUnit DefUnit;
  unsigned Line = 0;
  if (!PLoc.isInvalid()) {
    DefUnit = getOrCreateCompileUnit(Decl->getLocation());
    Line = PLoc.getLine();
  }

  // Records and classes and unions can all be recursive.  To handle them, we
  // first generate a debug descriptor for the struct as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the struct is recursive) and replace all
  // uses of the forward declaration with the final definition.
  llvm::DICompositeType FwdDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, Name, DefUnit, Line, 0, 0, 0, 0,
                                     llvm::DIType(), llvm::DIArray());

  // If this is just a forward declaration, return it.
  if (!Decl->getDefinition(M->getContext()))
    return FwdDecl;

  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl.getNode();

  // Convert all the elements.
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  const ASTRecordLayout &RL = M->getContext().getASTRecordLayout(Decl);

  unsigned FieldNo = 0;
  for (RecordDecl::field_iterator I = Decl->field_begin(),
                                  E = Decl->field_end();
       I != E; ++I, ++FieldNo) {
    FieldDecl *Field = *I;
    llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);

    std::string FieldName = Field->getNameAsString();

    // Ignore unnamed fields.
    if (FieldName.empty())
      continue;

    // Get the location for the field.
    SourceLocation FieldDefLoc = Field->getLocation();
    PresumedLoc PLoc = SM.getPresumedLoc(FieldDefLoc);
    llvm::DICompileUnit FieldDefUnit;
    unsigned FieldLine = 0;

    if (!PLoc.isInvalid()) {
      FieldDefUnit = getOrCreateCompileUnit(FieldDefLoc);
      FieldLine = PLoc.getLine();
    }

    QualType FType = Field->getType();
    uint64_t FieldSize = 0;
    unsigned FieldAlign = 0;
    if (!FType->isIncompleteArrayType()) {

      // Bit size, align and offset of the type.
      FieldSize = M->getContext().getTypeSize(FType);
      Expr *BitWidth = Field->getBitWidth();
      if (BitWidth)
        FieldSize = BitWidth->EvaluateAsInt(M->getContext()).getZExtValue();

      FieldAlign =  M->getContext().getTypeAlign(FType);
    }

    uint64_t FieldOffset = RL.getFieldOffset(FieldNo);

    // Create a DW_TAG_member node to remember the offset of this field in the
    // struct.  FIXME: This is an absolutely insane way to capture this
    // information.  When we gut debug info, this should be fixed.
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             FieldName, FieldDefUnit,
                                             FieldLine, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
  }

  llvm::DIArray Elements =
    DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);

  llvm::DICompositeType RealDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, Name, DefUnit, Line, Size,
                                     Align, 0, 0, llvm::DIType(), Elements);

  // Update TypeCache.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = RealDecl.getNode();

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  FwdDecl.replaceAllUsesWith(RealDecl);

  return RealDecl;
}

/// CreateType - get objective-c interface type.
llvm::DIType CGDebugInfo::CreateType(const ObjCInterfaceType *Ty,
                                     llvm::DICompileUnit Unit) {
  ObjCInterfaceDecl *Decl = Ty->getDecl();

  unsigned Tag = llvm::dwarf::DW_TAG_structure_type;
  SourceManager &SM = M->getContext().getSourceManager();

  // Get overall information about the record type for the debug info.
  std::string Name = Decl->getNameAsString();

  llvm::DICompileUnit DefUnit = getOrCreateCompileUnit(Decl->getLocation());
  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  unsigned Line = PLoc.isInvalid() ? 0 : PLoc.getLine();


  unsigned RuntimeLang = DefUnit.getLanguage();

  // To handle recursive interface, we
  // first generate a debug descriptor for the struct as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the struct is recursive) and replace all
  // uses of the forward declaration with the final definition.
  llvm::DICompositeType FwdDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, Name, DefUnit, Line, 0, 0, 0, 0,
                                     llvm::DIType(), llvm::DIArray(),
                                     RuntimeLang);

  // If this is just a forward declaration, return it.
  if (Decl->isForwardDecl())
    return FwdDecl;

  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl.getNode();

  // Convert all the elements.
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  ObjCInterfaceDecl *SClass = Decl->getSuperClass();
  if (SClass) {
    llvm::DIType SClassTy =
      getOrCreateType(M->getContext().getObjCInterfaceType(SClass), Unit);
    llvm::DIType InhTag =
      DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_inheritance,
                                     Unit, "", llvm::DICompileUnit(), 0, 0, 0,
                                     0 /* offset */, 0, SClassTy);
    EltTys.push_back(InhTag);
  }

  const ASTRecordLayout &RL = M->getContext().getASTObjCInterfaceLayout(Decl);

  unsigned FieldNo = 0;
  for (ObjCInterfaceDecl::ivar_iterator I = Decl->ivar_begin(),
         E = Decl->ivar_end();  I != E; ++I, ++FieldNo) {
    ObjCIvarDecl *Field = *I;
    llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);

    std::string FieldName = Field->getNameAsString();

    // Ignore unnamed fields.
    if (FieldName.empty())
      continue;

    // Get the location for the field.
    SourceLocation FieldDefLoc = Field->getLocation();
    llvm::DICompileUnit FieldDefUnit = getOrCreateCompileUnit(FieldDefLoc);
    PresumedLoc PLoc = SM.getPresumedLoc(FieldDefLoc);
    unsigned FieldLine = PLoc.isInvalid() ? 0 : PLoc.getLine();


    QualType FType = Field->getType();
    uint64_t FieldSize = 0;
    unsigned FieldAlign = 0;

    if (!FType->isIncompleteArrayType()) {

      // Bit size, align and offset of the type.
      FieldSize = M->getContext().getTypeSize(FType);
      Expr *BitWidth = Field->getBitWidth();
      if (BitWidth)
        FieldSize = BitWidth->EvaluateAsInt(M->getContext()).getZExtValue();

      FieldAlign =  M->getContext().getTypeAlign(FType);
    }

    uint64_t FieldOffset = RL.getFieldOffset(FieldNo);

    unsigned Flags = 0;
    if (Field->getAccessControl() == ObjCIvarDecl::Protected)
      Flags = llvm::DIType::FlagProtected;
    else if (Field->getAccessControl() == ObjCIvarDecl::Private)
      Flags = llvm::DIType::FlagPrivate;

    // Create a DW_TAG_member node to remember the offset of this field in the
    // struct.  FIXME: This is an absolutely insane way to capture this
    // information.  When we gut debug info, this should be fixed.
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             FieldName, FieldDefUnit,
                                             FieldLine, FieldSize, FieldAlign,
                                             FieldOffset, Flags, FieldTy);
    EltTys.push_back(FieldTy);
  }

  llvm::DIArray Elements =
    DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(Ty);
  uint64_t Align = M->getContext().getTypeAlign(Ty);

  llvm::DICompositeType RealDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, Name, DefUnit, Line, Size,
                                     Align, 0, 0, llvm::DIType(), Elements,
                                     RuntimeLang);

  // Update TypeCache.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = RealDecl.getNode();

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  FwdDecl.replaceAllUsesWith(RealDecl);

  return RealDecl;
}

llvm::DIType CGDebugInfo::CreateType(const EnumType *Ty,
                                     llvm::DICompileUnit Unit) {
  EnumDecl *Decl = Ty->getDecl();

  llvm::SmallVector<llvm::DIDescriptor, 32> Enumerators;

  // Create DIEnumerator elements for each enumerator.
  for (EnumDecl::enumerator_iterator
         Enum = Decl->enumerator_begin(), EnumEnd = Decl->enumerator_end();
       Enum != EnumEnd; ++Enum) {
    Enumerators.push_back(DebugFactory.CreateEnumerator(Enum->getNameAsString(),
                                            Enum->getInitVal().getZExtValue()));
  }

  // Return a CompositeType for the enum itself.
  llvm::DIArray EltArray =
    DebugFactory.GetOrCreateArray(Enumerators.data(), Enumerators.size());

  std::string EnumName = Decl->getNameAsString();
  SourceLocation DefLoc = Decl->getLocation();
  llvm::DICompileUnit DefUnit = getOrCreateCompileUnit(DefLoc);
  SourceManager &SM = M->getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(DefLoc);
  unsigned Line = PLoc.isInvalid() ? 0 : PLoc.getLine();


  // Size and align of the type.
  uint64_t Size = 0;
  unsigned Align = 0;
  if (!Ty->isIncompleteType()) {
    Size = M->getContext().getTypeSize(Ty);
    Align = M->getContext().getTypeAlign(Ty);
  }

  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_enumeration_type,
                                          Unit, EnumName, DefUnit, Line,
                                          Size, Align, 0, 0,
                                          llvm::DIType(), EltArray);
}

llvm::DIType CGDebugInfo::CreateType(const TagType *Ty,
                                     llvm::DICompileUnit Unit) {
  if (const RecordType *RT = dyn_cast<RecordType>(Ty))
    return CreateType(RT, Unit);
  else if (const EnumType *ET = dyn_cast<EnumType>(Ty))
    return CreateType(ET, Unit);

  return llvm::DIType();
}

llvm::DIType CGDebugInfo::CreateType(const ArrayType *Ty,
                                     llvm::DICompileUnit Unit) {
  uint64_t Size;
  uint64_t Align;


  // FIXME: make getTypeAlign() aware of VLAs and incomplete array types
  if (const VariableArrayType *VAT = dyn_cast<VariableArrayType>(Ty)) {
    Size = 0;
    Align =
      M->getContext().getTypeAlign(M->getContext().getBaseElementType(VAT));
  } else if (Ty->isIncompleteArrayType()) {
    Size = 0;
    Align = M->getContext().getTypeAlign(Ty->getElementType());
  } else {
    // Size and align of the whole array, not the element type.
    Size = M->getContext().getTypeSize(Ty);
    Align = M->getContext().getTypeAlign(Ty);
  }

  // Add the dimensions of the array.  FIXME: This loses CV qualifiers from
  // interior arrays, do we care?  Why aren't nested arrays represented the
  // obvious/recursive way?
  llvm::SmallVector<llvm::DIDescriptor, 8> Subscripts;
  QualType EltTy(Ty, 0);
  while ((Ty = dyn_cast<ArrayType>(EltTy))) {
    uint64_t Upper = 0;
    if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(Ty))
      if (CAT->getSize().getZExtValue())
        Upper = CAT->getSize().getZExtValue() - 1;
    // FIXME: Verify this is right for VLAs.
    Subscripts.push_back(DebugFactory.GetOrCreateSubrange(0, Upper));
    EltTy = Ty->getElementType();
  }

  llvm::DIArray SubscriptArray =
    DebugFactory.GetOrCreateArray(Subscripts.data(), Subscripts.size());

  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_array_type,
                                          Unit, "", llvm::DICompileUnit(),
                                          0, Size, Align, 0, 0,
                                          getOrCreateType(EltTy, Unit),
                                          SubscriptArray);
}


/// getOrCreateType - Get the type from the cache or create a new
/// one if necessary.
llvm::DIType CGDebugInfo::getOrCreateType(QualType Ty,
                                          llvm::DICompileUnit Unit) {
  if (Ty.isNull())
    return llvm::DIType();

  // Check for existing entry.
  std::map<void *, llvm::WeakVH>::iterator it =
    TypeCache.find(Ty.getAsOpaquePtr());
  if (it != TypeCache.end()) {
    // Verify that the debug info still exists.
    if (&*it->second)
      return llvm::DIType(cast<llvm::MDNode>(it->second));
  }

  // Otherwise create the type.
  llvm::DIType Res = CreateTypeNode(Ty, Unit);
  TypeCache.insert(std::make_pair(Ty.getAsOpaquePtr(), Res.getNode()));
  return Res;
}

/// getOrCreateTypeNode - Get the type metadata node from the cache or create a
/// new one if necessary.
llvm::DIType CGDebugInfo::CreateTypeNode(QualType Ty,
                                         llvm::DICompileUnit Unit) {
  // Handle CVR qualifiers, which recursively handles what they refer to.
  if (Ty.getCVRQualifiers())
    return CreateCVRType(Ty, Unit);

  // Work out details of type.
  switch (Ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    assert(false && "Dependent types cannot show up in debug information");

  default:
  case Type::LValueReference:
  case Type::RValueReference:
  case Type::Vector:
  case Type::ExtVector:
  case Type::FixedWidthInt:
  case Type::MemberPointer:
  case Type::TemplateSpecialization:
  case Type::QualifiedName:
    // Unsupported types
    return llvm::DIType();
  case Type::ObjCObjectPointer:
    return CreateType(cast<ObjCObjectPointerType>(Ty), Unit);
  case Type::ObjCInterface:
    return CreateType(cast<ObjCInterfaceType>(Ty), Unit);
  case Type::Builtin: return CreateType(cast<BuiltinType>(Ty), Unit);
  case Type::Complex: return CreateType(cast<ComplexType>(Ty), Unit);
  case Type::Pointer: return CreateType(cast<PointerType>(Ty), Unit);
  case Type::BlockPointer:
    return CreateType(cast<BlockPointerType>(Ty), Unit);
  case Type::Typedef: return CreateType(cast<TypedefType>(Ty), Unit);
  case Type::Record:
  case Type::Enum:
    return CreateType(cast<TagType>(Ty), Unit);
  case Type::FunctionProto:
  case Type::FunctionNoProto:
    return CreateType(cast<FunctionType>(Ty), Unit);
  case Type::Elaborated:
    return getOrCreateType(cast<ElaboratedType>(Ty)->getUnderlyingType(),
                           Unit);

  case Type::ConstantArray:
  case Type::ConstantArrayWithExpr:
  case Type::ConstantArrayWithoutExpr:
  case Type::VariableArray:
  case Type::IncompleteArray:
    return CreateType(cast<ArrayType>(Ty), Unit);
  case Type::TypeOfExpr:
    return getOrCreateType(cast<TypeOfExprType>(Ty)->getUnderlyingExpr()
                           ->getType(), Unit);
  case Type::TypeOf:
    return getOrCreateType(cast<TypeOfType>(Ty)->getUnderlyingType(), Unit);
  case Type::Decltype:
    return getOrCreateType(cast<DecltypeType>(Ty)->getUnderlyingType(), Unit);
  }
}

/// EmitFunctionStart - Constructs the debug code for entering a function -
/// "llvm.dbg.func.start.".
void CGDebugInfo::EmitFunctionStart(const char *Name, QualType ReturnType,
                                    llvm::Function *Fn,
                                    CGBuilderTy &Builder) {
  const char *LinkageName = Name;

  // Skip the asm prefix if it exists.
  //
  // FIXME: This should probably be the unmangled name?
  if (Name[0] == '\01')
    ++Name;

  // FIXME: Why is this using CurLoc???
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(CurLoc);
  SourceManager &SM = M->getContext().getSourceManager();
  unsigned LineNo = SM.getPresumedLoc(CurLoc).getLine();

  llvm::DISubprogram SP =
    DebugFactory.CreateSubprogram(Unit, Name, Name, LinkageName, Unit, LineNo,
                                  getOrCreateType(ReturnType, Unit),
                                  Fn->hasInternalLinkage(), true/*definition*/);

  DebugFactory.InsertSubprogramStart(SP, Builder.GetInsertBlock());

  // Push function on region stack.
  RegionStack.push_back(SP);
}


void CGDebugInfo::EmitStopPoint(llvm::Function *Fn, CGBuilderTy &Builder) {
  if (CurLoc.isInvalid() || CurLoc.isMacroID()) return;

  // Don't bother if things are the same as last time.
  SourceManager &SM = M->getContext().getSourceManager();
  if (CurLoc == PrevLoc
       || (SM.getInstantiationLineNumber(CurLoc) ==
           SM.getInstantiationLineNumber(PrevLoc)
           && SM.isFromSameFile(CurLoc, PrevLoc)))
    return;

  // Update last state.
  PrevLoc = CurLoc;

  // Get the appropriate compile unit.
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(CurLoc);
  PresumedLoc PLoc = SM.getPresumedLoc(CurLoc);
  DebugFactory.InsertStopPoint(Unit, PLoc.getLine(), PLoc.getColumn(),
                               Builder.GetInsertBlock());
}

/// EmitRegionStart- Constructs the debug code for entering a declarative
/// region - "llvm.dbg.region.start.".
void CGDebugInfo::EmitRegionStart(llvm::Function *Fn, CGBuilderTy &Builder) {
  llvm::DIDescriptor D;
  if (!RegionStack.empty())
    D = RegionStack.back();
  D = DebugFactory.CreateLexicalBlock(D);
  RegionStack.push_back(D);
  DebugFactory.InsertRegionStart(D, Builder.GetInsertBlock());
}

/// EmitRegionEnd - Constructs the debug code for exiting a declarative
/// region - "llvm.dbg.region.end."
void CGDebugInfo::EmitRegionEnd(llvm::Function *Fn, CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Provide an region stop point.
  EmitStopPoint(Fn, Builder);

  DebugFactory.InsertRegionEnd(RegionStack.back(), Builder.GetInsertBlock());
  RegionStack.pop_back();
}

/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const VarDecl *Decl, unsigned Tag,
                              llvm::Value *Storage, CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Do not emit variable debug information while generating optimized code.
  // The llvm optimizer and code generator are not yet ready to support
  // optimized code debugging.
  const CompileOptions &CO = M->getCompileOpts();
  if (CO.OptimizationLevel)
    return;

  llvm::DICompileUnit Unit = getOrCreateCompileUnit(Decl->getLocation());
  QualType Type = Decl->getType();
  llvm::DIType Ty = getOrCreateType(Type, Unit);
  if (Decl->hasAttr<BlocksAttr>()) {
    llvm::DICompileUnit DefUnit;
    unsigned Tag = llvm::dwarf::DW_TAG_structure_type;

    llvm::SmallVector<llvm::DIDescriptor, 5> EltTys;

    llvm::DIType FieldTy;

    QualType FType;
    uint64_t FieldSize, FieldOffset;
    unsigned FieldAlign;

    llvm::DIArray Elements;
    llvm::DIType EltTy;
    
    // Build up structure for the byref.  See BuildByRefType.
    FieldOffset = 0;
    FType = M->getContext().getPointerType(M->getContext().VoidTy);
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = M->getContext().getTypeSize(FType);
    FieldAlign = M->getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__isa", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = M->getContext().getPointerType(M->getContext().VoidTy);
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = M->getContext().getTypeSize(FType);
    FieldAlign = M->getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__forwarding", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = M->getContext().getFixedWidthIntType(32, true); // Int32Ty;
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = M->getContext().getTypeSize(FType);
    FieldAlign = M->getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__flags", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = M->getContext().getFixedWidthIntType(32, true); // Int32Ty;
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = M->getContext().getTypeSize(FType);
    FieldAlign = M->getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__size", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;
    
    bool HasCopyAndDispose = M->BlockRequiresCopying(Type);
    if (HasCopyAndDispose) {
      FType = M->getContext().getPointerType(M->getContext().VoidTy);
      FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
      FieldSize = M->getContext().getTypeSize(FType);
      FieldAlign = M->getContext().getTypeAlign(FType);
      FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                               "__copy_helper", DefUnit,
                                               0, FieldSize, FieldAlign,
                                               FieldOffset, 0, FieldTy);
      EltTys.push_back(FieldTy);
      FieldOffset += FieldSize;

      FType = M->getContext().getPointerType(M->getContext().VoidTy);
      FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
      FieldSize = M->getContext().getTypeSize(FType);
      FieldAlign = M->getContext().getTypeAlign(FType);
      FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                               "__destroy_helper", DefUnit,
                                               0, FieldSize, FieldAlign,
                                               FieldOffset, 0, FieldTy);
      EltTys.push_back(FieldTy);
      FieldOffset += FieldSize;
    }
    
    unsigned Align = M->getContext().getDeclAlignInBytes(Decl);
    if (Align > M->getContext().Target.getPointerAlign(0) / 8) {
      unsigned AlignedOffsetInBytes
        = llvm::RoundUpToAlignment(FieldOffset/8, Align);
      unsigned NumPaddingBytes
        = AlignedOffsetInBytes - FieldOffset/8;

      if (NumPaddingBytes > 0) {
        llvm::APInt pad(32, NumPaddingBytes);
        FType = M->getContext().getConstantArrayType(M->getContext().CharTy,
                                                     pad, ArrayType::Normal, 0);
        FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
        FieldSize = M->getContext().getTypeSize(FType);
        FieldAlign = M->getContext().getTypeAlign(FType);
        FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member,
                                                 Unit, "", DefUnit,
                                                 0, FieldSize, FieldAlign,
                                                 FieldOffset, 0, FieldTy);
        EltTys.push_back(FieldTy);
        FieldOffset += FieldSize;
      }
    }

    FType = Type;
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = M->getContext().getTypeSize(FType);
    FieldAlign = Align*8;
    std::string Name = Decl->getNameAsString();
    
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             Name, DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    Elements = DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

    unsigned Flags = llvm::DIType::FlagBlockByrefStruct;

    Ty = DebugFactory.CreateCompositeType(Tag, Unit, "",
                                          llvm::DICompileUnit(),
                                          0, FieldOffset, 0, 0, Flags,
                                          llvm::DIType(), Elements);
  }

  // Get location information.
  SourceManager &SM = M->getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  unsigned Line = 0;
  if (!PLoc.isInvalid())
    Line = PLoc.getLine();
  else
    Unit = llvm::DICompileUnit();


  // Create the descriptor for the variable.
  llvm::DIVariable D =
    DebugFactory.CreateVariable(Tag, RegionStack.back(),Decl->getNameAsString(),
                                Unit, Line, Ty);
  // Insert an llvm.dbg.declare into the current block.
  DebugFactory.InsertDeclare(Storage, D, Builder.GetInsertBlock());
}

void CGDebugInfo::EmitDeclareOfAutoVariable(const VarDecl *Decl,
                                            llvm::Value *Storage,
                                            CGBuilderTy &Builder) {
  EmitDeclare(Decl, llvm::dwarf::DW_TAG_auto_variable, Storage, Builder);
}

/// EmitDeclareOfArgVariable - Emit call to llvm.dbg.declare for an argument
/// variable declaration.
void CGDebugInfo::EmitDeclareOfArgVariable(const VarDecl *Decl, llvm::Value *AI,
                                           CGBuilderTy &Builder) {
  EmitDeclare(Decl, llvm::dwarf::DW_TAG_arg_variable, AI, Builder);
}



/// EmitGlobalVariable - Emit information about a global variable.
void CGDebugInfo::EmitGlobalVariable(llvm::GlobalVariable *Var,
                                     const VarDecl *Decl) {

  // Do not emit variable debug information while generating optimized code.
  // The llvm optimizer and code generator are not yet ready to support
  // optimized code debugging.
  const CompileOptions &CO = M->getCompileOpts();
  if (CO.OptimizationLevel)
    return;

  // Create global variable debug descriptor.
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(Decl->getLocation());
  SourceManager &SM = M->getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  unsigned LineNo = PLoc.isInvalid() ? 0 : PLoc.getLine();

  std::string Name = Decl->getNameAsString();

  QualType T = Decl->getType();
  if (T->isIncompleteArrayType()) {

    // CodeGen turns int[] into int[1] so we'll do the same here.
    llvm::APSInt ConstVal(32);

    ConstVal = 1;
    QualType ET = M->getContext().getAsArrayType(T)->getElementType();

    T = M->getContext().getConstantArrayType(ET, ConstVal,
                                           ArrayType::Normal, 0);
  }

  DebugFactory.CreateGlobalVariable(Unit, Name, Name, "", Unit, LineNo,
                                    getOrCreateType(T, Unit),
                                    Var->hasInternalLinkage(),
                                    true/*definition*/, Var);
}

/// EmitGlobalVariable - Emit information about an objective-c interface.
void CGDebugInfo::EmitGlobalVariable(llvm::GlobalVariable *Var,
                                     ObjCInterfaceDecl *Decl) {
  // Create global variable debug descriptor.
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(Decl->getLocation());
  SourceManager &SM = M->getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  unsigned LineNo = PLoc.isInvalid() ? 0 : PLoc.getLine();

  std::string Name = Decl->getNameAsString();

  QualType T = M->getContext().getObjCInterfaceType(Decl);
  if (T->isIncompleteArrayType()) {

    // CodeGen turns int[] into int[1] so we'll do the same here.
    llvm::APSInt ConstVal(32);

    ConstVal = 1;
    QualType ET = M->getContext().getAsArrayType(T)->getElementType();

    T = M->getContext().getConstantArrayType(ET, ConstVal,
                                           ArrayType::Normal, 0);
  }

  DebugFactory.CreateGlobalVariable(Unit, Name, Name, "", Unit, LineNo,
                                    getOrCreateType(T, Unit),
                                    Var->hasInternalLinkage(),
                                    true/*definition*/, Var);
}
