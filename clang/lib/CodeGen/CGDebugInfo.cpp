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
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/CodeGenOptions.h"
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

CGDebugInfo::CGDebugInfo(CodeGenModule &CGM)
  : CGM(CGM), isMainCompileUnitCreated(false), DebugFactory(CGM.getModule()),
    BlockLiteralGenericSet(false) {
}

CGDebugInfo::~CGDebugInfo() {
  assert(RegionStack.empty() && "Region stack mismatch, stack not empty!");
}

void CGDebugInfo::setLocation(SourceLocation Loc) {
  if (Loc.isValid())
    CurLoc = CGM.getContext().getSourceManager().getInstantiationLoc(Loc);
}

/// getContext - Get context info for the decl.
llvm::DIDescriptor CGDebugInfo::getContext(const VarDecl *Decl,
                                           llvm::DIDescriptor &CompileUnit) {
  if (Decl->isFileVarDecl())
    return CompileUnit;
  if (Decl->getDeclContext()->isFunctionOrMethod()) {
    // Find the last subprogram in region stack.
    for (unsigned RI = RegionStack.size(), RE = 0; RI != RE; --RI) {
      llvm::DIDescriptor R(RegionStack[RI - 1]);
      if (R.isSubprogram())
        return R;
    }
  }
  return CompileUnit;
}

/// getFunctionName - Get function name for the given FunctionDecl. If the
/// name is constructred on demand (e.g. C++ destructor) then the name
/// is stored on the side.
llvm::StringRef CGDebugInfo::getFunctionName(const FunctionDecl *FD) {
  assert (FD && "Invalid FunctionDecl!");
  IdentifierInfo *FII = FD->getIdentifier();
  if (FII)
    return FII->getName();

  // Otherwise construct human readable name for debug info.
  std::string NS = FD->getNameAsString();

  // Copy this name on the side and use its reference.
  char *StrPtr = FunctionNames.Allocate<char>(NS.length());
  memcpy(StrPtr, NS.data(), NS.length());
  return llvm::StringRef(StrPtr, NS.length());
}

/// getOrCreateCompileUnit - Get the compile unit from the cache or create a new
/// one if necessary. This returns null for invalid source locations.
llvm::DICompileUnit CGDebugInfo::getOrCreateCompileUnit(SourceLocation Loc) {
  // Get source file information.
  const char *FileName =  "<unknown>";
  SourceManager &SM = CGM.getContext().getSourceManager();
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
  AbsFileName.makeAbsolute();

  // See if thie compile unit is representing main source file. Each source
  // file has corresponding compile unit. There is only one main source
  // file at a time.
  bool isMain = false;
  const LangOptions &LO = CGM.getLangOptions();
  const CodeGenOptions &CGO = CGM.getCodeGenOpts();
  if (isMainCompileUnitCreated == false) {
    if (!CGO.MainFileName.empty()) {
      if (AbsFileName.getLast() == CGO.MainFileName)
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

  const char *Producer =
#ifdef CLANG_VENDOR
    CLANG_VENDOR
#endif
    "clang " CLANG_VERSION_STRING;

  // Figure out which version of the ObjC runtime we have.
  unsigned RuntimeVers = 0;
  if (LO.ObjC1)
    RuntimeVers = LO.ObjCNonFragileABI ? 2 : 1;

  // Create new compile unit.
  return Unit = DebugFactory.CreateCompileUnit(
    LangTag, AbsFileName.getLast(), AbsFileName.getDirname(), Producer, isMain,
    LO.Optimize, CGM.getCodeGenOpts().DwarfDebugFlags, RuntimeVers);
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
  case BuiltinType::LongDouble:
  case BuiltinType::Double:    Encoding = llvm::dwarf::DW_ATE_float; break;
  }
  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(BT);
  uint64_t Align = CGM.getContext().getTypeAlign(BT);
  uint64_t Offset = 0;

  llvm::DIType DbgTy = 
    DebugFactory.CreateBasicType(Unit,
                                 BT->getName(CGM.getContext().getLangOptions()),
                                 Unit, 0, Size, Align,
                                 Offset, /*flags*/ 0, Encoding);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const ComplexType *Ty,
                                     llvm::DICompileUnit Unit) {
  // Bit size, align and offset of the type.
  unsigned Encoding = llvm::dwarf::DW_ATE_complex_float;
  if (Ty->isComplexIntegerType())
    Encoding = llvm::dwarf::DW_ATE_lo_user;

  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);
  uint64_t Offset = 0;

  llvm::DIType DbgTy = 
    DebugFactory.CreateBasicType(Unit, "complex",
                                 Unit, 0, Size, Align,
                                 Offset, /*flags*/ 0, Encoding);
  return DbgTy;
}

/// CreateCVRType - Get the qualified type from the cache or create
/// a new one if necessary.
llvm::DIType CGDebugInfo::CreateQualifiedType(QualType Ty, llvm::DICompileUnit Unit) {
  QualifierCollector Qc;
  const Type *T = Qc.strip(Ty);

  // Ignore these qualifiers for now.
  Qc.removeObjCGCAttr();
  Qc.removeAddressSpace();

  // We will create one Derived type for one qualifier and recurse to handle any
  // additional ones.
  unsigned Tag;
  if (Qc.hasConst()) {
    Tag = llvm::dwarf::DW_TAG_const_type;
    Qc.removeConst();
  } else if (Qc.hasVolatile()) {
    Tag = llvm::dwarf::DW_TAG_volatile_type;
    Qc.removeVolatile();
  } else if (Qc.hasRestrict()) {
    Tag = llvm::dwarf::DW_TAG_restrict_type;
    Qc.removeRestrict();
  } else {
    assert(Qc.empty() && "Unknown type qualifier for debug info");
    return getOrCreateType(QualType(T, 0), Unit);
  }

  llvm::DIType FromTy = getOrCreateType(Qc.apply(T), Unit);

  // No need to fill in the Name, Line, Size, Alignment, Offset in case of
  // CVR derived types.
  llvm::DIType DbgTy =
    DebugFactory.CreateDerivedType(Tag, Unit, "", llvm::DICompileUnit(),
                                   0, 0, 0, 0, 0, FromTy);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const ObjCObjectPointerType *Ty,
                                     llvm::DICompileUnit Unit) {
  llvm::DIType DbgTy =
    CreatePointerLikeType(llvm::dwarf::DW_TAG_pointer_type, Ty, 
                          Ty->getPointeeType(), Unit);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const PointerType *Ty,
                                     llvm::DICompileUnit Unit) {
  return CreatePointerLikeType(llvm::dwarf::DW_TAG_pointer_type, Ty, 
                               Ty->getPointeeType(), Unit);
}

llvm::DIType CGDebugInfo::CreatePointerLikeType(unsigned Tag,
                                                const Type *Ty, 
                                                QualType PointeeTy,
                                                llvm::DICompileUnit Unit) {
  llvm::DIType EltTy = getOrCreateType(PointeeTy, Unit);

  // Bit size, align and offset of the type.
  
  // Size is always the size of a pointer. We can't use getTypeSize here
  // because that does not return the correct value for references.
  uint64_t Size = 
    CGM.getContext().Target.getPointerWidth(PointeeTy.getAddressSpace());
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  return
    DebugFactory.CreateDerivedType(Tag, Unit, "", llvm::DICompileUnit(),
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
  FType = CGM.getContext().UnsignedLongTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "reserved", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().UnsignedLongTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "Size", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  Elements = DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());
  EltTys.clear();

  unsigned Flags = llvm::DIType::FlagAppleBlock;

  EltTy = DebugFactory.CreateCompositeType(Tag, Unit, "__block_descriptor",
                                           DefUnit, 0, FieldOffset, 0, 0, Flags,
                                           llvm::DIType(), Elements);

  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  DescTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type,
                                          Unit, "", llvm::DICompileUnit(),
                                          0, Size, Align, 0, 0, EltTy);

  FieldOffset = 0;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__isa", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().IntTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__flags", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().IntTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__reserved", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__FuncPtr", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  FieldTy = DescTy;
  FieldSize = CGM.getContext().getTypeSize(Ty);
  FieldAlign = CGM.getContext().getTypeAlign(Ty);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__descriptor", DefUnit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  Elements = DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

  EltTy = DebugFactory.CreateCompositeType(Tag, Unit, "__block_literal_generic",
                                           DefUnit, 0, FieldOffset, 0, 0, Flags,
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
  SourceLocation DefLoc = Ty->getDecl()->getLocation();
  llvm::DICompileUnit DefUnit = getOrCreateCompileUnit(DefLoc);

  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(DefLoc);
  unsigned Line = PLoc.isInvalid() ? 0 : PLoc.getLine();

  llvm::DIType DbgTy = 
    DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_typedef, Unit,
                                   Ty->getDecl()->getName(),
                                   DefUnit, Line, 0, 0, 0, 0, Src);
  return DbgTy;
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

  llvm::DIType DbgTy =
    DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_subroutine_type,
                                     Unit, "", llvm::DICompileUnit(),
                                     0, 0, 0, 0, 0,
                                     llvm::DIType(), EltTypeArray);
  return DbgTy;
}

/// CollectRecordFields - A helper function to collect debug info for
/// record fields. This is used while creating debug info entry for a Record.
void CGDebugInfo::
CollectRecordFields(const RecordDecl *Decl,
                    llvm::DICompileUnit Unit,
                    llvm::SmallVectorImpl<llvm::DIDescriptor> &EltTys) {
  unsigned FieldNo = 0;
  SourceManager &SM = CGM.getContext().getSourceManager();
  const ASTRecordLayout &RL = CGM.getContext().getASTRecordLayout(Decl);
  for (RecordDecl::field_iterator I = Decl->field_begin(),
                                  E = Decl->field_end();
       I != E; ++I, ++FieldNo) {
    FieldDecl *Field = *I;
    llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);

    llvm::StringRef FieldName = Field->getName();

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
      FieldSize = CGM.getContext().getTypeSize(FType);
      Expr *BitWidth = Field->getBitWidth();
      if (BitWidth)
        FieldSize = BitWidth->EvaluateAsInt(CGM.getContext()).getZExtValue();

      FieldAlign =  CGM.getContext().getTypeAlign(FType);
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
}

/// CreateCXXMemberFunction - A helper function to create a DISubprogram for
/// a single member function GlobalDecl.
llvm::DISubprogram
CGDebugInfo::CreateCXXMemberFunction(const CXXMethodDecl *Method,
                                     llvm::DICompileUnit Unit,
                                     llvm::DICompositeType &RecordTy) {
  bool IsCtorOrDtor = 
    isa<CXXConstructorDecl>(Method) || isa<CXXDestructorDecl>(Method);
  
  llvm::StringRef MethodName = getFunctionName(Method);
  llvm::StringRef MethodLinkageName;
  llvm::DIType MethodTy = getOrCreateType(Method->getType(), Unit);
  
  // Since a single ctor/dtor corresponds to multiple functions, it doesn't
  // make sense to give a single ctor/dtor a linkage name.
  if (!IsCtorOrDtor)
    MethodLinkageName = CGM.getMangledName(Method);

  SourceManager &SM = CGM.getContext().getSourceManager();

  // Get the location for the method.
  SourceLocation MethodDefLoc = Method->getLocation();
  PresumedLoc PLoc = SM.getPresumedLoc(MethodDefLoc);
  llvm::DICompileUnit MethodDefUnit;
  unsigned MethodLine = 0;

  if (!PLoc.isInvalid()) {
    MethodDefUnit = getOrCreateCompileUnit(MethodDefLoc);
    MethodLine = PLoc.getLine();
  }

  // Collect virtual method info.
  llvm::DIType ContainingType;
  unsigned Virtuality = 0; 
  unsigned VIndex = 0;
  
  if (Method->isVirtual()) {
    if (Method->isPure())
      Virtuality = llvm::dwarf::DW_VIRTUALITY_pure_virtual;
    else
      Virtuality = llvm::dwarf::DW_VIRTUALITY_virtual;
    
    // It doesn't make sense to give a virtual destructor a vtable index,
    // since a single destructor has two entries in the vtable.
    if (!isa<CXXDestructorDecl>(Method))
      VIndex = CGM.getVtableInfo().getMethodVtableIndex(Method);
    ContainingType = RecordTy;
  }

  llvm::DISubprogram SP =
    DebugFactory.CreateSubprogram(RecordTy , MethodName, MethodName, 
                                  MethodLinkageName,
                                  MethodDefUnit, MethodLine,
                                  MethodTy, /*isLocalToUnit=*/false, 
                                  Method->isThisDeclarationADefinition(),
                                  Virtuality, VIndex, ContainingType);
  
  // Don't cache ctors or dtors since we have to emit multiple functions for
  // a single ctor or dtor.
  if (!IsCtorOrDtor && Method->isThisDeclarationADefinition())
    SPCache[Method] = llvm::WeakVH(SP.getNode());

  return SP;
}

/// CollectCXXMemberFunctions - A helper function to collect debug info for
/// C++ member functions.This is used while creating debug info entry for 
/// a Record.
void CGDebugInfo::
CollectCXXMemberFunctions(const CXXRecordDecl *Decl,
                          llvm::DICompileUnit Unit,
                          llvm::SmallVectorImpl<llvm::DIDescriptor> &EltTys,
                          llvm::DICompositeType &RecordTy) {
  for(CXXRecordDecl::method_iterator I = Decl->method_begin(),
        E = Decl->method_end(); I != E; ++I) {
    const CXXMethodDecl *Method = *I;
    
    if (Method->isImplicit())
      continue;

    EltTys.push_back(CreateCXXMemberFunction(Method, Unit, RecordTy));
  }
}                                 

/// CollectCXXBases - A helper function to collect debug info for
/// C++ base classes. This is used while creating debug info entry for 
/// a Record.
void CGDebugInfo::
CollectCXXBases(const CXXRecordDecl *Decl,
                llvm::DICompileUnit Unit,
                llvm::SmallVectorImpl<llvm::DIDescriptor> &EltTys,
                llvm::DICompositeType &RecordTy) {

    const ASTRecordLayout &RL = CGM.getContext().getASTRecordLayout(Decl);
    for (CXXRecordDecl::base_class_const_iterator BI = Decl->bases_begin(),
           BE = Decl->bases_end(); BI != BE; ++BI) {
      unsigned BFlags = 0;
      uint64_t BaseOffset;
      
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(BI->getType()->getAs<RecordType>()->getDecl());

      if (BI->isVirtual()) {
        BaseOffset = RL.getVBaseClassOffset(Base);
        BFlags = llvm::DIType::FlagVirtual;
      } else
        BaseOffset = RL.getBaseClassOffset(Base);

      AccessSpecifier Access = BI->getAccessSpecifier();
      if (Access == clang::AS_private)
        BFlags |= llvm::DIType::FlagPrivate;
      else if (Access == clang::AS_protected)
        BFlags |= llvm::DIType::FlagProtected;

      llvm::DIType DTy =
        DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_inheritance,
                                       RecordTy, llvm::StringRef(), 
                                       llvm::DICompileUnit(), 0, 0, 0,
                                       BaseOffset, BFlags,
                                       getOrCreateType(BI->getType(),
                                                       Unit));
      EltTys.push_back(DTy);
    }
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

  SourceManager &SM = CGM.getContext().getSourceManager();

  // Get overall information about the record type for the debug info.
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

  // A Decl->getName() is not unique. However, the debug info descriptors 
  // are uniqued. The debug info descriptor describing record's context is
  // necessary to keep two Decl's descriptor unique if their name match.
  // FIXME : Use RecordDecl's DeclContext's descriptor. As a temp. step
  // use type's name in FwdDecl.
  std::string STy = QualType(Ty, 0).getAsString();
  llvm::DICompositeType FwdDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, STy.c_str(),
                                     DefUnit, Line, 0, 0, 0, 0,
                                     llvm::DIType(), llvm::DIArray());

  // If this is just a forward declaration, return it.
  if (!Decl->getDefinition(CGM.getContext()))
    return FwdDecl;

  llvm::TrackingVH<llvm::MDNode> FwdDeclNode = FwdDecl.getNode();
  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl.getNode();

  // Convert all the elements.
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  CollectRecordFields(Decl, Unit, EltTys);
  if (CXXRecordDecl *CXXDecl = dyn_cast<CXXRecordDecl>(Decl)) {
    CollectCXXMemberFunctions(CXXDecl, Unit, EltTys, FwdDecl);
    CollectCXXBases(CXXDecl, Unit, EltTys, FwdDecl);
  }

  llvm::DIArray Elements =
    DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  llvm::DICompositeType RealDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, Decl->getName(),
                                     DefUnit, Line, Size, Align, 0, 0, 
                                     llvm::DIType(), Elements);

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  llvm::DIDerivedType(FwdDeclNode).replaceAllUsesWith(RealDecl);

  return RealDecl;
}

/// CreateType - get objective-c interface type.
llvm::DIType CGDebugInfo::CreateType(const ObjCInterfaceType *Ty,
                                     llvm::DICompileUnit Unit) {
  ObjCInterfaceDecl *Decl = Ty->getDecl();

  unsigned Tag = llvm::dwarf::DW_TAG_structure_type;
  SourceManager &SM = CGM.getContext().getSourceManager();

  // Get overall information about the record type for the debug info.
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
    DebugFactory.CreateCompositeType(Tag, Unit, Decl->getName(),
                                     DefUnit, Line, 0, 0, 0, 0,
                                     llvm::DIType(), llvm::DIArray(),
                                     RuntimeLang);

  // If this is just a forward declaration, return it.
  if (Decl->isForwardDecl())
    return FwdDecl;

  llvm::TrackingVH<llvm::MDNode> FwdDeclNode = FwdDecl.getNode();
  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl.getNode();

  // Convert all the elements.
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  ObjCInterfaceDecl *SClass = Decl->getSuperClass();
  if (SClass) {
    llvm::DIType SClassTy =
      getOrCreateType(CGM.getContext().getObjCInterfaceType(SClass), Unit);
    llvm::DIType InhTag =
      DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_inheritance,
                                     Unit, "", llvm::DICompileUnit(), 0, 0, 0,
                                     0 /* offset */, 0, SClassTy);
    EltTys.push_back(InhTag);
  }

  const ASTRecordLayout &RL = CGM.getContext().getASTObjCInterfaceLayout(Decl);

  unsigned FieldNo = 0;
  for (ObjCInterfaceDecl::ivar_iterator I = Decl->ivar_begin(),
         E = Decl->ivar_end();  I != E; ++I, ++FieldNo) {
    ObjCIvarDecl *Field = *I;
    llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);

    llvm::StringRef FieldName = Field->getName();

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
      FieldSize = CGM.getContext().getTypeSize(FType);
      Expr *BitWidth = Field->getBitWidth();
      if (BitWidth)
        FieldSize = BitWidth->EvaluateAsInt(CGM.getContext()).getZExtValue();

      FieldAlign =  CGM.getContext().getTypeAlign(FType);
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
  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  llvm::DICompositeType RealDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, Decl->getName(), DefUnit,
                                     Line, Size, Align, 0, 0, llvm::DIType(), 
                                     Elements, RuntimeLang);

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  llvm::DIDerivedType(FwdDeclNode).replaceAllUsesWith(RealDecl);

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
    Enumerators.push_back(DebugFactory.CreateEnumerator(Enum->getName(),
                                            Enum->getInitVal().getZExtValue()));
  }

  // Return a CompositeType for the enum itself.
  llvm::DIArray EltArray =
    DebugFactory.GetOrCreateArray(Enumerators.data(), Enumerators.size());

  SourceLocation DefLoc = Decl->getLocation();
  llvm::DICompileUnit DefUnit = getOrCreateCompileUnit(DefLoc);
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(DefLoc);
  unsigned Line = PLoc.isInvalid() ? 0 : PLoc.getLine();


  // Size and align of the type.
  uint64_t Size = 0;
  unsigned Align = 0;
  if (!Ty->isIncompleteType()) {
    Size = CGM.getContext().getTypeSize(Ty);
    Align = CGM.getContext().getTypeAlign(Ty);
  }

  llvm::DIType DbgTy = 
    DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_enumeration_type,
                                     Unit, Decl->getName(), DefUnit, Line,
                                     Size, Align, 0, 0,
                                     llvm::DIType(), EltArray);
  return DbgTy;
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
      CGM.getContext().getTypeAlign(CGM.getContext().getBaseElementType(VAT));
  } else if (Ty->isIncompleteArrayType()) {
    Size = 0;
    Align = CGM.getContext().getTypeAlign(Ty->getElementType());
  } else {
    // Size and align of the whole array, not the element type.
    Size = CGM.getContext().getTypeSize(Ty);
    Align = CGM.getContext().getTypeAlign(Ty);
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

  llvm::DIType DbgTy = 
    DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_array_type,
                                     Unit, "", llvm::DICompileUnit(),
                                     0, Size, Align, 0, 0,
                                     getOrCreateType(EltTy, Unit),
                                     SubscriptArray);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const LValueReferenceType *Ty, 
                                     llvm::DICompileUnit Unit) {
  return CreatePointerLikeType(llvm::dwarf::DW_TAG_reference_type, 
                               Ty, Ty->getPointeeType(), Unit);
}

llvm::DIType CGDebugInfo::CreateType(const MemberPointerType *Ty, 
                                     llvm::DICompileUnit U) {
  QualType PointerDiffTy = CGM.getContext().getPointerDiffType();
  llvm::DIType PointerDiffDITy = getOrCreateType(PointerDiffTy, U);
  
  if (!Ty->getPointeeType()->isFunctionType()) {
    // We have a data member pointer type.
    return PointerDiffDITy;
  }
  
  // We have a member function pointer type. Treat it as a struct with two
  // ptrdiff_t members.
  std::pair<uint64_t, unsigned> Info = CGM.getContext().getTypeInfo(Ty);

  uint64_t FieldOffset = 0;
  llvm::DIDescriptor ElementTypes[2];
  
  // FIXME: This should probably be a function type instead.
  ElementTypes[0] =
    DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, U,
                                   "ptr", llvm::DICompileUnit(), 0,
                                   Info.first, Info.second, FieldOffset, 0,
                                   PointerDiffDITy);
  FieldOffset += Info.first;
  
  ElementTypes[1] =
    DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, U,
                                   "ptr", llvm::DICompileUnit(), 0,
                                   Info.first, Info.second, FieldOffset, 0,
                                   PointerDiffDITy);
  
  llvm::DIArray Elements = 
    DebugFactory.GetOrCreateArray(&ElementTypes[0],
                                  llvm::array_lengthof(ElementTypes));

  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_structure_type, 
                                          U, llvm::StringRef("test"), 
                                          llvm::DICompileUnit(), 0, FieldOffset, 
                                          0, 0, 0, llvm::DIType(), Elements);
}

static QualType UnwrapTypeForDebugInfo(QualType T) {
  do {
    QualType LastT = T;
    switch (T->getTypeClass()) {
    default:
      return T;
    case Type::TemplateSpecialization:
      T = cast<TemplateSpecializationType>(T)->desugar();
      break;
    case Type::TypeOfExpr: {
      TypeOfExprType *Ty = cast<TypeOfExprType>(T);
      T = Ty->getUnderlyingExpr()->getType();
      break;
    }
    case Type::TypeOf:
      T = cast<TypeOfType>(T)->getUnderlyingType();
      break;
    case Type::Decltype:
      T = cast<DecltypeType>(T)->getUnderlyingType();
      break;
    case Type::QualifiedName:
      T = cast<QualifiedNameType>(T)->getNamedType();
      break;
    case Type::SubstTemplateTypeParm:
      T = cast<SubstTemplateTypeParmType>(T)->getReplacementType();
      break;
    case Type::Elaborated:
      T = cast<ElaboratedType>(T)->getUnderlyingType();
      break;
    }
    
    assert(T != LastT && "Type unwrapping failed to unwrap!");
    if (T == LastT)
      return T;
  } while (true);
  
  return T;
}

/// getOrCreateType - Get the type from the cache or create a new
/// one if necessary.
llvm::DIType CGDebugInfo::getOrCreateType(QualType Ty,
                                          llvm::DICompileUnit Unit) {
  if (Ty.isNull())
    return llvm::DIType();

  // Unwrap the type as needed for debug information.
  Ty = UnwrapTypeForDebugInfo(Ty);
  
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

  // And update the type cache.
  TypeCache[Ty.getAsOpaquePtr()] = Res.getNode();  
  return Res;
}

/// CreateTypeNode - Create a new debug type node.
llvm::DIType CGDebugInfo::CreateTypeNode(QualType Ty,
                                         llvm::DICompileUnit Unit) {
  // Handle qualifiers, which recursively handles what they refer to.
  if (Ty.hasLocalQualifiers())
    return CreateQualifiedType(Ty, Unit);

  const char *Diag = 0;
  
  // Work out details of type.
  switch (Ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    assert(false && "Dependent types cannot show up in debug information");

  // FIXME: Handle these.
  case Type::ExtVector:
  case Type::Vector:
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
  case Type::ConstantArray:
  case Type::VariableArray:
  case Type::IncompleteArray:
    return CreateType(cast<ArrayType>(Ty), Unit);

  case Type::LValueReference:
    return CreateType(cast<LValueReferenceType>(Ty), Unit);

  case Type::MemberPointer:
    return CreateType(cast<MemberPointerType>(Ty), Unit);

  case Type::TemplateSpecialization:
  case Type::Elaborated:
  case Type::QualifiedName:
  case Type::SubstTemplateTypeParm:
  case Type::TypeOfExpr:
  case Type::TypeOf:
  case Type::Decltype:
    llvm_unreachable("type should have been unwrapped!");
    return llvm::DIType();
      
  case Type::RValueReference:
    // FIXME: Implement!
    Diag = "rvalue references";
    break;
  }
  
  assert(Diag && "Fall through without a diagnostic?");
  unsigned DiagID = CGM.getDiags().getCustomDiagID(Diagnostic::Error,
                               "debug information for %0 is not yet supported");
  CGM.getDiags().Report(FullSourceLoc(), DiagID)
    << Diag;
  return llvm::DIType();
}

/// EmitFunctionStart - Constructs the debug code for entering a function -
/// "llvm.dbg.func.start.".
void CGDebugInfo::EmitFunctionStart(GlobalDecl GD, QualType FnType,
                                    llvm::Function *Fn,
                                    CGBuilderTy &Builder) {

  llvm::StringRef Name;
  llvm::StringRef LinkageName;

  const Decl *D = GD.getDecl();
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // If there is a DISubprogram for  this function available then use it.
    llvm::DenseMap<const FunctionDecl *, llvm::WeakVH>::iterator
      FI = SPCache.find(FD);
    if (FI != SPCache.end()) {
      llvm::DISubprogram SP(dyn_cast_or_null<llvm::MDNode>(FI->second));
      if (!SP.isNull() && SP.isSubprogram() && SP.isDefinition()) {
        RegionStack.push_back(SP.getNode());
        return;
      }
    }
    Name = getFunctionName(FD);
    if (!Name.empty() && Name[0] == '\01')
      Name = Name.substr(1);
    // Use mangled name as linkage name for c/c++ functions.
    LinkageName = CGM.getMangledName(GD);
  } else {
    // Use llvm function name as linkage name.
    Name = Fn->getName();
    LinkageName = Name;
    if (!Name.empty() && Name[0] == '\01')
      Name = Name.substr(1);
  }

  // It is expected that CurLoc is set before using EmitFunctionStart.
  // Usually, CurLoc points to the left bracket location of compound
  // statement representing function body.
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(CurLoc);
  SourceManager &SM = CGM.getContext().getSourceManager();
  unsigned LineNo = SM.getPresumedLoc(CurLoc).getLine();

  llvm::DISubprogram SP =
    DebugFactory.CreateSubprogram(Unit, Name, Name, LinkageName, Unit, LineNo,
                                  getOrCreateType(FnType, Unit),
                                  Fn->hasInternalLinkage(), true/*definition*/);

  // Push function on region stack.
  RegionStack.push_back(SP.getNode());
}


void CGDebugInfo::EmitStopPoint(llvm::Function *Fn, CGBuilderTy &Builder) {
  if (CurLoc.isInvalid() || CurLoc.isMacroID()) return;

  // Don't bother if things are the same as last time.
  SourceManager &SM = CGM.getContext().getSourceManager();
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

  llvm::DIDescriptor DR(RegionStack.back());
  llvm::DIScope DS = llvm::DIScope(DR.getNode());
  llvm::DILocation DO(NULL);
  llvm::DILocation DL = 
    DebugFactory.CreateLocation(PLoc.getLine(), PLoc.getColumn(),
                                DS, DO);
  Builder.SetCurrentDebugLocation(DL.getNode());
}

/// EmitRegionStart- Constructs the debug code for entering a declarative
/// region - "llvm.dbg.region.start.".
void CGDebugInfo::EmitRegionStart(llvm::Function *Fn, CGBuilderTy &Builder) {
  llvm::DIDescriptor D =
    DebugFactory.CreateLexicalBlock(RegionStack.empty() ? 
                                    llvm::DIDescriptor() : 
                                    llvm::DIDescriptor(RegionStack.back()));
  RegionStack.push_back(D.getNode());
}

/// EmitRegionEnd - Constructs the debug code for exiting a declarative
/// region - "llvm.dbg.region.end."
void CGDebugInfo::EmitRegionEnd(llvm::Function *Fn, CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Provide an region stop point.
  EmitStopPoint(Fn, Builder);

  RegionStack.pop_back();
}

/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const VarDecl *Decl, unsigned Tag,
                              llvm::Value *Storage, CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Do not emit variable debug information while generating optimized code.
  // The llvm optimizer and code generator are not yet ready to support
  // optimized code debugging.
  const CodeGenOptions &CGO = CGM.getCodeGenOpts();
  if (CGO.OptimizationLevel)
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
    FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__isa", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__forwarding", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = CGM.getContext().IntTy;
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__flags", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = CGM.getContext().IntTy;
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__size", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;
    
    bool HasCopyAndDispose = CGM.BlockRequiresCopying(Type);
    if (HasCopyAndDispose) {
      FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
      FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
      FieldSize = CGM.getContext().getTypeSize(FType);
      FieldAlign = CGM.getContext().getTypeAlign(FType);
      FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                               "__copy_helper", DefUnit,
                                               0, FieldSize, FieldAlign,
                                               FieldOffset, 0, FieldTy);
      EltTys.push_back(FieldTy);
      FieldOffset += FieldSize;

      FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
      FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
      FieldSize = CGM.getContext().getTypeSize(FType);
      FieldAlign = CGM.getContext().getTypeAlign(FType);
      FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                               "__destroy_helper", DefUnit,
                                               0, FieldSize, FieldAlign,
                                               FieldOffset, 0, FieldTy);
      EltTys.push_back(FieldTy);
      FieldOffset += FieldSize;
    }
    
    CharUnits Align = CGM.getContext().getDeclAlign(Decl);
    if (Align > CharUnits::fromQuantity(
          CGM.getContext().Target.getPointerAlign(0) / 8)) {
      unsigned AlignedOffsetInBytes
        = llvm::RoundUpToAlignment(FieldOffset/8, Align.getQuantity());
      unsigned NumPaddingBytes
        = AlignedOffsetInBytes - FieldOffset/8;

      if (NumPaddingBytes > 0) {
        llvm::APInt pad(32, NumPaddingBytes);
        FType = CGM.getContext().getConstantArrayType(CGM.getContext().CharTy,
                                                     pad, ArrayType::Normal, 0);
        FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
        FieldSize = CGM.getContext().getTypeSize(FType);
        FieldAlign = CGM.getContext().getTypeAlign(FType);
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
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = Align.getQuantity()*8;
    
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             Decl->getName(), DefUnit,
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
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  unsigned Line = 0;
  unsigned Column = 0;
  if (!PLoc.isInvalid()) {
    Line = PLoc.getLine();
    Column = PLoc.getColumn();
  } else {
    Unit = llvm::DICompileUnit();
  }

  // Create the descriptor for the variable.
  llvm::DIVariable D =
    DebugFactory.CreateVariable(Tag, llvm::DIDescriptor(RegionStack.back()),
                                Decl->getName(),
                                Unit, Line, Ty);
  // Insert an llvm.dbg.declare into the current block.
  llvm::Instruction *Call =
    DebugFactory.InsertDeclare(Storage, D, Builder.GetInsertBlock());

  llvm::DIScope DS(RegionStack.back());
  llvm::DILocation DO(NULL);
  llvm::DILocation DL = DebugFactory.CreateLocation(Line, Column, DS, DO);
  
  Call->setMetadata("dbg", DL.getNode());
}

/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const BlockDeclRefExpr *BDRE, unsigned Tag,
                              llvm::Value *Storage, CGBuilderTy &Builder,
                              CodeGenFunction *CGF) {
  const ValueDecl *Decl = BDRE->getDecl();
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Do not emit variable debug information while generating optimized code.
  // The llvm optimizer and code generator are not yet ready to support
  // optimized code debugging.
  const CodeGenOptions &CGO = CGM.getCodeGenOpts();
  if (CGO.OptimizationLevel || Builder.GetInsertBlock() == 0)
    return;

  uint64_t XOffset = 0;
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
    FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__isa", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__forwarding", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = CGM.getContext().IntTy;
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__flags", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;

    FType = CGM.getContext().IntTy;
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__size", DefUnit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;
    
    bool HasCopyAndDispose = CGM.BlockRequiresCopying(Type);
    if (HasCopyAndDispose) {
      FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
      FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
      FieldSize = CGM.getContext().getTypeSize(FType);
      FieldAlign = CGM.getContext().getTypeAlign(FType);
      FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                               "__copy_helper", DefUnit,
                                               0, FieldSize, FieldAlign,
                                               FieldOffset, 0, FieldTy);
      EltTys.push_back(FieldTy);
      FieldOffset += FieldSize;

      FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
      FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
      FieldSize = CGM.getContext().getTypeSize(FType);
      FieldAlign = CGM.getContext().getTypeAlign(FType);
      FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                               "__destroy_helper", DefUnit,
                                               0, FieldSize, FieldAlign,
                                               FieldOffset, 0, FieldTy);
      EltTys.push_back(FieldTy);
      FieldOffset += FieldSize;
    }
    
    CharUnits Align = CGM.getContext().getDeclAlign(Decl);
    if (Align > CharUnits::fromQuantity(
          CGM.getContext().Target.getPointerAlign(0) / 8)) {
      unsigned AlignedOffsetInBytes
        = llvm::RoundUpToAlignment(FieldOffset/8, Align.getQuantity());
      unsigned NumPaddingBytes
        = AlignedOffsetInBytes - FieldOffset/8;

      if (NumPaddingBytes > 0) {
        llvm::APInt pad(32, NumPaddingBytes);
        FType = CGM.getContext().getConstantArrayType(CGM.getContext().CharTy,
                                                     pad, ArrayType::Normal, 0);
        FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
        FieldSize = CGM.getContext().getTypeSize(FType);
        FieldAlign = CGM.getContext().getTypeAlign(FType);
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
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = Align.getQuantity()*8;
    
    XOffset = FieldOffset;
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             Decl->getName(), DefUnit,
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
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  unsigned Line = 0;
  if (!PLoc.isInvalid())
    Line = PLoc.getLine();
  else
    Unit = llvm::DICompileUnit();

  CharUnits offset = CGF->BlockDecls[Decl];
  llvm::SmallVector<llvm::Value *, 9> addr;
  const llvm::Type *Int64Ty = llvm::Type::getInt64Ty(CGM.getLLVMContext());
  addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpDeref));
  addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpPlus));
  addr.push_back(llvm::ConstantInt::get(Int64Ty, offset.getQuantity()));
  if (BDRE->isByRef()) {
    addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpDeref));
    addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpPlus));
    // offset of __forwarding field
    offset = CharUnits::fromQuantity(CGF->LLVMPointerWidth/8);
    addr.push_back(llvm::ConstantInt::get(Int64Ty, offset.getQuantity()));
    addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpDeref));
    addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpPlus));
    // offset of x field
    offset = CharUnits::fromQuantity(XOffset/8);
    addr.push_back(llvm::ConstantInt::get(Int64Ty, offset.getQuantity()));
  }

  // Create the descriptor for the variable.
  llvm::DIVariable D =
    DebugFactory.CreateComplexVariable(Tag,
                                       llvm::DIDescriptor(RegionStack.back()),
                                       Decl->getName(), Unit, Line, Ty,
                                       addr);
  // Insert an llvm.dbg.declare into the current block.
  llvm::Instruction *Call = 
    DebugFactory.InsertDeclare(Storage, D, Builder.GetInsertBlock());

  llvm::DIScope DS(RegionStack.back());
  llvm::DILocation DO(NULL);
  llvm::DILocation DL = 
    DebugFactory.CreateLocation(Line, PLoc.getColumn(), DS, DO);
  
  Call->setMetadata("dbg", DL.getNode());
}

void CGDebugInfo::EmitDeclareOfAutoVariable(const VarDecl *Decl,
                                            llvm::Value *Storage,
                                            CGBuilderTy &Builder) {
  EmitDeclare(Decl, llvm::dwarf::DW_TAG_auto_variable, Storage, Builder);
}

void CGDebugInfo::EmitDeclareOfBlockDeclRefVariable(
  const BlockDeclRefExpr *BDRE, llvm::Value *Storage, CGBuilderTy &Builder,
  CodeGenFunction *CGF) {
  EmitDeclare(BDRE, llvm::dwarf::DW_TAG_auto_variable, Storage, Builder, CGF);
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

  // Create global variable debug descriptor.
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(Decl->getLocation());
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  unsigned LineNo = PLoc.isInvalid() ? 0 : PLoc.getLine();

  QualType T = Decl->getType();
  if (T->isIncompleteArrayType()) {

    // CodeGen turns int[] into int[1] so we'll do the same here.
    llvm::APSInt ConstVal(32);

    ConstVal = 1;
    QualType ET = CGM.getContext().getAsArrayType(T)->getElementType();

    T = CGM.getContext().getConstantArrayType(ET, ConstVal,
                                           ArrayType::Normal, 0);
  }
  llvm::StringRef DeclName = Decl->getName();
  DebugFactory.CreateGlobalVariable(getContext(Decl, Unit), DeclName, DeclName,
                                    llvm::StringRef(), Unit, LineNo,
                                    getOrCreateType(T, Unit),
                                    Var->hasInternalLinkage(),
                                    true/*definition*/, Var);
}

/// EmitGlobalVariable - Emit information about an objective-c interface.
void CGDebugInfo::EmitGlobalVariable(llvm::GlobalVariable *Var,
                                     ObjCInterfaceDecl *Decl) {
  // Create global variable debug descriptor.
  llvm::DICompileUnit Unit = getOrCreateCompileUnit(Decl->getLocation());
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Decl->getLocation());
  unsigned LineNo = PLoc.isInvalid() ? 0 : PLoc.getLine();

  llvm::StringRef Name = Decl->getName();

  QualType T = CGM.getContext().getObjCInterfaceType(Decl);
  if (T->isIncompleteArrayType()) {

    // CodeGen turns int[] into int[1] so we'll do the same here.
    llvm::APSInt ConstVal(32);

    ConstVal = 1;
    QualType ET = CGM.getContext().getAsArrayType(T)->getElementType();

    T = CGM.getContext().getConstantArrayType(ET, ConstVal,
                                           ArrayType::Normal, 0);
  }

  DebugFactory.CreateGlobalVariable(Unit, Name, Name, Name, Unit, LineNo,
                                    getOrCreateType(T, Unit),
                                    Var->hasInternalLinkage(),
                                    true/*definition*/, Var);
}
