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
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
using namespace clang;
using namespace clang::CodeGen;

CGDebugInfo::CGDebugInfo(CodeGenModule &CGM)
  : CGM(CGM), DBuilder(CGM.getModule()),
    BlockLiteralGenericSet(false) {
  CreateCompileUnit();
}

CGDebugInfo::~CGDebugInfo() {
  assert(RegionStack.empty() && "Region stack mismatch, stack not empty!");
}

void CGDebugInfo::setLocation(SourceLocation Loc) {
  if (Loc.isValid())
    CurLoc = CGM.getContext().getSourceManager().getInstantiationLoc(Loc);
}

/// getContextDescriptor - Get context info for the decl.
llvm::DIDescriptor CGDebugInfo::getContextDescriptor(const Decl *Context) {
  if (!Context)
    return TheCU;

  llvm::DenseMap<const Decl *, llvm::WeakVH>::iterator
    I = RegionMap.find(Context);
  if (I != RegionMap.end())
    return llvm::DIDescriptor(dyn_cast_or_null<llvm::MDNode>(&*I->second));

  // Check namespace.
  if (const NamespaceDecl *NSDecl = dyn_cast<NamespaceDecl>(Context))
    return llvm::DIDescriptor(getOrCreateNameSpace(NSDecl));

  if (const RecordDecl *RDecl = dyn_cast<RecordDecl>(Context)) {
    if (!RDecl->isDependentType()) {
      llvm::DIType Ty = getOrCreateType(CGM.getContext().getTypeDeclType(RDecl),
                                        getOrCreateMainFile());
      return llvm::DIDescriptor(Ty);
    }
  }
  return TheCU;
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
  char *StrPtr = DebugInfoNames.Allocate<char>(NS.length());
  memcpy(StrPtr, NS.data(), NS.length());
  return llvm::StringRef(StrPtr, NS.length());
}

llvm::StringRef CGDebugInfo::getObjCMethodName(const ObjCMethodDecl *OMD) {
  llvm::SmallString<256> MethodName;
  llvm::raw_svector_ostream OS(MethodName);
  OS << (OMD->isInstanceMethod() ? '-' : '+') << '[';
  const DeclContext *DC = OMD->getDeclContext();
  if (const ObjCImplementationDecl *OID = 
      dyn_cast<const ObjCImplementationDecl>(DC)) {
     OS << OID->getName();
  } else if (const ObjCInterfaceDecl *OID = 
             dyn_cast<const ObjCInterfaceDecl>(DC)) {
      OS << OID->getName();
  } else if (const ObjCCategoryImplDecl *OCD = 
             dyn_cast<const ObjCCategoryImplDecl>(DC)){
      OS << ((NamedDecl *)OCD)->getIdentifier()->getNameStart() << '(' <<
          OCD->getIdentifier()->getNameStart() << ')';
  }
  OS << ' ' << OMD->getSelector().getAsString() << ']';

  char *StrPtr = DebugInfoNames.Allocate<char>(OS.tell());
  memcpy(StrPtr, MethodName.begin(), OS.tell());
  return llvm::StringRef(StrPtr, OS.tell());
}

/// getClassName - Get class name including template argument list.
llvm::StringRef 
CGDebugInfo::getClassName(RecordDecl *RD) {
  ClassTemplateSpecializationDecl *Spec
    = dyn_cast<ClassTemplateSpecializationDecl>(RD);
  if (!Spec)
    return RD->getName();

  const TemplateArgument *Args;
  unsigned NumArgs;
  std::string Buffer;
  if (TypeSourceInfo *TAW = Spec->getTypeAsWritten()) {
    const TemplateSpecializationType *TST =
      cast<TemplateSpecializationType>(TAW->getType());
    Args = TST->getArgs();
    NumArgs = TST->getNumArgs();
  } else {
    const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
    Args = TemplateArgs.data();
    NumArgs = TemplateArgs.size();
  }
  Buffer = RD->getIdentifier()->getNameStart();
  PrintingPolicy Policy(CGM.getLangOptions());
  Buffer += TemplateSpecializationType::PrintTemplateArgumentList(Args,
                                                                  NumArgs,
                                                                  Policy);

  // Copy this name on the side and use its reference.
  char *StrPtr = DebugInfoNames.Allocate<char>(Buffer.length());
  memcpy(StrPtr, Buffer.data(), Buffer.length());
  return llvm::StringRef(StrPtr, Buffer.length());
}

/// getOrCreateFile - Get the file debug info descriptor for the input location.
llvm::DIFile CGDebugInfo::getOrCreateFile(SourceLocation Loc) {
  if (!Loc.isValid())
    // If Location is not valid then use main input file.
    return DBuilder.CreateFile(TheCU.getFilename(), TheCU.getDirectory());

  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Loc);

  if (PLoc.isInvalid())
    // If the location is not valid then use main input file.
    return DBuilder.CreateFile(TheCU.getFilename(), TheCU.getDirectory());

  // Cache the results.
  const char *fname = PLoc.getFilename();
  llvm::DenseMap<const char *, llvm::WeakVH>::iterator it =
    DIFileCache.find(fname);

  if (it != DIFileCache.end()) {
    // Verify that the information still exists.
    if (&*it->second)
      return llvm::DIFile(cast<llvm::MDNode>(it->second));
  }

  llvm::DIFile F = DBuilder.CreateFile(PLoc.getFilename(), getCurrentDirname());

  DIFileCache[fname] = F;
  return F;

}

/// getOrCreateMainFile - Get the file info for main compile unit.
llvm::DIFile CGDebugInfo::getOrCreateMainFile() {
  return DBuilder.CreateFile(TheCU.getFilename(), TheCU.getDirectory());
}

/// getLineNumber - Get line number for the location. If location is invalid
/// then use current location.
unsigned CGDebugInfo::getLineNumber(SourceLocation Loc) {
  assert (CurLoc.isValid() && "Invalid current location!");
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Loc.isValid() ? Loc : CurLoc);
  return PLoc.isValid()? PLoc.getLine() : 0;
}

/// getColumnNumber - Get column number for the location. If location is 
/// invalid then use current location.
unsigned CGDebugInfo::getColumnNumber(SourceLocation Loc) {
  assert (CurLoc.isValid() && "Invalid current location!");
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Loc.isValid() ? Loc : CurLoc);
  return PLoc.isValid()? PLoc.getColumn() : 0;
}

llvm::StringRef CGDebugInfo::getCurrentDirname() {
  if (!CWDName.empty())
    return CWDName;
  char *CompDirnamePtr = NULL;
  llvm::sys::Path CWD = llvm::sys::Path::GetCurrentDirectory();
  CompDirnamePtr = DebugInfoNames.Allocate<char>(CWD.size());
  memcpy(CompDirnamePtr, CWD.c_str(), CWD.size());
  return CWDName = llvm::StringRef(CompDirnamePtr, CWD.size());
}

/// CreateCompileUnit - Create new compile unit.
void CGDebugInfo::CreateCompileUnit() {

  // Get absolute path name.
  SourceManager &SM = CGM.getContext().getSourceManager();
  std::string MainFileName = CGM.getCodeGenOpts().MainFileName;
  if (MainFileName.empty())
    MainFileName = "<unknown>";

  // The main file name provided via the "-main-file-name" option contains just
  // the file name itself with no path information. This file name may have had
  // a relative path, so we look into the actual file entry for the main
  // file to determine the real absolute path for the file.
  std::string MainFileDir;
  if (const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID())) {
    MainFileDir = MainFile->getDir()->getName();
    if (MainFileDir != ".")
      MainFileName = MainFileDir + "/" + MainFileName;
  }

  // Save filename string.
  char *FilenamePtr = DebugInfoNames.Allocate<char>(MainFileName.length());
  memcpy(FilenamePtr, MainFileName.c_str(), MainFileName.length());
  llvm::StringRef Filename(FilenamePtr, MainFileName.length());
  
  unsigned LangTag;
  const LangOptions &LO = CGM.getLangOptions();
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

  std::string Producer = getClangFullVersion();

  // Figure out which version of the ObjC runtime we have.
  unsigned RuntimeVers = 0;
  if (LO.ObjC1)
    RuntimeVers = LO.ObjCNonFragileABI ? 2 : 1;

  // Create new compile unit.
  DBuilder.CreateCompileUnit(
    LangTag, Filename, getCurrentDirname(),
    Producer,
    LO.Optimize, CGM.getCodeGenOpts().DwarfDebugFlags, RuntimeVers);
  // FIXME - Eliminate TheCU.
  TheCU = llvm::DICompileUnit(DBuilder.getCU());
}

/// CreateType - Get the Basic type from the cache or create a new
/// one if necessary.
llvm::DIType CGDebugInfo::CreateType(const BuiltinType *BT) {
  unsigned Encoding = 0;
  const char *BTName = NULL;
  switch (BT->getKind()) {
  default:
  case BuiltinType::Void:
    return llvm::DIType();
  case BuiltinType::ObjCClass:
    return DBuilder.CreateStructType(TheCU, "objc_class", 
                                     getOrCreateMainFile(), 0, 0, 0,
                                     llvm::DIDescriptor::FlagFwdDecl, 
                                     llvm::DIArray());
  case BuiltinType::ObjCId: {
    // typedef struct objc_class *Class;
    // typedef struct objc_object {
    //  Class isa;
    // } *id;

    llvm::DIType OCTy = 
      DBuilder.CreateStructType(TheCU, "objc_class", 
                                getOrCreateMainFile(), 0, 0, 0,
                                llvm::DIDescriptor::FlagFwdDecl, 
                                llvm::DIArray());
    unsigned Size = CGM.getContext().getTypeSize(CGM.getContext().VoidPtrTy);
    
    llvm::DIType ISATy = DBuilder.CreatePointerType(OCTy, Size);

    llvm::SmallVector<llvm::Value *, 16> EltTys;
    llvm::DIType FieldTy = 
      DBuilder.CreateMemberType("isa", getOrCreateMainFile(),
                                0,Size, 0, 0, 0, ISATy);
    EltTys.push_back(FieldTy);
    llvm::DIArray Elements =
      DBuilder.GetOrCreateArray(EltTys.data(), EltTys.size());
    
    return DBuilder.CreateStructType(TheCU, "objc_object", 
                                     getOrCreateMainFile(),
                                     0, 0, 0, 0, Elements);
  }
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

  switch (BT->getKind()) {
  case BuiltinType::Long:      BTName = "long int"; break;
  case BuiltinType::LongLong:  BTName = "long long int"; break;
  case BuiltinType::ULong:     BTName = "long unsigned int"; break;
  case BuiltinType::ULongLong: BTName = "long long unsigned int"; break;
  default:
    BTName = BT->getName(CGM.getContext().getLangOptions());
    break;
  }
  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(BT);
  uint64_t Align = CGM.getContext().getTypeAlign(BT);
  llvm::DIType DbgTy = 
    DBuilder.CreateBasicType(BTName, Size, Align, Encoding);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const ComplexType *Ty) {
  // Bit size, align and offset of the type.
  unsigned Encoding = llvm::dwarf::DW_ATE_complex_float;
  if (Ty->isComplexIntegerType())
    Encoding = llvm::dwarf::DW_ATE_lo_user;

  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);
  llvm::DIType DbgTy = 
    DBuilder.CreateBasicType("complex", Size, Align, Encoding);

  return DbgTy;
}

/// CreateCVRType - Get the qualified type from the cache or create
/// a new one if necessary.
llvm::DIType CGDebugInfo::CreateQualifiedType(QualType Ty, llvm::DIFile Unit) {
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

  llvm::DIType FromTy = getOrCreateType(Qc.apply(CGM.getContext(), T), Unit);

  // No need to fill in the Name, Line, Size, Alignment, Offset in case of
  // CVR derived types.
  llvm::DIType DbgTy = DBuilder.CreateQualifiedType(Tag, FromTy);
  
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const ObjCObjectPointerType *Ty,
                                     llvm::DIFile Unit) {
  llvm::DIType DbgTy =
    CreatePointerLikeType(llvm::dwarf::DW_TAG_pointer_type, Ty, 
                          Ty->getPointeeType(), Unit);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const PointerType *Ty,
                                     llvm::DIFile Unit) {
  return CreatePointerLikeType(llvm::dwarf::DW_TAG_pointer_type, Ty, 
                               Ty->getPointeeType(), Unit);
}

/// CreatePointeeType - Create PointTee type. If Pointee is a record
/// then emit record's fwd if debug info size reduction is enabled.
llvm::DIType CGDebugInfo::CreatePointeeType(QualType PointeeTy,
                                            llvm::DIFile Unit) {
  if (!CGM.getCodeGenOpts().LimitDebugInfo)
    return getOrCreateType(PointeeTy, Unit);
  
  if (const RecordType *RTy = dyn_cast<RecordType>(PointeeTy)) {
    RecordDecl *RD = RTy->getDecl();
    llvm::DIFile DefUnit = getOrCreateFile(RD->getLocation());
    unsigned Line = getLineNumber(RD->getLocation());
    llvm::DIDescriptor FDContext =
      getContextDescriptor(dyn_cast<Decl>(RD->getDeclContext()));

    if (RD->isStruct())
      return DBuilder.CreateStructType(FDContext, RD->getName(), DefUnit,
                                       Line, 0, 0, llvm::DIType::FlagFwdDecl,
                                       llvm::DIArray());
    else if (RD->isUnion())
      return DBuilder.CreateUnionType(FDContext, RD->getName(), DefUnit,
                                      Line, 0, 0, llvm::DIType::FlagFwdDecl,
                                      llvm::DIArray());
    else {
      assert(RD->isClass() && "Unknown RecordType!");
      return DBuilder.CreateClassType(FDContext, RD->getName(), DefUnit,
                                      Line, 0, 0, 0, llvm::DIType::FlagFwdDecl,
                                      llvm::DIType(), llvm::DIArray());
    }
  }
  return getOrCreateType(PointeeTy, Unit);

}

llvm::DIType CGDebugInfo::CreatePointerLikeType(unsigned Tag,
                                                const Type *Ty, 
                                                QualType PointeeTy,
                                                llvm::DIFile Unit) {

  if (Tag == llvm::dwarf::DW_TAG_reference_type)
    return DBuilder.CreateReferenceType(CreatePointeeType(PointeeTy, Unit));
                                    
  // Bit size, align and offset of the type.
  // Size is always the size of a pointer. We can't use getTypeSize here
  // because that does not return the correct value for references.
  uint64_t Size = 
    CGM.getContext().Target.getPointerWidth(PointeeTy.getAddressSpace());
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  return 
    DBuilder.CreatePointerType(CreatePointeeType(PointeeTy, Unit), Size, Align);
}

llvm::DIType CGDebugInfo::CreateType(const BlockPointerType *Ty,
                                     llvm::DIFile Unit) {
  if (BlockLiteralGenericSet)
    return BlockLiteralGeneric;

  llvm::SmallVector<llvm::Value *, 8> EltTys;
  llvm::DIType FieldTy;
  QualType FType;
  uint64_t FieldSize, FieldOffset;
  unsigned FieldAlign;
  llvm::DIArray Elements;
  llvm::DIType EltTy, DescTy;

  FieldOffset = 0;
  FType = CGM.getContext().UnsignedLongTy;
  EltTys.push_back(CreateMemberType(Unit, FType, "reserved", &FieldOffset));
  EltTys.push_back(CreateMemberType(Unit, FType, "Size", &FieldOffset));

  Elements = DBuilder.GetOrCreateArray(EltTys.data(), EltTys.size());
  EltTys.clear();

  unsigned Flags = llvm::DIDescriptor::FlagAppleBlock;
  unsigned LineNo = getLineNumber(CurLoc);

  EltTy = DBuilder.CreateStructType(Unit, "__block_descriptor",
                                    Unit, LineNo, FieldOffset, 0,
                                    Flags, Elements);

  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(Ty);

  DescTy = DBuilder.CreatePointerType(EltTy, Size);

  FieldOffset = 0;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  EltTys.push_back(CreateMemberType(Unit, FType, "__isa", &FieldOffset));
  FType = CGM.getContext().IntTy;
  EltTys.push_back(CreateMemberType(Unit, FType, "__flags", &FieldOffset));
  EltTys.push_back(CreateMemberType(Unit, FType, "__reserved", &FieldOffset));
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  EltTys.push_back(CreateMemberType(Unit, FType, "__FuncPtr", &FieldOffset));

  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  FieldTy = DescTy;
  FieldSize = CGM.getContext().getTypeSize(Ty);
  FieldAlign = CGM.getContext().getTypeAlign(Ty);
  FieldTy = DBuilder.CreateMemberType("__descriptor", Unit,
                                      LineNo, FieldSize, FieldAlign,
                                      FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  Elements = DBuilder.GetOrCreateArray(EltTys.data(), EltTys.size());

  EltTy = DBuilder.CreateStructType(Unit, "__block_literal_generic",
                                    Unit, LineNo, FieldOffset, 0,
                                    Flags, Elements);

  BlockLiteralGenericSet = true;
  BlockLiteralGeneric = DBuilder.CreatePointerType(EltTy, Size);
  return BlockLiteralGeneric;
}

llvm::DIType CGDebugInfo::CreateType(const TypedefType *Ty,
                                     llvm::DIFile Unit) {
  // Typedefs are derived from some other type.  If we have a typedef of a
  // typedef, make sure to emit the whole chain.
  llvm::DIType Src = getOrCreateType(Ty->getDecl()->getUnderlyingType(), Unit);
  if (!Src.Verify())
    return llvm::DIType();
  // We don't set size information, but do specify where the typedef was
  // declared.
  unsigned Line = getLineNumber(Ty->getDecl()->getLocation());
  llvm::DIType DbgTy = DBuilder.CreateTypedef(Src, Ty->getDecl()->getName(),
                                              Unit, Line);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const FunctionType *Ty,
                                     llvm::DIFile Unit) {
  llvm::SmallVector<llvm::Value *, 16> EltTys;

  // Add the result type at least.
  EltTys.push_back(getOrCreateType(Ty->getResultType(), Unit));

  // Set up remainder of arguments if there is a prototype.
  // FIXME: IF NOT, HOW IS THIS REPRESENTED?  llvm-gcc doesn't represent '...'!
  if (isa<FunctionNoProtoType>(Ty))
    EltTys.push_back(DBuilder.CreateUnspecifiedParameter());
  else if (const FunctionProtoType *FTP = dyn_cast<FunctionProtoType>(Ty)) {
    for (unsigned i = 0, e = FTP->getNumArgs(); i != e; ++i)
      EltTys.push_back(getOrCreateType(FTP->getArgType(i), Unit));
  }

  llvm::DIArray EltTypeArray =
    DBuilder.GetOrCreateArray(EltTys.data(), EltTys.size());

  llvm::DIType DbgTy = DBuilder.CreateSubroutineType(Unit, EltTypeArray);
  return DbgTy;
}

/// CollectRecordFields - A helper function to collect debug info for
/// record fields. This is used while creating debug info entry for a Record.
void CGDebugInfo::
CollectRecordFields(const RecordDecl *RD, llvm::DIFile Unit,
                    llvm::SmallVectorImpl<llvm::Value *> &EltTys) {
  unsigned FieldNo = 0;
  const ASTRecordLayout &RL = CGM.getContext().getASTRecordLayout(RD);
  for (RecordDecl::field_iterator I = RD->field_begin(),
                                  E = RD->field_end();
       I != E; ++I, ++FieldNo) {
    FieldDecl *Field = *I;
    llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);
    llvm::StringRef FieldName = Field->getName();

    // Ignore unnamed fields. Do not ignore unnamed records.
    if (FieldName.empty() && !isa<RecordType>(Field->getType()))
      continue;

    // Get the location for the field.
    llvm::DIFile FieldDefUnit = getOrCreateFile(Field->getLocation());
    unsigned FieldLine = getLineNumber(Field->getLocation());
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
    AccessSpecifier Access = I->getAccess();
    if (Access == clang::AS_private)
      Flags |= llvm::DIDescriptor::FlagPrivate;
    else if (Access == clang::AS_protected)
      Flags |= llvm::DIDescriptor::FlagProtected;

    FieldTy = DBuilder.CreateMemberType(FieldName, FieldDefUnit,
                                        FieldLine, FieldSize, FieldAlign,
                                        FieldOffset, Flags, FieldTy);
    EltTys.push_back(FieldTy);
  }
}

/// getOrCreateMethodType - CXXMethodDecl's type is a FunctionType. This
/// function type is not updated to include implicit "this" pointer. Use this
/// routine to get a method type which includes "this" pointer.
llvm::DIType
CGDebugInfo::getOrCreateMethodType(const CXXMethodDecl *Method,
                                   llvm::DIFile Unit) {
  llvm::DIType FnTy
    = getOrCreateType(QualType(Method->getType()->getAs<FunctionProtoType>(),
                               0),
                      Unit);
  
  // Add "this" pointer.

  llvm::DIArray Args = llvm::DICompositeType(FnTy).getTypeArray();
  assert (Args.getNumElements() && "Invalid number of arguments!");

  llvm::SmallVector<llvm::Value *, 16> Elts;

  // First element is always return type. For 'void' functions it is NULL.
  Elts.push_back(Args.getElement(0));

  if (!Method->isStatic())
  {
        // "this" pointer is always first argument.
        ASTContext &Context = CGM.getContext();
        QualType ThisPtr =
          Context.getPointerType(Context.getTagDeclType(Method->getParent()));
        llvm::DIType ThisPtrType =
          DBuilder.CreateArtificialType(getOrCreateType(ThisPtr, Unit));

        TypeCache[ThisPtr.getAsOpaquePtr()] = ThisPtrType;
        Elts.push_back(ThisPtrType);
    }

  // Copy rest of the arguments.
  for (unsigned i = 1, e = Args.getNumElements(); i != e; ++i)
    Elts.push_back(Args.getElement(i));

  llvm::DIArray EltTypeArray =
    DBuilder.GetOrCreateArray(Elts.data(), Elts.size());

  return DBuilder.CreateSubroutineType(Unit, EltTypeArray);
}

/// isFunctionLocalClass - Return true if CXXRecordDecl is defined 
/// inside a function.
static bool isFunctionLocalClass(const CXXRecordDecl *RD) {
  if (const CXXRecordDecl *NRD = 
      dyn_cast<CXXRecordDecl>(RD->getDeclContext()))
    return isFunctionLocalClass(NRD);
  else if (isa<FunctionDecl>(RD->getDeclContext()))
    return true;
  return false;
  
}
/// CreateCXXMemberFunction - A helper function to create a DISubprogram for
/// a single member function GlobalDecl.
llvm::DISubprogram
CGDebugInfo::CreateCXXMemberFunction(const CXXMethodDecl *Method,
                                     llvm::DIFile Unit,
                                     llvm::DIType RecordTy) {
  bool IsCtorOrDtor = 
    isa<CXXConstructorDecl>(Method) || isa<CXXDestructorDecl>(Method);
  
  llvm::StringRef MethodName = getFunctionName(Method);
  llvm::DIType MethodTy = getOrCreateMethodType(Method, Unit);
  
  // Since a single ctor/dtor corresponds to multiple functions, it doesn't
  // make sense to give a single ctor/dtor a linkage name.
  llvm::StringRef MethodLinkageName;
  if (!IsCtorOrDtor && !isFunctionLocalClass(Method->getParent()))
    MethodLinkageName = CGM.getMangledName(Method);

  // Get the location for the method.
  llvm::DIFile MethodDefUnit = getOrCreateFile(Method->getLocation());
  unsigned MethodLine = getLineNumber(Method->getLocation());

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
      VIndex = CGM.getVTables().getMethodVTableIndex(Method);
    ContainingType = RecordTy;
  }

  unsigned Flags = 0;
  if (Method->isImplicit())
    Flags |= llvm::DIDescriptor::FlagArtificial;
  AccessSpecifier Access = Method->getAccess();
  if (Access == clang::AS_private)
    Flags |= llvm::DIDescriptor::FlagPrivate;
  else if (Access == clang::AS_protected)
    Flags |= llvm::DIDescriptor::FlagProtected;
  if (const CXXConstructorDecl *CXXC = dyn_cast<CXXConstructorDecl>(Method)) {
    if (CXXC->isExplicit())
      Flags |= llvm::DIDescriptor::FlagExplicit;
  } else if (const CXXConversionDecl *CXXC = 
             dyn_cast<CXXConversionDecl>(Method)) {
    if (CXXC->isExplicit())
      Flags |= llvm::DIDescriptor::FlagExplicit;
  }
  if (Method->hasPrototype())
    Flags |= llvm::DIDescriptor::FlagPrototyped;
    
  llvm::DISubprogram SP =
    DBuilder.CreateMethod(RecordTy , MethodName, MethodLinkageName, 
                          MethodDefUnit, MethodLine,
                          MethodTy, /*isLocalToUnit=*/false, 
                          /* isDefinition=*/ false,
                          Virtuality, VIndex, ContainingType,
                          Flags, CGM.getLangOptions().Optimize);
  
  // Don't cache ctors or dtors since we have to emit multiple functions for
  // a single ctor or dtor.
  if (!IsCtorOrDtor && Method->isThisDeclarationADefinition())
    SPCache[Method] = llvm::WeakVH(SP);

  return SP;
}

/// CollectCXXMemberFunctions - A helper function to collect debug info for
/// C++ member functions.This is used while creating debug info entry for 
/// a Record.
void CGDebugInfo::
CollectCXXMemberFunctions(const CXXRecordDecl *RD, llvm::DIFile Unit,
                          llvm::SmallVectorImpl<llvm::Value *> &EltTys,
                          llvm::DIType RecordTy) {
  for(CXXRecordDecl::method_iterator I = RD->method_begin(),
        E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *Method = *I;
    
    if (Method->isImplicit() && !Method->isUsed())
      continue;

    EltTys.push_back(CreateCXXMemberFunction(Method, Unit, RecordTy));
  }
}                                 

/// CollectCXXFriends - A helper function to collect debug info for
/// C++ base classes. This is used while creating debug info entry for
/// a Record.
void CGDebugInfo::
CollectCXXFriends(const CXXRecordDecl *RD, llvm::DIFile Unit,
                llvm::SmallVectorImpl<llvm::Value *> &EltTys,
                llvm::DIType RecordTy) {

  for (CXXRecordDecl::friend_iterator BI =  RD->friend_begin(),
         BE = RD->friend_end(); BI != BE; ++BI) {
    if (TypeSourceInfo *TInfo = (*BI)->getFriendType())
      EltTys.push_back(DBuilder.CreateFriend(RecordTy, 
                                             getOrCreateType(TInfo->getType(), 
                                                             Unit)));
  }
}

/// CollectCXXBases - A helper function to collect debug info for
/// C++ base classes. This is used while creating debug info entry for 
/// a Record.
void CGDebugInfo::
CollectCXXBases(const CXXRecordDecl *RD, llvm::DIFile Unit,
                llvm::SmallVectorImpl<llvm::Value *> &EltTys,
                llvm::DIType RecordTy) {

  const ASTRecordLayout &RL = CGM.getContext().getASTRecordLayout(RD);
  for (CXXRecordDecl::base_class_const_iterator BI = RD->bases_begin(),
         BE = RD->bases_end(); BI != BE; ++BI) {
    unsigned BFlags = 0;
    uint64_t BaseOffset;
    
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(BI->getType()->getAs<RecordType>()->getDecl());
    
    if (BI->isVirtual()) {
      // virtual base offset offset is -ve. The code generator emits dwarf
      // expression where it expects +ve number.
      BaseOffset = 0 - CGM.getVTables().getVirtualBaseOffsetOffset(RD, Base);
      BFlags = llvm::DIDescriptor::FlagVirtual;
    } else
      BaseOffset = RL.getBaseClassOffsetInBits(Base);
    
    AccessSpecifier Access = BI->getAccessSpecifier();
    if (Access == clang::AS_private)
      BFlags |= llvm::DIDescriptor::FlagPrivate;
    else if (Access == clang::AS_protected)
      BFlags |= llvm::DIDescriptor::FlagProtected;
    
    llvm::DIType DTy = 
      DBuilder.CreateInheritance(RecordTy,                                     
                                 getOrCreateType(BI->getType(), Unit),
                                 BaseOffset, BFlags);
    EltTys.push_back(DTy);
  }
}

/// getOrCreateVTablePtrType - Return debug info descriptor for vtable.
llvm::DIType CGDebugInfo::getOrCreateVTablePtrType(llvm::DIFile Unit) {
  if (VTablePtrType.isValid())
    return VTablePtrType;

  ASTContext &Context = CGM.getContext();

  /* Function type */
  llvm::Value *STy = getOrCreateType(Context.IntTy, Unit);
  llvm::DIArray SElements = DBuilder.GetOrCreateArray(&STy, 1);
  llvm::DIType SubTy = DBuilder.CreateSubroutineType(Unit, SElements);
  unsigned Size = Context.getTypeSize(Context.VoidPtrTy);
  llvm::DIType vtbl_ptr_type = DBuilder.CreatePointerType(SubTy, Size, 0,
                                                          "__vtbl_ptr_type");
  VTablePtrType = DBuilder.CreatePointerType(vtbl_ptr_type, Size);
  return VTablePtrType;
}

/// getVTableName - Get vtable name for the given Class.
llvm::StringRef CGDebugInfo::getVTableName(const CXXRecordDecl *RD) {
  // Otherwise construct gdb compatible name name.
  std::string Name = "_vptr$" + RD->getNameAsString();

  // Copy this name on the side and use its reference.
  char *StrPtr = DebugInfoNames.Allocate<char>(Name.length());
  memcpy(StrPtr, Name.data(), Name.length());
  return llvm::StringRef(StrPtr, Name.length());
}


/// CollectVTableInfo - If the C++ class has vtable info then insert appropriate
/// debug info entry in EltTys vector.
void CGDebugInfo::
CollectVTableInfo(const CXXRecordDecl *RD, llvm::DIFile Unit,
                  llvm::SmallVectorImpl<llvm::Value *> &EltTys) {
  const ASTRecordLayout &RL = CGM.getContext().getASTRecordLayout(RD);

  // If there is a primary base then it will hold vtable info.
  if (RL.getPrimaryBase())
    return;

  // If this class is not dynamic then there is not any vtable info to collect.
  if (!RD->isDynamicClass())
    return;

  unsigned Size = CGM.getContext().getTypeSize(CGM.getContext().VoidPtrTy);
  llvm::DIType VPTR
    = DBuilder.CreateMemberType(getVTableName(RD), Unit,
                                0, Size, 0, 0, 0, 
                                getOrCreateVTablePtrType(Unit));
  EltTys.push_back(VPTR);
}

/// getOrCreateRecordType - Emit record type's standalone debug info. 
llvm::DIType CGDebugInfo::getOrCreateRecordType(QualType RTy, 
                                                SourceLocation Loc) {
  llvm::DIType T =  getOrCreateType(RTy, getOrCreateFile(Loc));
  DBuilder.RetainType(T);
  return T;
}

/// CreateType - get structure or union type.
llvm::DIType CGDebugInfo::CreateType(const RecordType *Ty) {
  RecordDecl *RD = Ty->getDecl();
  llvm::DIFile Unit = getOrCreateFile(RD->getLocation());

  // Get overall information about the record type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(RD->getLocation());
  unsigned Line = getLineNumber(RD->getLocation());

  // Records and classes and unions can all be recursive.  To handle them, we
  // first generate a debug descriptor for the struct as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the struct is recursive) and replace all
  // uses of the forward declaration with the final definition.
  llvm::DIDescriptor FDContext =
    getContextDescriptor(dyn_cast<Decl>(RD->getDeclContext()));

  // If this is just a forward declaration, construct an appropriately
  // marked node and just return it.
  if (!RD->getDefinition()) {
    llvm::DIType FwdDecl =
      DBuilder.CreateStructType(FDContext, RD->getName(),
                                DefUnit, Line, 0, 0,
                                llvm::DIDescriptor::FlagFwdDecl,
                                llvm::DIArray());

      return FwdDecl;
  }

  llvm::DIType FwdDecl = DBuilder.CreateTemporaryType(DefUnit);

  llvm::MDNode *MN = FwdDecl;
  llvm::TrackingVH<llvm::MDNode> FwdDeclNode = MN;
  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;
  // Push the struct on region stack.
  RegionStack.push_back(FwdDeclNode);
  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);

  // Convert all the elements.
  llvm::SmallVector<llvm::Value *, 16> EltTys;

  const CXXRecordDecl *CXXDecl = dyn_cast<CXXRecordDecl>(RD);
  if (CXXDecl) {
    CollectCXXBases(CXXDecl, Unit, EltTys, FwdDecl);
    CollectVTableInfo(CXXDecl, Unit, EltTys);
  }
  
  // Collect static variables with initializers.
  for (RecordDecl::decl_iterator I = RD->decls_begin(), E = RD->decls_end();
       I != E; ++I)
    if (const VarDecl *V = dyn_cast<VarDecl>(*I)) {
      if (const Expr *Init = V->getInit()) {
        Expr::EvalResult Result;
        if (Init->Evaluate(Result, CGM.getContext()) && Result.Val.isInt()) {
          llvm::ConstantInt *CI 
            = llvm::ConstantInt::get(CGM.getLLVMContext(), Result.Val.getInt());
          
          // Create the descriptor for static variable.
          llvm::DIFile VUnit = getOrCreateFile(V->getLocation());
          llvm::StringRef VName = V->getName();
          llvm::DIType VTy = getOrCreateType(V->getType(), VUnit);
          // Do not use DIGlobalVariable for enums.
          if (VTy.getTag() != llvm::dwarf::DW_TAG_enumeration_type) {
            DBuilder.CreateStaticVariable(FwdDecl, VName, VName, VUnit,
                                          getLineNumber(V->getLocation()),
                                          VTy, true, CI);
          }
        }
      }
    }

  CollectRecordFields(RD, Unit, EltTys);
  if (CXXDecl) {
    CollectCXXMemberFunctions(CXXDecl, Unit, EltTys, FwdDecl);
    CollectCXXFriends(CXXDecl, Unit, EltTys, FwdDecl);
  }

  RegionStack.pop_back();
  llvm::DenseMap<const Decl *, llvm::WeakVH>::iterator RI = 
    RegionMap.find(Ty->getDecl());
  if (RI != RegionMap.end())
    RegionMap.erase(RI);

  llvm::DIDescriptor RDContext =  
    getContextDescriptor(dyn_cast<Decl>(RD->getDeclContext()));
  llvm::StringRef RDName = RD->getName();
  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);
  llvm::DIArray Elements =
    DBuilder.GetOrCreateArray(EltTys.data(), EltTys.size());
  llvm::MDNode *RealDecl = NULL;

  if (RD->isStruct()) 
    RealDecl = DBuilder.CreateStructType(RDContext, RDName, DefUnit, Line,
                                         Size, Align, 0, Elements);
  else if (RD->isUnion())
    RealDecl = DBuilder.CreateUnionType(RDContext, RDName, DefUnit, Line,
                                         Size, Align, 0, Elements);
  else {
    assert(RD->isClass() && "Unknown RecordType!");
    RDName = getClassName(RD);
     // A class's primary base or the class itself contains the vtable.
    llvm::MDNode *ContainingType = NULL;
    const ASTRecordLayout &RL = CGM.getContext().getASTRecordLayout(RD);
    if (const CXXRecordDecl *PBase = RL.getPrimaryBase()) {
      // Seek non virtual primary base root.
      while (1) {
        const ASTRecordLayout &BRL = CGM.getContext().getASTRecordLayout(PBase);
        const CXXRecordDecl *PBT = BRL.getPrimaryBase();
        if (PBT && !BRL.isPrimaryBaseVirtual())
          PBase = PBT;
        else 
          break;
      }
      ContainingType = 
        getOrCreateType(QualType(PBase->getTypeForDecl(), 0), Unit);
    }
    else if (CXXDecl->isDynamicClass()) 
      ContainingType = FwdDecl;
   RealDecl = DBuilder.CreateClassType(RDContext, RDName, DefUnit, Line,
                                       Size, Align, 0, 0, llvm::DIType(),
                                       Elements, ContainingType);
  }

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  llvm::DIType(FwdDeclNode).replaceAllUsesWith(RealDecl);
  RegionMap[RD] = llvm::WeakVH(RealDecl);
  return llvm::DIType(RealDecl);
}

/// CreateType - get objective-c object type.
llvm::DIType CGDebugInfo::CreateType(const ObjCObjectType *Ty,
                                     llvm::DIFile Unit) {
  // Ignore protocols.
  return getOrCreateType(Ty->getBaseType(), Unit);
}

/// CreateType - get objective-c interface type.
llvm::DIType CGDebugInfo::CreateType(const ObjCInterfaceType *Ty,
                                     llvm::DIFile Unit) {
  ObjCInterfaceDecl *ID = Ty->getDecl();
  if (!ID)
    return llvm::DIType();

  // Get overall information about the record type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(ID->getLocation());
  unsigned Line = getLineNumber(ID->getLocation());
  unsigned RuntimeLang = TheCU.getLanguage();

  // If this is just a forward declaration, return a special forward-declaration
  // debug type.
  if (ID->isForwardDecl()) {
    llvm::DIType FwdDecl =
      DBuilder.CreateStructType(Unit, ID->getName(),
                                DefUnit, Line, 0, 0, 0,
                                llvm::DIArray(), RuntimeLang);
    return FwdDecl;
  }

  // To handle recursive interface, we
  // first generate a debug descriptor for the struct as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the struct is recursive) and replace all
  // uses of the forward declaration with the final definition.
  llvm::DIType FwdDecl = DBuilder.CreateTemporaryType(DefUnit);

  llvm::MDNode *MN = FwdDecl;
  llvm::TrackingVH<llvm::MDNode> FwdDeclNode = MN;
  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;
  // Push the struct on region stack.
  RegionStack.push_back(FwdDeclNode);
  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);

  // Convert all the elements.
  llvm::SmallVector<llvm::Value *, 16> EltTys;

  ObjCInterfaceDecl *SClass = ID->getSuperClass();
  if (SClass) {
    llvm::DIType SClassTy =
      getOrCreateType(CGM.getContext().getObjCInterfaceType(SClass), Unit);
    if (!SClassTy.isValid())
      return llvm::DIType();
    
    llvm::DIType InhTag =
      DBuilder.CreateInheritance(FwdDecl, SClassTy, 0, 0);
    EltTys.push_back(InhTag);
  }

  const ASTRecordLayout &RL = CGM.getContext().getASTObjCInterfaceLayout(ID);

  unsigned FieldNo = 0;
  for (ObjCIvarDecl *Field = ID->all_declared_ivar_begin(); Field;
       Field = Field->getNextIvar(), ++FieldNo) {
    llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);
    if (!FieldTy.isValid())
      return llvm::DIType();
    
    llvm::StringRef FieldName = Field->getName();

    // Ignore unnamed fields.
    if (FieldName.empty())
      continue;

    // Get the location for the field.
    llvm::DIFile FieldDefUnit = getOrCreateFile(Field->getLocation());
    unsigned FieldLine = getLineNumber(Field->getLocation());
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
      Flags = llvm::DIDescriptor::FlagProtected;
    else if (Field->getAccessControl() == ObjCIvarDecl::Private)
      Flags = llvm::DIDescriptor::FlagPrivate;

    FieldTy = DBuilder.CreateMemberType(FieldName, FieldDefUnit,
                                        FieldLine, FieldSize, FieldAlign,
                                        FieldOffset, Flags, FieldTy);
    EltTys.push_back(FieldTy);
  }

  llvm::DIArray Elements =
    DBuilder.GetOrCreateArray(EltTys.data(), EltTys.size());

  RegionStack.pop_back();
  llvm::DenseMap<const Decl *, llvm::WeakVH>::iterator RI = 
    RegionMap.find(Ty->getDecl());
  if (RI != RegionMap.end())
    RegionMap.erase(RI);

  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  llvm::DIType RealDecl =
    DBuilder.CreateStructType(Unit, ID->getName(), DefUnit,
                                  Line, Size, Align, 0,
                                  Elements, RuntimeLang);

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  llvm::DIType(FwdDeclNode).replaceAllUsesWith(RealDecl);
  RegionMap[ID] = llvm::WeakVH(RealDecl);

  return RealDecl;
}

llvm::DIType CGDebugInfo::CreateType(const TagType *Ty) {
  if (const RecordType *RT = dyn_cast<RecordType>(Ty))
    return CreateType(RT);
  else if (const EnumType *ET = dyn_cast<EnumType>(Ty))
    return CreateEnumType(ET->getDecl());

  return llvm::DIType();
}

llvm::DIType CGDebugInfo::CreateType(const VectorType *Ty,
                                     llvm::DIFile Unit) {
  llvm::DIType ElementTy = getOrCreateType(Ty->getElementType(), Unit);
  uint64_t NumElems = Ty->getNumElements();
  if (NumElems > 0)
    --NumElems;

  llvm::Value *Subscript = DBuilder.GetOrCreateSubrange(0, NumElems);
  llvm::DIArray SubscriptArray = DBuilder.GetOrCreateArray(&Subscript, 1);

  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  return
    DBuilder.CreateVectorType(Size, Align, ElementTy, SubscriptArray);
}

llvm::DIType CGDebugInfo::CreateType(const ArrayType *Ty,
                                     llvm::DIFile Unit) {
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
  llvm::SmallVector<llvm::Value *, 8> Subscripts;
  QualType EltTy(Ty, 0);
  if (Ty->isIncompleteArrayType())
    EltTy = Ty->getElementType();
  else {
    while ((Ty = dyn_cast<ArrayType>(EltTy))) {
      uint64_t Upper = 0;
      if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(Ty))
        if (CAT->getSize().getZExtValue())
          Upper = CAT->getSize().getZExtValue() - 1;
      // FIXME: Verify this is right for VLAs.
      Subscripts.push_back(DBuilder.GetOrCreateSubrange(0, Upper));
      EltTy = Ty->getElementType();
    }
  }

  llvm::DIArray SubscriptArray =
    DBuilder.GetOrCreateArray(Subscripts.data(), Subscripts.size());

  llvm::DIType DbgTy = 
    DBuilder.CreateArrayType(Size, Align, getOrCreateType(EltTy, Unit),
                             SubscriptArray);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const LValueReferenceType *Ty, 
                                     llvm::DIFile Unit) {
  return CreatePointerLikeType(llvm::dwarf::DW_TAG_reference_type, 
                               Ty, Ty->getPointeeType(), Unit);
}

llvm::DIType CGDebugInfo::CreateType(const MemberPointerType *Ty, 
                                     llvm::DIFile U) {
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
  llvm::Value *ElementTypes[2];
  
  // FIXME: This should probably be a function type instead.
  ElementTypes[0] =
    DBuilder.CreateMemberType("ptr", U, 0,
                              Info.first, Info.second, FieldOffset, 0,
                              PointerDiffDITy);
  FieldOffset += Info.first;
  
  ElementTypes[1] =
    DBuilder.CreateMemberType("ptr", U, 0,
                              Info.first, Info.second, FieldOffset, 0,
                              PointerDiffDITy);
  
  llvm::DIArray Elements = 
    DBuilder.GetOrCreateArray(&ElementTypes[0],
                              llvm::array_lengthof(ElementTypes));

  return DBuilder.CreateStructType(U, llvm::StringRef("test"), 
                                   U, 0, FieldOffset, 
                                   0, 0, Elements);
}

/// CreateEnumType - get enumeration type.
llvm::DIType CGDebugInfo::CreateEnumType(const EnumDecl *ED) {
  llvm::DIFile Unit = getOrCreateFile(ED->getLocation());
  llvm::SmallVector<llvm::Value *, 16> Enumerators;

  // Create DIEnumerator elements for each enumerator.
  for (EnumDecl::enumerator_iterator
         Enum = ED->enumerator_begin(), EnumEnd = ED->enumerator_end();
       Enum != EnumEnd; ++Enum) {
    Enumerators.push_back(
      DBuilder.CreateEnumerator(Enum->getName(),
                                Enum->getInitVal().getZExtValue()));
  }

  // Return a CompositeType for the enum itself.
  llvm::DIArray EltArray =
    DBuilder.GetOrCreateArray(Enumerators.data(), Enumerators.size());

  llvm::DIFile DefUnit = getOrCreateFile(ED->getLocation());
  unsigned Line = getLineNumber(ED->getLocation());
  uint64_t Size = 0;
  uint64_t Align = 0;
  if (!ED->getTypeForDecl()->isIncompleteType()) {
    Size = CGM.getContext().getTypeSize(ED->getTypeForDecl());
    Align = CGM.getContext().getTypeAlign(ED->getTypeForDecl());
  }
  llvm::DIDescriptor EnumContext = 
    getContextDescriptor(dyn_cast<Decl>(ED->getDeclContext()));
  llvm::DIType DbgTy = 
    DBuilder.CreateEnumerationType(EnumContext, ED->getName(), DefUnit, Line,
                                   Size, Align, EltArray);
  return DbgTy;
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
    case Type::Attributed:
      T = cast<AttributedType>(T)->getEquivalentType();
    case Type::Elaborated:
      T = cast<ElaboratedType>(T)->getNamedType();
      break;
    case Type::Paren:
      T = cast<ParenType>(T)->getInnerType();
      break;
    case Type::SubstTemplateTypeParm:
      T = cast<SubstTemplateTypeParmType>(T)->getReplacementType();
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
                                          llvm::DIFile Unit) {
  if (Ty.isNull())
    return llvm::DIType();

  // Unwrap the type as needed for debug information.
  Ty = UnwrapTypeForDebugInfo(Ty);
  
  // Check for existing entry.
  llvm::DenseMap<void *, llvm::WeakVH>::iterator it =
    TypeCache.find(Ty.getAsOpaquePtr());
  if (it != TypeCache.end()) {
    // Verify that the debug info still exists.
    if (&*it->second)
      return llvm::DIType(cast<llvm::MDNode>(it->second));
  }

  // Otherwise create the type.
  llvm::DIType Res = CreateTypeNode(Ty, Unit);

  // And update the type cache.
  TypeCache[Ty.getAsOpaquePtr()] = Res;  
  return Res;
}

/// CreateTypeNode - Create a new debug type node.
llvm::DIType CGDebugInfo::CreateTypeNode(QualType Ty,
                                         llvm::DIFile Unit) {
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
    return llvm::DIType();

  case Type::Vector:
    return CreateType(cast<VectorType>(Ty), Unit);
  case Type::ObjCObjectPointer:
    return CreateType(cast<ObjCObjectPointerType>(Ty), Unit);
  case Type::ObjCObject:
    return CreateType(cast<ObjCObjectType>(Ty), Unit);
  case Type::ObjCInterface:
    return CreateType(cast<ObjCInterfaceType>(Ty), Unit);
  case Type::Builtin: return CreateType(cast<BuiltinType>(Ty));
  case Type::Complex: return CreateType(cast<ComplexType>(Ty));
  case Type::Pointer: return CreateType(cast<PointerType>(Ty), Unit);
  case Type::BlockPointer:
    return CreateType(cast<BlockPointerType>(Ty), Unit);
  case Type::Typedef: return CreateType(cast<TypedefType>(Ty), Unit);
  case Type::Record:
  case Type::Enum:
    return CreateType(cast<TagType>(Ty));
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

  case Type::Attributed:
  case Type::TemplateSpecialization:
  case Type::Elaborated:
  case Type::Paren:
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
  CGM.getDiags().Report(DiagID)
    << Diag;
  return llvm::DIType();
}

/// CreateMemberType - Create new member and increase Offset by FType's size.
llvm::DIType CGDebugInfo::CreateMemberType(llvm::DIFile Unit, QualType FType,
                                           llvm::StringRef Name,
                                           uint64_t *Offset) {
  llvm::DIType FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  uint64_t FieldSize = CGM.getContext().getTypeSize(FType);
  unsigned FieldAlign = CGM.getContext().getTypeAlign(FType);
  llvm::DIType Ty = DBuilder.CreateMemberType(Name, Unit, 0,
                                              FieldSize, FieldAlign,
                                              *Offset, 0, FieldTy);
  *Offset += FieldSize;
  return Ty;
}

/// EmitFunctionStart - Constructs the debug code for entering a function -
/// "llvm.dbg.func.start.".
void CGDebugInfo::EmitFunctionStart(GlobalDecl GD, QualType FnType,
                                    llvm::Function *Fn,
                                    CGBuilderTy &Builder) {

  llvm::StringRef Name;
  llvm::StringRef LinkageName;

  FnBeginRegionCount.push_back(RegionStack.size());

  const Decl *D = GD.getDecl();
  unsigned Flags = 0;
  llvm::DIFile Unit = getOrCreateFile(CurLoc);
  llvm::DIDescriptor FDContext(Unit);
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // If there is a DISubprogram for  this function available then use it.
    llvm::DenseMap<const FunctionDecl *, llvm::WeakVH>::iterator
      FI = SPCache.find(FD);
    if (FI != SPCache.end()) {
      llvm::DIDescriptor SP(dyn_cast_or_null<llvm::MDNode>(&*FI->second));
      if (SP.isSubprogram() && llvm::DISubprogram(SP).isDefinition()) {
        llvm::MDNode *SPN = SP;
        RegionStack.push_back(SPN);
        RegionMap[D] = llvm::WeakVH(SP);
        return;
      }
    }
    Name = getFunctionName(FD);
    // Use mangled name as linkage name for c/c++ functions.
    LinkageName = CGM.getMangledName(GD);
    if (LinkageName == Name)
      LinkageName = llvm::StringRef();
    if (FD->hasPrototype())
      Flags |= llvm::DIDescriptor::FlagPrototyped;
    if (const NamespaceDecl *NSDecl =
        dyn_cast_or_null<NamespaceDecl>(FD->getDeclContext()))
      FDContext = getOrCreateNameSpace(NSDecl);
  } else if (const ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(D)) {
    Name = getObjCMethodName(OMD);
    Flags |= llvm::DIDescriptor::FlagPrototyped;
  } else {
    // Use llvm function name.
    Name = Fn->getName();
    Flags |= llvm::DIDescriptor::FlagPrototyped;
  }
  if (!Name.empty() && Name[0] == '\01')
    Name = Name.substr(1);

  // It is expected that CurLoc is set before using EmitFunctionStart.
  // Usually, CurLoc points to the left bracket location of compound
  // statement representing function body.
  unsigned LineNo = getLineNumber(CurLoc);
  if (D->isImplicit())
    Flags |= llvm::DIDescriptor::FlagArtificial;
  llvm::DISubprogram SP =
    DBuilder.CreateFunction(FDContext, Name, LinkageName, Unit,
                            LineNo, getOrCreateType(FnType, Unit),
                            Fn->hasInternalLinkage(), true/*definition*/,
                            Flags, CGM.getLangOptions().Optimize, Fn);

  // Push function on region stack.
  llvm::MDNode *SPN = SP;
  RegionStack.push_back(SPN);
  RegionMap[D] = llvm::WeakVH(SP);

  // Clear stack used to keep track of #line directives.
  LineDirectiveFiles.clear();
}


void CGDebugInfo::EmitStopPoint(CGBuilderTy &Builder) {
  if (CurLoc.isInvalid() || CurLoc.isMacroID()) return;

  // Don't bother if things are the same as last time.
  SourceManager &SM = CGM.getContext().getSourceManager();
  if (CurLoc == PrevLoc
       || (SM.getInstantiationLineNumber(CurLoc) ==
           SM.getInstantiationLineNumber(PrevLoc)
           && SM.isFromSameFile(CurLoc, PrevLoc)))
    // New Builder may not be in sync with CGDebugInfo.
    if (!Builder.getCurrentDebugLocation().isUnknown())
      return;

  // Update last state.
  PrevLoc = CurLoc;

  llvm::MDNode *Scope = RegionStack.back();
  Builder.SetCurrentDebugLocation(llvm::DebugLoc::get(getLineNumber(CurLoc),
                                                      getColumnNumber(CurLoc),
                                                      Scope));
}

/// UpdateLineDirectiveRegion - Update region stack only if #line directive
/// has introduced scope change.
void CGDebugInfo::UpdateLineDirectiveRegion(CGBuilderTy &Builder) {
  if (CurLoc.isInvalid() || CurLoc.isMacroID() ||
      PrevLoc.isInvalid() || PrevLoc.isMacroID())
    return;
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PCLoc = SM.getPresumedLoc(CurLoc);
  PresumedLoc PPLoc = SM.getPresumedLoc(PrevLoc);

  if (PCLoc.isInvalid() || PPLoc.isInvalid() ||
      !strcmp(PPLoc.getFilename(), PCLoc.getFilename()))
    return;

  // If #line directive stack is empty then we are entering a new scope.
  if (LineDirectiveFiles.empty()) {
    EmitRegionStart(Builder);
    LineDirectiveFiles.push_back(PCLoc.getFilename());
    return;
  }

  assert (RegionStack.size() >= LineDirectiveFiles.size()
          && "error handling  #line regions!");

  bool SeenThisFile = false;
  // Chek if current file is already seen earlier.
  for(std::vector<const char *>::iterator I = LineDirectiveFiles.begin(),
        E = LineDirectiveFiles.end(); I != E; ++I)
    if (!strcmp(PCLoc.getFilename(), *I)) {
      SeenThisFile = true;
      break;
    }

  // If #line for this file is seen earlier then pop out #line regions.
  if (SeenThisFile) {
    while (!LineDirectiveFiles.empty()) {
      const char *LastFile = LineDirectiveFiles.back();
      RegionStack.pop_back();
      LineDirectiveFiles.pop_back();
      if (!strcmp(PPLoc.getFilename(), LastFile))
        break;
    }
    return;
  } 

  // .. otherwise insert new #line region.
  EmitRegionStart(Builder);
  LineDirectiveFiles.push_back(PCLoc.getFilename());

  return;
}
/// EmitRegionStart- Constructs the debug code for entering a declarative
/// region - "llvm.dbg.region.start.".
void CGDebugInfo::EmitRegionStart(CGBuilderTy &Builder) {
  llvm::DIDescriptor D =
    DBuilder.CreateLexicalBlock(RegionStack.empty() ? 
                                llvm::DIDescriptor() : 
                                llvm::DIDescriptor(RegionStack.back()),
                                getOrCreateFile(CurLoc),
                                getLineNumber(CurLoc), 
                                getColumnNumber(CurLoc));
  llvm::MDNode *DN = D;
  RegionStack.push_back(DN);
}

/// EmitRegionEnd - Constructs the debug code for exiting a declarative
/// region - "llvm.dbg.region.end."
void CGDebugInfo::EmitRegionEnd(CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Provide an region stop point.
  EmitStopPoint(Builder);

  RegionStack.pop_back();
}

/// EmitFunctionEnd - Constructs the debug code for exiting a function.
void CGDebugInfo::EmitFunctionEnd(CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");
  unsigned RCount = FnBeginRegionCount.back();
  assert(RCount <= RegionStack.size() && "Region stack mismatch");

  // Pop all regions for this function.
  while (RegionStack.size() != RCount)
    EmitRegionEnd(Builder);
  FnBeginRegionCount.pop_back();
}

// EmitTypeForVarWithBlocksAttr - Build up structure info for the byref.  
// See BuildByRefType.
llvm::DIType CGDebugInfo::EmitTypeForVarWithBlocksAttr(const ValueDecl *VD,
                                                       uint64_t *XOffset) {

  llvm::SmallVector<llvm::Value *, 5> EltTys;
  QualType FType;
  uint64_t FieldSize, FieldOffset;
  unsigned FieldAlign;
  
  llvm::DIFile Unit = getOrCreateFile(VD->getLocation());
  QualType Type = VD->getType();  

  FieldOffset = 0;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  EltTys.push_back(CreateMemberType(Unit, FType, "__isa", &FieldOffset));
  EltTys.push_back(CreateMemberType(Unit, FType, "__forwarding", &FieldOffset));
  FType = CGM.getContext().IntTy;
  EltTys.push_back(CreateMemberType(Unit, FType, "__flags", &FieldOffset));
  EltTys.push_back(CreateMemberType(Unit, FType, "__size", &FieldOffset));

  bool HasCopyAndDispose = CGM.BlockRequiresCopying(Type);
  if (HasCopyAndDispose) {
    FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
    EltTys.push_back(CreateMemberType(Unit, FType, "__copy_helper",
                                      &FieldOffset));
    EltTys.push_back(CreateMemberType(Unit, FType, "__destroy_helper",
                                      &FieldOffset));
  }
  
  CharUnits Align = CGM.getContext().getDeclAlign(VD);
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
      EltTys.push_back(CreateMemberType(Unit, FType, "", &FieldOffset));
    }
  }
  
  FType = Type;
  llvm::DIType FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = Align.getQuantity()*8;

  *XOffset = FieldOffset;  
  FieldTy = DBuilder.CreateMemberType(VD->getName(), Unit,
                                      0, FieldSize, FieldAlign,
                                      FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);
  FieldOffset += FieldSize;
  
  llvm::DIArray Elements = 
    DBuilder.GetOrCreateArray(EltTys.data(), EltTys.size());
  
  unsigned Flags = llvm::DIDescriptor::FlagBlockByrefStruct;
  
  return DBuilder.CreateStructType(Unit, "", Unit, 0, FieldOffset, 0, Flags,
                                   Elements);
}

/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const VarDecl *VD, unsigned Tag,
                              llvm::Value *Storage, CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  llvm::DIFile Unit = getOrCreateFile(VD->getLocation());
  llvm::DIType Ty;
  uint64_t XOffset = 0;
  if (VD->hasAttr<BlocksAttr>())
    Ty = EmitTypeForVarWithBlocksAttr(VD, &XOffset);
  else 
    Ty = getOrCreateType(VD->getType(), Unit);

  // If there is not any debug info for type then do not emit debug info
  // for this variable.
  if (!Ty)
    return;

  // Get location information.
  unsigned Line = getLineNumber(VD->getLocation());
  unsigned Column = getColumnNumber(VD->getLocation());
  unsigned Flags = 0;
  if (VD->isImplicit())
    Flags |= llvm::DIDescriptor::FlagArtificial;
  llvm::MDNode *Scope = RegionStack.back();
    
  llvm::StringRef Name = VD->getName();
  if (!Name.empty()) {
    if (VD->hasAttr<BlocksAttr>()) {
      CharUnits offset = CharUnits::fromQuantity(32);
      llvm::SmallVector<llvm::Value *, 9> addr;
      const llvm::Type *Int64Ty = llvm::Type::getInt64Ty(CGM.getLLVMContext());
      addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpPlus));
      // offset of __forwarding field
      offset = 
        CharUnits::fromQuantity(CGM.getContext().Target.getPointerWidth(0)/8);
      addr.push_back(llvm::ConstantInt::get(Int64Ty, offset.getQuantity()));
      addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpDeref));
      addr.push_back(llvm::ConstantInt::get(Int64Ty, llvm::DIFactory::OpPlus));
      // offset of x field
      offset = CharUnits::fromQuantity(XOffset/8);
      addr.push_back(llvm::ConstantInt::get(Int64Ty, offset.getQuantity()));

      // Create the descriptor for the variable.
      llvm::DIVariable D =
        DBuilder.CreateComplexVariable(Tag, 
                                       llvm::DIDescriptor(RegionStack.back()),
                                       VD->getName(), Unit, Line, Ty,
                                       addr.data(), addr.size());
      
      // Insert an llvm.dbg.declare into the current block.
      llvm::Instruction *Call =
        DBuilder.InsertDeclare(Storage, D, Builder.GetInsertBlock());
      
      Call->setDebugLoc(llvm::DebugLoc::get(Line, Column, Scope));
      return;
    } 
      // Create the descriptor for the variable.
    llvm::DIVariable D =
      DBuilder.CreateLocalVariable(Tag, llvm::DIDescriptor(Scope), 
                                   Name, Unit, Line, Ty, 
                                   CGM.getLangOptions().Optimize, Flags);
    
    // Insert an llvm.dbg.declare into the current block.
    llvm::Instruction *Call =
      DBuilder.InsertDeclare(Storage, D, Builder.GetInsertBlock());
    
    Call->setDebugLoc(llvm::DebugLoc::get(Line, Column, Scope));
    return;
  }
  
  // If VD is an anonymous union then Storage represents value for
  // all union fields.
  if (const RecordType *RT = dyn_cast<RecordType>(VD->getType()))
    if (const RecordDecl *RD = dyn_cast<RecordDecl>(RT->getDecl()))
      if (RD->isUnion()) {
        for (RecordDecl::field_iterator I = RD->field_begin(),
               E = RD->field_end();
             I != E; ++I) {
          FieldDecl *Field = *I;
          llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);
          llvm::StringRef FieldName = Field->getName();
          
          // Ignore unnamed fields. Do not ignore unnamed records.
          if (FieldName.empty() && !isa<RecordType>(Field->getType()))
            continue;
          
          // Use VarDecl's Tag, Scope and Line number.
          llvm::DIVariable D =
            DBuilder.CreateLocalVariable(Tag, llvm::DIDescriptor(Scope),
                                         FieldName, Unit, Line, FieldTy, 
                                         CGM.getLangOptions().Optimize, Flags);
          
          // Insert an llvm.dbg.declare into the current block.
          llvm::Instruction *Call =
            DBuilder.InsertDeclare(Storage, D, Builder.GetInsertBlock());
          
          Call->setDebugLoc(llvm::DebugLoc::get(Line, Column, Scope));
        }
      }
}

/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const BlockDeclRefExpr *BDRE, unsigned Tag,
                              llvm::Value *Storage, CGBuilderTy &Builder,
                              CodeGenFunction *CGF) {
  const ValueDecl *VD = BDRE->getDecl();
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  if (Builder.GetInsertBlock() == 0)
    return;

  uint64_t XOffset = 0;
  llvm::DIFile Unit = getOrCreateFile(VD->getLocation());
  llvm::DIType Ty;
  if (VD->hasAttr<BlocksAttr>())
    Ty = EmitTypeForVarWithBlocksAttr(VD, &XOffset);
  else 
    Ty = getOrCreateType(VD->getType(), Unit);

  // Get location information.
  unsigned Line = getLineNumber(VD->getLocation());
  unsigned Column = getColumnNumber(VD->getLocation());

  CharUnits offset = CGF->BlockDecls[VD];
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
    DBuilder.CreateComplexVariable(Tag, llvm::DIDescriptor(RegionStack.back()),
                                   VD->getName(), Unit, Line, Ty,
                                   addr.data(), addr.size());
  // Insert an llvm.dbg.declare into the current block.
  llvm::Instruction *Call = 
    DBuilder.InsertDeclare(Storage, D, Builder.GetInsertBlock());
  
  llvm::MDNode *Scope = RegionStack.back();
  Call->setDebugLoc(llvm::DebugLoc::get(Line, Column, Scope));
}

void CGDebugInfo::EmitDeclareOfAutoVariable(const VarDecl *VD,
                                            llvm::Value *Storage,
                                            CGBuilderTy &Builder) {
  EmitDeclare(VD, llvm::dwarf::DW_TAG_auto_variable, Storage, Builder);
}

void CGDebugInfo::EmitDeclareOfBlockDeclRefVariable(
  const BlockDeclRefExpr *BDRE, llvm::Value *Storage, CGBuilderTy &Builder,
  CodeGenFunction *CGF) {
  EmitDeclare(BDRE, llvm::dwarf::DW_TAG_auto_variable, Storage, Builder, CGF);
}

/// EmitDeclareOfArgVariable - Emit call to llvm.dbg.declare for an argument
/// variable declaration.
void CGDebugInfo::EmitDeclareOfArgVariable(const VarDecl *VD, llvm::Value *AI,
                                           CGBuilderTy &Builder) {
  EmitDeclare(VD, llvm::dwarf::DW_TAG_arg_variable, AI, Builder);
}



/// EmitGlobalVariable - Emit information about a global variable.
void CGDebugInfo::EmitGlobalVariable(llvm::GlobalVariable *Var,
                                     const VarDecl *D) {
  
  // Create global variable debug descriptor.
  llvm::DIFile Unit = getOrCreateFile(D->getLocation());
  unsigned LineNo = getLineNumber(D->getLocation());

  QualType T = D->getType();
  if (T->isIncompleteArrayType()) {

    // CodeGen turns int[] into int[1] so we'll do the same here.
    llvm::APSInt ConstVal(32);

    ConstVal = 1;
    QualType ET = CGM.getContext().getAsArrayType(T)->getElementType();

    T = CGM.getContext().getConstantArrayType(ET, ConstVal,
                                           ArrayType::Normal, 0);
  }
  llvm::StringRef DeclName = D->getName();
  llvm::StringRef LinkageName;
  if (D->getDeclContext() && !isa<FunctionDecl>(D->getDeclContext()))
    LinkageName = Var->getName();
  if (LinkageName == DeclName)
    LinkageName = llvm::StringRef();
  llvm::DIDescriptor DContext = 
    getContextDescriptor(dyn_cast<Decl>(D->getDeclContext()));
  DBuilder.CreateStaticVariable(DContext, DeclName, LinkageName,
                                Unit, LineNo, getOrCreateType(T, Unit),
                                Var->hasInternalLinkage(), Var);
}

/// EmitGlobalVariable - Emit information about an objective-c interface.
void CGDebugInfo::EmitGlobalVariable(llvm::GlobalVariable *Var,
                                     ObjCInterfaceDecl *ID) {
  // Create global variable debug descriptor.
  llvm::DIFile Unit = getOrCreateFile(ID->getLocation());
  unsigned LineNo = getLineNumber(ID->getLocation());

  llvm::StringRef Name = ID->getName();

  QualType T = CGM.getContext().getObjCInterfaceType(ID);
  if (T->isIncompleteArrayType()) {

    // CodeGen turns int[] into int[1] so we'll do the same here.
    llvm::APSInt ConstVal(32);

    ConstVal = 1;
    QualType ET = CGM.getContext().getAsArrayType(T)->getElementType();

    T = CGM.getContext().getConstantArrayType(ET, ConstVal,
                                           ArrayType::Normal, 0);
  }

  DBuilder.CreateGlobalVariable(Name, Unit, LineNo,
                                getOrCreateType(T, Unit),
                                Var->hasInternalLinkage(), Var);
}

/// EmitGlobalVariable - Emit global variable's debug info.
void CGDebugInfo::EmitGlobalVariable(const ValueDecl *VD, 
                                     llvm::Constant *Init) {
  // Create the descriptor for the variable.
  llvm::DIFile Unit = getOrCreateFile(VD->getLocation());
  llvm::StringRef Name = VD->getName();
  llvm::DIType Ty = getOrCreateType(VD->getType(), Unit);
  if (const EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(VD)) {
    if (const EnumDecl *ED = dyn_cast<EnumDecl>(ECD->getDeclContext()))
      Ty = CreateEnumType(ED);
  }
  // Do not use DIGlobalVariable for enums.
  if (Ty.getTag() == llvm::dwarf::DW_TAG_enumeration_type)
    return;
  DBuilder.CreateStaticVariable(Unit, Name, Name, Unit,
                                getLineNumber(VD->getLocation()),
                                Ty, true, Init);
}

/// getOrCreateNamesSpace - Return namespace descriptor for the given
/// namespace decl.
llvm::DINameSpace 
CGDebugInfo::getOrCreateNameSpace(const NamespaceDecl *NSDecl) {
  llvm::DenseMap<const NamespaceDecl *, llvm::WeakVH>::iterator I = 
    NameSpaceCache.find(NSDecl);
  if (I != NameSpaceCache.end())
    return llvm::DINameSpace(cast<llvm::MDNode>(I->second));
  
  unsigned LineNo = getLineNumber(NSDecl->getLocation());
  llvm::DIFile FileD = getOrCreateFile(NSDecl->getLocation());
  llvm::DIDescriptor Context = 
    getContextDescriptor(dyn_cast<Decl>(NSDecl->getDeclContext()));
  llvm::DINameSpace NS =
    DBuilder.CreateNameSpace(Context, NSDecl->getName(), FileD, LineNo);
  NameSpaceCache[NSDecl] = llvm::WeakVH(NS);
  return NS;
}
