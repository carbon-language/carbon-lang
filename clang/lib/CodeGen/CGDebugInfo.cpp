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
  : CGM(CGM), DebugFactory(CGM.getModule()),
    FwdDeclCount(0), BlockLiteralGenericSet(false) {
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
llvm::DIDescriptor CGDebugInfo::getContextDescriptor(const Decl *Context,
                                              llvm::DIDescriptor &CompileUnit) {
  if (!Context)
    return CompileUnit;

  llvm::DenseMap<const Decl *, llvm::WeakVH>::iterator
    I = RegionMap.find(Context);
  if (I != RegionMap.end())
    return llvm::DIDescriptor(dyn_cast_or_null<llvm::MDNode>(I->second));

  // Check namespace.
  if (const NamespaceDecl *NSDecl = dyn_cast<NamespaceDecl>(Context))
    return llvm::DIDescriptor(getOrCreateNameSpace(NSDecl, CompileUnit));
  
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
  char *StrPtr = DebugInfoNames.Allocate<char>(NS.length());
  memcpy(StrPtr, NS.data(), NS.length());
  return llvm::StringRef(StrPtr, NS.length());
}

/// getOrCreateFile - Get the file debug info descriptor for the input location.
llvm::DIFile CGDebugInfo::getOrCreateFile(SourceLocation Loc) {
  if (!Loc.isValid())
    // If Location is not valid then use main input file.
    return DebugFactory.CreateFile(TheCU.getFilename(), TheCU.getDirectory(),
                                   TheCU);
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Loc);

  // Cache the results.
  const char *fname = PLoc.getFilename();
  llvm::DenseMap<const char *, llvm::WeakVH>::iterator it =
    DIFileCache.find(fname);

  if (it != DIFileCache.end()) {
    // Verify that the information still exists.
    if (&*it->second)
      return llvm::DIFile(cast<llvm::MDNode>(it->second));
  }

  // FIXME: We shouldn't even need to call 'makeAbsolute()' in the cases
  // where we can consult the FileEntry.
  llvm::sys::Path AbsFileName(PLoc.getFilename());
  AbsFileName.makeAbsolute();

  llvm::DIFile F = DebugFactory.CreateFile(AbsFileName.getLast(),
                                           AbsFileName.getDirname(), TheCU);

  DIFileCache[fname] = F.getNode();
  return F;

}
/// CreateCompileUnit - Create new compile unit.
void CGDebugInfo::CreateCompileUnit() {

  // Get absolute path name.
  SourceManager &SM = CGM.getContext().getSourceManager();
  std::string MainFileName = CGM.getCodeGenOpts().MainFileName;
  if (MainFileName.empty())
    MainFileName = "<unknown>";

  llvm::sys::Path AbsFileName(MainFileName);
  AbsFileName.makeAbsolute();

  // The main file name provided via the "-main-file-name" option contains just
  // the file name itself with no path information. This file name may have had
  // a relative path, so we look into the actual file entry for the main
  // file to determine the real absolute path for the file.
  std::string MainFileDir;
  if (const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID()))
    MainFileDir = MainFile->getDir()->getName();
  else
    MainFileDir = AbsFileName.getDirname();

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
  TheCU = DebugFactory.CreateCompileUnit(
    LangTag, AbsFileName.getLast(), MainFileDir, Producer, true,
    LO.Optimize, CGM.getCodeGenOpts().DwarfDebugFlags, RuntimeVers);
}

/// CreateType - Get the Basic type from the cache or create a new
/// one if necessary.
llvm::DIType CGDebugInfo::CreateType(const BuiltinType *BT,
                                     llvm::DIFile Unit) {
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
                                     llvm::DIFile Unit) {
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

  llvm::DIType FromTy = getOrCreateType(Qc.apply(T), Unit);

  // No need to fill in the Name, Line, Size, Alignment, Offset in case of
  // CVR derived types.
  llvm::DIType DbgTy =
    DebugFactory.CreateDerivedType(Tag, Unit, "", Unit,
                                   0, 0, 0, 0, 0, FromTy);
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

llvm::DIType CGDebugInfo::CreatePointerLikeType(unsigned Tag,
                                                const Type *Ty, 
                                                QualType PointeeTy,
                                                llvm::DIFile Unit) {
  llvm::DIType EltTy = getOrCreateType(PointeeTy, Unit);

  // Bit size, align and offset of the type.
  
  // Size is always the size of a pointer. We can't use getTypeSize here
  // because that does not return the correct value for references.
  uint64_t Size = 
    CGM.getContext().Target.getPointerWidth(PointeeTy.getAddressSpace());
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  return
    DebugFactory.CreateDerivedType(Tag, Unit, "", Unit,
                                   0, Size, Align, 0, 0, EltTy);
  
}

llvm::DIType CGDebugInfo::CreateType(const BlockPointerType *Ty,
                                     llvm::DIFile Unit) {
  if (BlockLiteralGenericSet)
    return BlockLiteralGeneric;

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
                                           "reserved", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().UnsignedLongTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "Size", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  Elements = DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());
  EltTys.clear();

  unsigned Flags = llvm::DIType::FlagAppleBlock;

  EltTy = DebugFactory.CreateCompositeType(Tag, Unit, "__block_descriptor",
                                           Unit, 0, FieldOffset, 0, 0, Flags,
                                           llvm::DIType(), Elements);

  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  DescTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type,
                                          Unit, "", Unit,
                                          0, Size, Align, 0, 0, EltTy);

  FieldOffset = 0;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__isa", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().IntTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__flags", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().IntTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__reserved", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__FuncPtr", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  FieldTy = DescTy;
  FieldSize = CGM.getContext().getTypeSize(Ty);
  FieldAlign = CGM.getContext().getTypeAlign(Ty);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__descriptor", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);

  FieldOffset += FieldSize;
  Elements = DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

  EltTy = DebugFactory.CreateCompositeType(Tag, Unit, "__block_literal_generic",
                                           Unit, 0, FieldOffset, 0, 0, Flags,
                                           llvm::DIType(), Elements);

  BlockLiteralGenericSet = true;
  BlockLiteralGeneric
    = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type, Unit,
                                     "", Unit,
                                     0, Size, Align, 0, 0, EltTy);
  return BlockLiteralGeneric;
}

llvm::DIType CGDebugInfo::CreateType(const TypedefType *Ty,
                                     llvm::DIFile Unit) {
  // Typedefs are derived from some other type.  If we have a typedef of a
  // typedef, make sure to emit the whole chain.
  llvm::DIType Src = getOrCreateType(Ty->getDecl()->getUnderlyingType(), Unit);

  // We don't set size information, but do specify where the typedef was
  // declared.
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(Ty->getDecl()->getLocation());
  unsigned Line = PLoc.isInvalid() ? 0 : PLoc.getLine();

  llvm::DIDescriptor TyContext 
    = getContextDescriptor(dyn_cast<Decl>(Ty->getDecl()->getDeclContext()),
                           Unit);
  llvm::DIType DbgTy = 
    DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_typedef, 
                                   TyContext,
                                   Ty->getDecl()->getName(), Unit,
                                   Line, 0, 0, 0, 0, Src);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const FunctionType *Ty,
                                     llvm::DIFile Unit) {
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
                                     Unit, "", Unit,
                                     0, 0, 0, 0, 0,
                                     llvm::DIType(), EltTypeArray);
  return DbgTy;
}

/// CollectRecordFields - A helper function to collect debug info for
/// record fields. This is used while creating debug info entry for a Record.
void CGDebugInfo::
CollectRecordFields(const RecordDecl *RD, llvm::DIFile Unit,
                    llvm::SmallVectorImpl<llvm::DIDescriptor> &EltTys) {
  unsigned FieldNo = 0;
  SourceManager &SM = CGM.getContext().getSourceManager();
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
    SourceLocation FieldDefLoc = Field->getLocation();
    PresumedLoc PLoc = SM.getPresumedLoc(FieldDefLoc);
    llvm::DIFile FieldDefUnit;
    unsigned FieldLine = 0;

    if (!PLoc.isInvalid()) {
      FieldDefUnit = getOrCreateFile(FieldDefLoc);
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

/// getOrCreateMethodType - CXXMethodDecl's type is a FunctionType. This
/// function type is not updated to include implicit "this" pointer. Use this
/// routine to get a method type which includes "this" pointer.
llvm::DIType
CGDebugInfo::getOrCreateMethodType(const CXXMethodDecl *Method,
                                   llvm::DIFile Unit) {
  llvm::DIType FnTy = getOrCreateType(Method->getType(), Unit);
  
  // Static methods do not need "this" pointer argument.
  if (Method->isStatic())
    return FnTy;

  // Add "this" pointer.

  llvm::DIArray Args = llvm::DICompositeType(FnTy.getNode()).getTypeArray();
  assert (Args.getNumElements() && "Invalid number of arguments!");

  llvm::SmallVector<llvm::DIDescriptor, 16> Elts;

  // First element is always return type. For 'void' functions it is NULL.
  Elts.push_back(Args.getElement(0));

  // "this" pointer is always first argument.
  ASTContext &Context = CGM.getContext();
  QualType ThisPtr = 
    Context.getPointerType(Context.getTagDeclType(Method->getParent()));
  llvm::DIType ThisPtrType = 
    DebugFactory.CreateArtificialType(getOrCreateType(ThisPtr, Unit));
  TypeCache[ThisPtr.getAsOpaquePtr()] = ThisPtrType.getNode();  
  Elts.push_back(ThisPtrType);

  // Copy rest of the arguments.
  for (unsigned i = 1, e = Args.getNumElements(); i != e; ++i)
    Elts.push_back(Args.getElement(i));

  llvm::DIArray EltTypeArray =
    DebugFactory.GetOrCreateArray(Elts.data(), Elts.size());

  return
    DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_subroutine_type,
                                     Unit, "", Unit,
                                     0, 0, 0, 0, 0,
                                     llvm::DIType(), EltTypeArray);
}

/// CreateCXXMemberFunction - A helper function to create a DISubprogram for
/// a single member function GlobalDecl.
llvm::DISubprogram
CGDebugInfo::CreateCXXMemberFunction(const CXXMethodDecl *Method,
                                     llvm::DIFile Unit,
                                     llvm::DICompositeType &RecordTy) {
  bool IsCtorOrDtor = 
    isa<CXXConstructorDecl>(Method) || isa<CXXDestructorDecl>(Method);
  
  llvm::StringRef MethodName = getFunctionName(Method);
  llvm::DIType MethodTy = getOrCreateMethodType(Method, Unit);
  
  // Since a single ctor/dtor corresponds to multiple functions, it doesn't
  // make sense to give a single ctor/dtor a linkage name.
  MangleBuffer MethodLinkageName;
  if (!IsCtorOrDtor)
    CGM.getMangledName(MethodLinkageName, Method);

  SourceManager &SM = CGM.getContext().getSourceManager();

  // Get the location for the method.
  SourceLocation MethodDefLoc = Method->getLocation();
  PresumedLoc PLoc = SM.getPresumedLoc(MethodDefLoc);
  llvm::DIFile MethodDefUnit;
  unsigned MethodLine = 0;

  if (!PLoc.isInvalid()) {
    MethodDefUnit = getOrCreateFile(MethodDefLoc);
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
      VIndex = CGM.getVTables().getMethodVtableIndex(Method);
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
CollectCXXMemberFunctions(const CXXRecordDecl *RD, llvm::DIFile Unit,
                          llvm::SmallVectorImpl<llvm::DIDescriptor> &EltTys,
                          llvm::DICompositeType &RecordTy) {
  for(CXXRecordDecl::method_iterator I = RD->method_begin(),
        E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *Method = *I;
    
    if (Method->isImplicit() && !Method->isUsed())
      continue;

    EltTys.push_back(CreateCXXMemberFunction(Method, Unit, RecordTy));
  }
}                                 

/// CollectCXXBases - A helper function to collect debug info for
/// C++ base classes. This is used while creating debug info entry for 
/// a Record.
void CGDebugInfo::
CollectCXXBases(const CXXRecordDecl *RD, llvm::DIFile Unit,
                llvm::SmallVectorImpl<llvm::DIDescriptor> &EltTys,
                llvm::DICompositeType &RecordTy) {

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
                                     Unit, 0, 0, 0,
                                     BaseOffset, BFlags,
                                     getOrCreateType(BI->getType(),
                                                     Unit));
    EltTys.push_back(DTy);
  }
}

/// getOrCreateVTablePtrType - Return debug info descriptor for vtable.
llvm::DIType CGDebugInfo::getOrCreateVTablePtrType(llvm::DIFile Unit) {
  if (VTablePtrType.isValid())
    return VTablePtrType;

  ASTContext &Context = CGM.getContext();

  /* Function type */
  llvm::DIDescriptor STy = getOrCreateType(Context.IntTy, Unit);
  llvm::DIArray SElements = DebugFactory.GetOrCreateArray(&STy, 1);
  llvm::DIType SubTy =
    DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_subroutine_type,
                                     Unit, "", Unit,
                                     0, 0, 0, 0, 0, llvm::DIType(), SElements);

  unsigned Size = Context.getTypeSize(Context.VoidPtrTy);
  llvm::DIType vtbl_ptr_type 
    = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type,
                                     Unit, "__vtbl_ptr_type", Unit,
                                     0, Size, 0, 0, 0, SubTy);

  VTablePtrType = 
    DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_pointer_type,
                                   Unit, "", Unit,
                                   0, Size, 0, 0, 0, vtbl_ptr_type);
  return VTablePtrType;
}

/// getVtableName - Get vtable name for the given Class.
llvm::StringRef CGDebugInfo::getVtableName(const CXXRecordDecl *RD) {
  // Otherwise construct gdb compatible name name.
  std::string Name = "_vptr$" + RD->getNameAsString();

  // Copy this name on the side and use its reference.
  char *StrPtr = DebugInfoNames.Allocate<char>(Name.length());
  memcpy(StrPtr, Name.data(), Name.length());
  return llvm::StringRef(StrPtr, Name.length());
}


/// CollectVtableInfo - If the C++ class has vtable info then insert appropriate
/// debug info entry in EltTys vector.
void CGDebugInfo::
CollectVtableInfo(const CXXRecordDecl *RD, llvm::DIFile Unit,
                  llvm::SmallVectorImpl<llvm::DIDescriptor> &EltTys) {
  const ASTRecordLayout &RL = CGM.getContext().getASTRecordLayout(RD);

  // If there is a primary base then it will hold vtable info.
  if (RL.getPrimaryBase())
    return;

  // If this class is not dynamic then there is not any vtable info to collect.
  if (!RD->isDynamicClass())
    return;

  unsigned Size = CGM.getContext().getTypeSize(CGM.getContext().VoidPtrTy);
  llvm::DIType VPTR
    = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                     getVtableName(RD), Unit,
                                     0, Size, 0, 0, 0, 
                                     getOrCreateVTablePtrType(Unit));
  EltTys.push_back(VPTR);
}

/// CreateType - get structure or union type.
llvm::DIType CGDebugInfo::CreateType(const RecordType *Ty,
                                     llvm::DIFile Unit) {
  RecordDecl *RD = Ty->getDecl();

  unsigned Tag;
  if (RD->isStruct())
    Tag = llvm::dwarf::DW_TAG_structure_type;
  else if (RD->isUnion())
    Tag = llvm::dwarf::DW_TAG_union_type;
  else {
    assert(RD->isClass() && "Unknown RecordType!");
    Tag = llvm::dwarf::DW_TAG_class_type;
  }

  SourceManager &SM = CGM.getContext().getSourceManager();

  // Get overall information about the record type for the debug info.
  PresumedLoc PLoc = SM.getPresumedLoc(RD->getLocation());
  llvm::DIFile DefUnit;
  unsigned Line = 0;
  if (!PLoc.isInvalid()) {
    DefUnit = getOrCreateFile(RD->getLocation());
    Line = PLoc.getLine();
  }

  // Records and classes and unions can all be recursive.  To handle them, we
  // first generate a debug descriptor for the struct as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the struct is recursive) and replace all
  // uses of the forward declaration with the final definition.

  // A RD->getName() is not unique. However, the debug info descriptors 
  // are uniqued so use type name to ensure uniquness.
  llvm::SmallString<128> FwdDeclName;
  llvm::raw_svector_ostream(FwdDeclName) << "fwd.type." << FwdDeclCount++;
  llvm::DIDescriptor FDContext = 
    getContextDescriptor(dyn_cast<Decl>(RD->getDeclContext()), Unit);
  llvm::DICompositeType FwdDecl =
    DebugFactory.CreateCompositeType(Tag, FDContext, FwdDeclName,
                                     DefUnit, Line, 0, 0, 0, 0,
                                     llvm::DIType(), llvm::DIArray());

  // If this is just a forward declaration, return it.
  if (!RD->getDefinition())
    return FwdDecl;

  llvm::TrackingVH<llvm::MDNode> FwdDeclNode = FwdDecl.getNode();
  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl.getNode();
  // Push the struct on region stack.
  RegionStack.push_back(FwdDecl.getNode());
  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl.getNode());

  // Convert all the elements.
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  const CXXRecordDecl *CXXDecl = dyn_cast<CXXRecordDecl>(RD);
  if (CXXDecl) {
    CollectCXXBases(CXXDecl, Unit, EltTys, FwdDecl);
    CollectVtableInfo(CXXDecl, Unit, EltTys);
  }
  CollectRecordFields(RD, Unit, EltTys);
  llvm::MDNode *ContainingType = NULL;
  if (CXXDecl) {
    CollectCXXMemberFunctions(CXXDecl, Unit, EltTys, FwdDecl);

    // A class's primary base or the class itself contains the vtable.
    const ASTRecordLayout &RL = CGM.getContext().getASTRecordLayout(RD);
    if (const CXXRecordDecl *PBase = RL.getPrimaryBase())
      ContainingType = 
        getOrCreateType(QualType(PBase->getTypeForDecl(), 0), Unit).getNode();
    else if (CXXDecl->isDynamicClass()) 
      ContainingType = FwdDecl.getNode();
  }

  llvm::DIArray Elements =
    DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());

  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  RegionStack.pop_back();
  llvm::DenseMap<const Decl *, llvm::WeakVH>::iterator RI = 
    RegionMap.find(Ty->getDecl());
  if (RI != RegionMap.end())
    RegionMap.erase(RI);

  llvm::DIDescriptor RDContext =  
    getContextDescriptor(dyn_cast<Decl>(RD->getDeclContext()), Unit);
  llvm::DICompositeType RealDecl =
    DebugFactory.CreateCompositeType(Tag, RDContext,
                                     RD->getName(),
                                     DefUnit, Line, Size, Align, 0, 0, 
                                     llvm::DIType(), Elements, 
                                     0, ContainingType);

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  llvm::DIDerivedType(FwdDeclNode).replaceAllUsesWith(RealDecl);
  RegionMap[RD] = llvm::WeakVH(RealDecl.getNode());
  return RealDecl;
}

/// CreateType - get objective-c interface type.
llvm::DIType CGDebugInfo::CreateType(const ObjCInterfaceType *Ty,
                                     llvm::DIFile Unit) {
  ObjCInterfaceDecl *ID = Ty->getDecl();

  unsigned Tag = llvm::dwarf::DW_TAG_structure_type;
  SourceManager &SM = CGM.getContext().getSourceManager();

  // Get overall information about the record type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(ID->getLocation());
  PresumedLoc PLoc = SM.getPresumedLoc(ID->getLocation());
  unsigned Line = PLoc.isInvalid() ? 0 : PLoc.getLine();


  unsigned RuntimeLang = TheCU.getLanguage();

  // To handle recursive interface, we
  // first generate a debug descriptor for the struct as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the struct is recursive) and replace all
  // uses of the forward declaration with the final definition.
  llvm::DICompositeType FwdDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, ID->getName(),
                                     DefUnit, Line, 0, 0, 0, 0,
                                     llvm::DIType(), llvm::DIArray(),
                                     RuntimeLang);

  // If this is just a forward declaration, return it.
  if (ID->isForwardDecl())
    return FwdDecl;

  llvm::TrackingVH<llvm::MDNode> FwdDeclNode = FwdDecl.getNode();
  // Otherwise, insert it into the TypeCache so that recursive uses will find
  // it.
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl.getNode();
  // Push the struct on region stack.
  RegionStack.push_back(FwdDecl.getNode());
  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl.getNode());

  // Convert all the elements.
  llvm::SmallVector<llvm::DIDescriptor, 16> EltTys;

  ObjCInterfaceDecl *SClass = ID->getSuperClass();
  if (SClass) {
    llvm::DIType SClassTy =
      getOrCreateType(CGM.getContext().getObjCInterfaceType(SClass), Unit);
    llvm::DIType InhTag =
      DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_inheritance,
                                     Unit, "", Unit, 0, 0, 0,
                                     0 /* offset */, 0, SClassTy);
    EltTys.push_back(InhTag);
  }

  const ASTRecordLayout &RL = CGM.getContext().getASTObjCInterfaceLayout(ID);

  unsigned FieldNo = 0;
  for (ObjCInterfaceDecl::ivar_iterator I = ID->ivar_begin(),
         E = ID->ivar_end();  I != E; ++I, ++FieldNo) {
    ObjCIvarDecl *Field = *I;
    llvm::DIType FieldTy = getOrCreateType(Field->getType(), Unit);

    llvm::StringRef FieldName = Field->getName();

    // Ignore unnamed fields.
    if (FieldName.empty())
      continue;

    // Get the location for the field.
    SourceLocation FieldDefLoc = Field->getLocation();
    llvm::DIFile FieldDefUnit = getOrCreateFile(FieldDefLoc);
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

  RegionStack.pop_back();
  llvm::DenseMap<const Decl *, llvm::WeakVH>::iterator RI = 
    RegionMap.find(Ty->getDecl());
  if (RI != RegionMap.end())
    RegionMap.erase(RI);

  // Bit size, align and offset of the type.
  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  llvm::DICompositeType RealDecl =
    DebugFactory.CreateCompositeType(Tag, Unit, ID->getName(), DefUnit,
                                     Line, Size, Align, 0, 0, llvm::DIType(), 
                                     Elements, RuntimeLang);

  // Now that we have a real decl for the struct, replace anything using the
  // old decl with the new one.  This will recursively update the debug info.
  llvm::DIDerivedType(FwdDeclNode).replaceAllUsesWith(RealDecl);
  RegionMap[ID] = llvm::WeakVH(RealDecl.getNode());

  return RealDecl;
}

llvm::DIType CGDebugInfo::CreateType(const EnumType *Ty,
                                     llvm::DIFile Unit) {
  EnumDecl *ED = Ty->getDecl();

  llvm::SmallVector<llvm::DIDescriptor, 32> Enumerators;

  // Create DIEnumerator elements for each enumerator.
  for (EnumDecl::enumerator_iterator
         Enum = ED->enumerator_begin(), EnumEnd = ED->enumerator_end();
       Enum != EnumEnd; ++Enum) {
    Enumerators.push_back(DebugFactory.CreateEnumerator(Enum->getName(),
                                            Enum->getInitVal().getZExtValue()));
  }

  // Return a CompositeType for the enum itself.
  llvm::DIArray EltArray =
    DebugFactory.GetOrCreateArray(Enumerators.data(), Enumerators.size());

  SourceLocation DefLoc = ED->getLocation();
  llvm::DIFile DefUnit = getOrCreateFile(DefLoc);
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
                                     Unit, ED->getName(), DefUnit, Line,
                                     Size, Align, 0, 0,
                                     llvm::DIType(), EltArray);
  return DbgTy;
}

llvm::DIType CGDebugInfo::CreateType(const TagType *Ty,
                                     llvm::DIFile Unit) {
  if (const RecordType *RT = dyn_cast<RecordType>(Ty))
    return CreateType(RT, Unit);
  else if (const EnumType *ET = dyn_cast<EnumType>(Ty))
    return CreateType(ET, Unit);

  return llvm::DIType();
}

llvm::DIType CGDebugInfo::CreateType(const VectorType *Ty,
				     llvm::DIFile Unit) {
  llvm::DIType ElementTy = getOrCreateType(Ty->getElementType(), Unit);
  uint64_t NumElems = Ty->getNumElements();
  if (NumElems > 0)
    --NumElems;

  llvm::DIDescriptor Subscript = DebugFactory.GetOrCreateSubrange(0, NumElems);
  llvm::DIArray SubscriptArray = DebugFactory.GetOrCreateArray(&Subscript, 1);

  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);

  return
    DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_vector_type,
                                     Unit, "", Unit,
                                     0, Size, Align, 0, 0,
				     ElementTy,  SubscriptArray);
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
                                     Unit, "", Unit,
                                     0, Size, Align, 0, 0,
                                     getOrCreateType(EltTy, Unit),
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
  llvm::DIDescriptor ElementTypes[2];
  
  // FIXME: This should probably be a function type instead.
  ElementTypes[0] =
    DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, U,
                                   "ptr", U, 0,
                                   Info.first, Info.second, FieldOffset, 0,
                                   PointerDiffDITy);
  FieldOffset += Info.first;
  
  ElementTypes[1] =
    DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, U,
                                   "ptr", U, 0,
                                   Info.first, Info.second, FieldOffset, 0,
                                   PointerDiffDITy);
  
  llvm::DIArray Elements = 
    DebugFactory.GetOrCreateArray(&ElementTypes[0],
                                  llvm::array_lengthof(ElementTypes));

  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_structure_type, 
                                          U, llvm::StringRef("test"), 
                                          U, 0, FieldOffset, 
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
  TypeCache[Ty.getAsOpaquePtr()] = Res.getNode();  
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

  case Type::InjectedClassName:
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
  MangleBuffer LinkageName;

  const Decl *D = GD.getDecl();
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // If there is a DISubprogram for  this function available then use it.
    llvm::DenseMap<const FunctionDecl *, llvm::WeakVH>::iterator
      FI = SPCache.find(FD);
    if (FI != SPCache.end()) {
      llvm::DIDescriptor SP(dyn_cast_or_null<llvm::MDNode>(FI->second));
      if (SP.isSubprogram() && llvm::DISubprogram(SP.getNode()).isDefinition()) {
        RegionStack.push_back(SP.getNode());
        RegionMap[D] = llvm::WeakVH(SP.getNode());
        return;
      }
    }
    Name = getFunctionName(FD);
    if (!Name.empty() && Name[0] == '\01')
      Name = Name.substr(1);
    // Use mangled name as linkage name for c/c++ functions.
    CGM.getMangledName(LinkageName, GD);
  } else {
    // Use llvm function name as linkage name.
    Name = Fn->getName();
    LinkageName.setString(Name);
    if (!Name.empty() && Name[0] == '\01')
      Name = Name.substr(1);
  }

  // It is expected that CurLoc is set before using EmitFunctionStart.
  // Usually, CurLoc points to the left bracket location of compound
  // statement representing function body.
  llvm::DIFile Unit = getOrCreateFile(CurLoc);
  SourceManager &SM = CGM.getContext().getSourceManager();
  unsigned LineNo = SM.getPresumedLoc(CurLoc).getLine();

  llvm::DISubprogram SP =
    DebugFactory.CreateSubprogram(Unit, Name, Name, LinkageName, Unit, LineNo,
                                  getOrCreateType(FnType, Unit),
                                  Fn->hasInternalLinkage(), true/*definition*/);

  // Push function on region stack.
  RegionStack.push_back(SP.getNode());
  RegionMap[D] = llvm::WeakVH(SP.getNode());
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
  llvm::DIFile Unit = getOrCreateFile(CurLoc);
  PresumedLoc PLoc = SM.getPresumedLoc(CurLoc);

  llvm::MDNode *Scope = RegionStack.back();
  Builder.SetCurrentDebugLocation(llvm::DebugLoc::get(PLoc.getLine(),
                                                      PLoc.getColumn(),
                                                      Scope));
}

/// EmitRegionStart- Constructs the debug code for entering a declarative
/// region - "llvm.dbg.region.start.".
void CGDebugInfo::EmitRegionStart(llvm::Function *Fn, CGBuilderTy &Builder) {
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(CurLoc);
  llvm::DIDescriptor D =
    DebugFactory.CreateLexicalBlock(RegionStack.empty() ? 
                                    llvm::DIDescriptor() : 
                                    llvm::DIDescriptor(RegionStack.back()),
                                    PLoc.getLine(), PLoc.getColumn());
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

// EmitTypeForVarWithBlocksAttr - Build up structure info for the byref.  
// See BuildByRefType.
llvm::DIType CGDebugInfo::EmitTypeForVarWithBlocksAttr(const ValueDecl *VD,
                                                       uint64_t *XOffset) {

  llvm::SmallVector<llvm::DIDescriptor, 5> EltTys;

  QualType FType;
  uint64_t FieldSize, FieldOffset;
  unsigned FieldAlign;
  
  llvm::DIFile Unit = getOrCreateFile(VD->getLocation());
  QualType Type = VD->getType();  

  FieldOffset = 0;
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  llvm::DIType FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__isa", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);
  FieldOffset += FieldSize;
  
  FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__forwarding", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);
  FieldOffset += FieldSize;
  
  FType = CGM.getContext().IntTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__flags", Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);
  FieldOffset += FieldSize;
  
  FType = CGM.getContext().IntTy;
  FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
  FieldSize = CGM.getContext().getTypeSize(FType);
  FieldAlign = CGM.getContext().getTypeAlign(FType);
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           "__size", Unit,
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
                                             "__copy_helper", Unit,                                             
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;
    
    FType = CGM.getContext().getPointerType(CGM.getContext().VoidTy);
    FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
    FieldSize = CGM.getContext().getTypeSize(FType);
    FieldAlign = CGM.getContext().getTypeAlign(FType);
    FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                             "__destroy_helper", Unit,
                                             0, FieldSize, FieldAlign,
                                             FieldOffset, 0, FieldTy);
    EltTys.push_back(FieldTy);
    FieldOffset += FieldSize;
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
      FieldTy = CGDebugInfo::getOrCreateType(FType, Unit);
      FieldSize = CGM.getContext().getTypeSize(FType);
      FieldAlign = CGM.getContext().getTypeAlign(FType);
      FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member,
                                               Unit, "", Unit,
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

  *XOffset = FieldOffset;  
  FieldTy = DebugFactory.CreateDerivedType(llvm::dwarf::DW_TAG_member, Unit,
                                           VD->getName(), Unit,
                                           0, FieldSize, FieldAlign,
                                           FieldOffset, 0, FieldTy);
  EltTys.push_back(FieldTy);
  FieldOffset += FieldSize;
  
  llvm::DIArray Elements = 
    DebugFactory.GetOrCreateArray(EltTys.data(), EltTys.size());
  
  unsigned Flags = llvm::DIType::FlagBlockByrefStruct;
  
  return DebugFactory.CreateCompositeType(llvm::dwarf::DW_TAG_structure_type, 
                                          Unit, "", Unit,
                                          0, FieldOffset, 0, 0, Flags,
                                          llvm::DIType(), Elements);
  
}
/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const VarDecl *VD, unsigned Tag,
                              llvm::Value *Storage, CGBuilderTy &Builder) {
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Do not emit variable debug information while generating optimized code.
  // The llvm optimizer and code generator are not yet ready to support
  // optimized code debugging.
  const CodeGenOptions &CGO = CGM.getCodeGenOpts();
  if (CGO.OptimizationLevel)
    return;

  llvm::DIFile Unit = getOrCreateFile(VD->getLocation());
  llvm::DIType Ty;
  uint64_t XOffset = 0;
  if (VD->hasAttr<BlocksAttr>())
    Ty = EmitTypeForVarWithBlocksAttr(VD, &XOffset);
  else 
    Ty = getOrCreateType(VD->getType(), Unit);

  // Get location information.
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(VD->getLocation());
  unsigned Line = 0;
  unsigned Column = 0;
  if (PLoc.isInvalid())
    PLoc = SM.getPresumedLoc(CurLoc);
  if (PLoc.isValid()) {
    Line = PLoc.getLine();
    Column = PLoc.getColumn();
    Unit = getOrCreateFile(CurLoc);
  } else {
    Unit = llvm::DIFile();
  }

  // Create the descriptor for the variable.
  llvm::DIVariable D =
    DebugFactory.CreateVariable(Tag, llvm::DIDescriptor(RegionStack.back()),
                                VD->getName(),
                                Unit, Line, Ty);
  // Insert an llvm.dbg.declare into the current block.
  llvm::Instruction *Call =
    DebugFactory.InsertDeclare(Storage, D, Builder.GetInsertBlock());

  llvm::MDNode *Scope = RegionStack.back();
  Call->setDebugLoc(llvm::DebugLoc::get(Line, Column, Scope));
}

/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const BlockDeclRefExpr *BDRE, unsigned Tag,
                              llvm::Value *Storage, CGBuilderTy &Builder,
                              CodeGenFunction *CGF) {
  const ValueDecl *VD = BDRE->getDecl();
  assert(!RegionStack.empty() && "Region stack mismatch, stack empty!");

  // Do not emit variable debug information while generating optimized code.
  // The llvm optimizer and code generator are not yet ready to support
  // optimized code debugging.
  const CodeGenOptions &CGO = CGM.getCodeGenOpts();
  if (CGO.OptimizationLevel || Builder.GetInsertBlock() == 0)
    return;

  uint64_t XOffset = 0;
  llvm::DIFile Unit = getOrCreateFile(VD->getLocation());
  llvm::DIType Ty;
  if (VD->hasAttr<BlocksAttr>())
    Ty = EmitTypeForVarWithBlocksAttr(VD, &XOffset);
  else 
    Ty = getOrCreateType(VD->getType(), Unit);

  // Get location information.
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(VD->getLocation());
  unsigned Line = 0;
  if (!PLoc.isInvalid())
    Line = PLoc.getLine();
  else
    Unit = llvm::DIFile();

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
    DebugFactory.CreateComplexVariable(Tag,
                                       llvm::DIDescriptor(RegionStack.back()),
                                       VD->getName(), Unit, Line, Ty,
                                       addr);
  // Insert an llvm.dbg.declare into the current block.
  llvm::Instruction *Call = 
    DebugFactory.InsertDeclare(Storage, D, Builder.GetInsertBlock());
  
  llvm::MDNode *Scope = RegionStack.back();
  Call->setDebugLoc(llvm::DebugLoc::get(Line, PLoc.getColumn(), Scope));
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
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(D->getLocation());
  unsigned LineNo = PLoc.isInvalid() ? 0 : PLoc.getLine();

  QualType T = D->getType();
  if (T->isIncompleteArrayType()) {

    // CodeGen turns int[] into int[1] so we'll do the same here.
    llvm::APSInt ConstVal(32);

    ConstVal = 1;
    QualType ET = CGM.getContext().getAsArrayType(T)->getElementType();

    T = CGM.getContext().getConstantArrayType(ET, ConstVal,
                                           ArrayType::Normal, 0);
  }
  llvm::StringRef DeclName = Var->getName();
  llvm::DIDescriptor DContext = 
    getContextDescriptor(dyn_cast<Decl>(D->getDeclContext()), Unit);
  DebugFactory.CreateGlobalVariable(DContext, DeclName,
                                    DeclName, llvm::StringRef(), Unit, LineNo,
                                    getOrCreateType(T, Unit),
                                    Var->hasInternalLinkage(),
                                    true/*definition*/, Var);
}

/// EmitGlobalVariable - Emit information about an objective-c interface.
void CGDebugInfo::EmitGlobalVariable(llvm::GlobalVariable *Var,
                                     ObjCInterfaceDecl *ID) {
  // Create global variable debug descriptor.
  llvm::DIFile Unit = getOrCreateFile(ID->getLocation());
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(ID->getLocation());
  unsigned LineNo = PLoc.isInvalid() ? 0 : PLoc.getLine();

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

  DebugFactory.CreateGlobalVariable(Unit, Name, Name, Name, Unit, LineNo,
                                    getOrCreateType(T, Unit),
                                    Var->hasInternalLinkage(),
                                    true/*definition*/, Var);
}

/// getOrCreateNamesSpace - Return namespace descriptor for the given
/// namespace decl.
llvm::DINameSpace 
CGDebugInfo::getOrCreateNameSpace(const NamespaceDecl *NSDecl, 
                                  llvm::DIDescriptor Unit) {
  llvm::DenseMap<const NamespaceDecl *, llvm::WeakVH>::iterator I = 
    NameSpaceCache.find(NSDecl);
  if (I != NameSpaceCache.end())
    return llvm::DINameSpace(cast<llvm::MDNode>(I->second));
  
  SourceManager &SM = CGM.getContext().getSourceManager();
  PresumedLoc PLoc = SM.getPresumedLoc(NSDecl->getLocation());
  unsigned LineNo = PLoc.isInvalid() ? 0 : PLoc.getLine();

  llvm::DIDescriptor Context = 
    getContextDescriptor(dyn_cast<Decl>(NSDecl->getDeclContext()), Unit);
  llvm::DINameSpace NS =
    DebugFactory.CreateNameSpace(Context, NSDecl->getName(), 
	llvm::DIFile(Unit.getNode()), LineNo);
  NameSpaceCache[NSDecl] = llvm::WeakVH(NS.getNode());
  return NS;
}
