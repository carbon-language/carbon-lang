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
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetMachine.h"
using namespace clang;
using namespace clang::CodeGen;

CGDebugInfo::CGDebugInfo(CodeGenModule *m)
: M(m)
, CurLoc()
, PrevLoc()
, CompileUnitCache()
, TypeCache()
, StopPointFn(NULL)
, FuncStartFn(NULL)
, DeclareFn(NULL)
, RegionStartFn(NULL)
, RegionEndFn(NULL)
, CompileUnitAnchor(NULL)
, SubprogramAnchor(NULL)
, GlobalVariableAnchor(NULL)
, RegionStack()
, VariableDescList()
, GlobalVarDescList()
, EnumDescList()
, SubrangeDescList()
, Subprogram(NULL)
{
  SR = new llvm::DISerializer();
  SR->setModule (&M->getModule());
}

CGDebugInfo::~CGDebugInfo()
{
  delete SR;

  // Free CompileUnitCache.
  for (std::map<unsigned, llvm::CompileUnitDesc *>::iterator I 
       = CompileUnitCache.begin(); I != CompileUnitCache.end(); ++I) {
    delete I->second;
  }
  CompileUnitCache.clear();

  // Free TypeCache.
  for (std::map<void *, llvm::TypeDesc *>::iterator I 
       = TypeCache.begin(); I != TypeCache.end(); ++I) {
    delete I->second;
  }
  TypeCache.clear();

  // Free region descriptors.
  for (std::vector<llvm::DebugInfoDesc *>::iterator I 
       = RegionStack.begin(); I != RegionStack.end(); ++I) {
    delete *I;
  }

  // Free local var descriptors.
  for (std::vector<llvm::VariableDesc *>::iterator I 
       = VariableDescList.begin(); I != VariableDescList.end(); ++I) {
    delete *I;
  }

  // Free global var descriptors.
  for (std::vector<llvm::GlobalVariableDesc *>::iterator I 
       = GlobalVarDescList.begin(); I != GlobalVarDescList.end(); ++I) {
    delete *I;
  }

  // Free enum constants descriptors.
  for (std::vector<llvm::EnumeratorDesc *>::iterator I 
       = EnumDescList.begin(); I != EnumDescList.end(); ++I) {
    delete *I;
  }

  // Free subrange descriptors.
  for (std::vector<llvm::SubrangeDesc *>::iterator I
       = SubrangeDescList.begin(); I != SubrangeDescList.end(); ++I) {
    delete *I;
  }

  delete CompileUnitAnchor;
  delete SubprogramAnchor;
  delete GlobalVariableAnchor;
}

void CGDebugInfo::setLocation(SourceLocation loc) {
  CurLoc = M->getContext().getSourceManager().getLogicalLoc(loc);
}

/// getCastValueFor - Return a llvm representation for a given debug information
/// descriptor cast to an empty struct pointer.
llvm::Value *CGDebugInfo::getCastValueFor(llvm::DebugInfoDesc *DD) {
  return llvm::ConstantExpr::getBitCast(SR->Serialize(DD), 
                                        SR->getEmptyStructPtrType());
}

/// getValueFor - Return a llvm representation for a given debug information
/// descriptor.
llvm::Value *CGDebugInfo::getValueFor(llvm::DebugInfoDesc *DD) {
  return SR->Serialize(DD);
}

/// getOrCreateCompileUnit - Get the compile unit from the cache or create a new
/// one if necessary.
llvm::CompileUnitDesc 
*CGDebugInfo::getOrCreateCompileUnit(const SourceLocation Loc) {

  // See if this compile unit has been used before.
  llvm::CompileUnitDesc *&Slot = CompileUnitCache[Loc.getFileID()];
  if (Slot) return Slot;

  // Create new compile unit.
  // FIXME: Where to free these?
  // One way is to iterate over the CompileUnitCache in ~CGDebugInfo.
  llvm::CompileUnitDesc *Unit = new llvm::CompileUnitDesc();

  // Make sure we have an anchor.
  if (!CompileUnitAnchor) {
    CompileUnitAnchor = new llvm::AnchorDesc(Unit);
  }

  // Get source file information.
  SourceManager &SM = M->getContext().getSourceManager();
  const FileEntry *FE = SM.getFileEntryForLoc(Loc);
  const char *FileName, *DirName;
  if (FE) {
    FileName = FE->getName();
    DirName = FE->getDir()->getName();
  } else {
    FileName = SM.getSourceName(Loc);
    DirName = "";
  }

  Unit->setAnchor(CompileUnitAnchor);
  Unit->setFileName(FileName);
  Unit->setDirectory(DirName);

  // Set up producer name.
  // FIXME: Do not know how to get clang version yet.
  Unit->setProducer("clang");

  // Set up Language number.
  // FIXME: Handle other languages as well.
  Unit->setLanguage(llvm::dwarf::DW_LANG_C89);

  // Update cache.
  Slot = Unit;

  return Unit;
}


/// getOrCreateCVRType - Get the CVR qualified type from the cache or create 
/// a new one if necessary.
llvm::TypeDesc *
CGDebugInfo::getOrCreateCVRType(QualType type, llvm::CompileUnitDesc *Unit)
{
  // We will create a Derived type.
  llvm::DerivedTypeDesc *DTy = NULL;
  llvm::TypeDesc *FromTy = NULL;

  if (type.isConstQualified()) {
    DTy = new llvm::DerivedTypeDesc(llvm::dwarf::DW_TAG_const_type);
    type.removeConst(); 
    FromTy = getOrCreateType(type, Unit);
  } else if (type.isVolatileQualified()) {
    DTy = new llvm::DerivedTypeDesc(llvm::dwarf::DW_TAG_volatile_type);
    type.removeVolatile(); 
    FromTy = getOrCreateType(type, Unit);
  } else if (type.isRestrictQualified()) {
    DTy = new llvm::DerivedTypeDesc(llvm::dwarf::DW_TAG_restrict_type);
    type.removeRestrict(); 
    FromTy = getOrCreateType(type, Unit);
  }

  // No need to fill in the Name, Line, Size, Alignment, Offset in case of        // CVR derived types.
  DTy->setContext(Unit);
  DTy->setFromType(FromTy);

  return DTy;
}

   
/// getOrCreateBuiltinType - Get the Basic type from the cache or create a new
/// one if necessary.
llvm::TypeDesc *
CGDebugInfo::getOrCreateBuiltinType(QualType type, llvm::CompileUnitDesc *Unit)
{
  assert (type->getTypeClass() == Type::Builtin);

  const BuiltinType *BT = type->getAsBuiltinType(); 

  unsigned Encoding = 0;
  switch (BT->getKind())
  {
    case BuiltinType::Void:
      return NULL;
    case BuiltinType::UChar:
    case BuiltinType::Char_U:
      Encoding = llvm::dwarf::DW_ATE_unsigned_char;
      break;
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      Encoding = llvm::dwarf::DW_ATE_signed_char;
      break;
    case BuiltinType::UShort:
    case BuiltinType::UInt:
    case BuiltinType::ULong:
    case BuiltinType::ULongLong:
      Encoding = llvm::dwarf::DW_ATE_unsigned;
      break;
    case BuiltinType::Short:
    case BuiltinType::Int:
    case BuiltinType::Long:
    case BuiltinType::LongLong:
      Encoding = llvm::dwarf::DW_ATE_signed;
      break;
    case BuiltinType::Bool:
      Encoding = llvm::dwarf::DW_ATE_boolean;
      break;
    case BuiltinType::Float:
    case BuiltinType::Double:
      Encoding = llvm::dwarf::DW_ATE_float;
      break;
    default:
      Encoding = llvm::dwarf::DW_ATE_signed;
      break;
  } 

  // Ty will have contain the resulting type.
  llvm::BasicTypeDesc *BTy = new llvm::BasicTypeDesc();

  // Get the name and location early to assist debugging.
  const char *TyName = BT->getName();

  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(type);
  uint64_t Align = M->getContext().getTypeAlign(type);
  uint64_t Offset = 0;
                                                  
  // If the type is defined, fill in the details.
  if (BTy) {
    BTy->setContext(Unit);
    BTy->setName(TyName);
    BTy->setSize(Size);
    BTy->setAlign(Align);
    BTy->setOffset(Offset);
    BTy->setEncoding(Encoding);
  }
                                                                                
  return BTy;
}

llvm::TypeDesc *
CGDebugInfo::getOrCreatePointerType(QualType type, llvm::CompileUnitDesc *Unit)
{
  // type*
  llvm::DerivedTypeDesc *DTy =
    new llvm::DerivedTypeDesc(llvm::dwarf::DW_TAG_pointer_type);

  // Handle the derived type.
  const PointerType *PTRT = type->getAsPointerType();
  llvm::TypeDesc *FromTy = getOrCreateType(PTRT->getPointeeType(), Unit);
 
  // Get the name and location early to assist debugging.
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Line = SM.getLogicalLineNumber(CurLoc);
                                                                               
  // Bit size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(type);
  uint64_t Align = M->getContext().getTypeAlign(type);
  uint64_t Offset = 0;
                                                                               
  // If the type is defined, fill in the details.
  if (DTy) {
    DTy->setContext(Unit);
    DTy->setLine(Line);
    DTy->setSize(Size);
    DTy->setAlign(Align);
    DTy->setOffset(Offset);
    DTy->setFromType(FromTy);
  }

  return DTy;
}

llvm::TypeDesc *
CGDebugInfo::getOrCreateTypedefType(QualType type, llvm::CompileUnitDesc *Unit)
{
  // typedefs are derived from some other type.
  llvm::DerivedTypeDesc *DTy =
    new llvm::DerivedTypeDesc(llvm::dwarf::DW_TAG_typedef);

  // Handle derived type.
  const TypedefType *TDT = type->getAsTypedefType();
  llvm::TypeDesc *FromTy = getOrCreateType(TDT->LookThroughTypedefs(),
                                               Unit);

  // Get the name and location early to assist debugging.
  const char *TyName = TDT->getDecl()->getName();
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Line = SM.getLogicalLineNumber(TDT->getDecl()->getLocation());
                                                                                
  // If the type is defined, fill in the details.
  if (DTy) {
    DTy->setContext(Unit);
    DTy->setFile(getOrCreateCompileUnit(TDT->getDecl()->getLocation()));
    DTy->setLine(Line);
    DTy->setName(TyName);
    DTy->setFromType(FromTy);
  }

  return DTy;
}

llvm::TypeDesc *
CGDebugInfo::getOrCreateFunctionType(QualType type, llvm::CompileUnitDesc *Unit)
{
  llvm::CompositeTypeDesc *SubrTy =
    new llvm::CompositeTypeDesc(llvm::dwarf::DW_TAG_subroutine_type);

  // Prepare to add the arguments for the subroutine.
  std::vector<llvm::DebugInfoDesc *> &Elements = SubrTy->getElements();

  // Get result type.
  const FunctionType *FT = type->getAsFunctionType();
  llvm::TypeDesc *ArgTy = getOrCreateType(FT->getResultType(), Unit);
  if (ArgTy) Elements.push_back(ArgTy);

  // Set up remainder of arguments.
  if (type->getTypeClass() == Type::FunctionProto) {
    const FunctionTypeProto *FTPro = dyn_cast<FunctionTypeProto>(type);
    for (unsigned int i =0; i < FTPro->getNumArgs(); i++) {
      QualType ParamType = FTPro->getArgType(i);
      ArgTy = getOrCreateType(ParamType, Unit);
      if (ArgTy) Elements.push_back(ArgTy);
    }
  }

  // FIXME: set other fields file, line here.
  SubrTy->setContext(Unit);

  return SubrTy;
}

/// getOrCreateRecordType - get structure or union type.
llvm::TypeDesc *
CGDebugInfo::getOrCreateRecordType(QualType type, llvm::CompileUnitDesc *Unit)
{
  llvm::CompositeTypeDesc *RecType;
  if(type->isStructureType())
    RecType = new llvm::CompositeTypeDesc(llvm::dwarf::DW_TAG_structure_type);
  else if(type->isUnionType())
    RecType = new llvm::CompositeTypeDesc(llvm::dwarf::DW_TAG_union_type);
  else
    return NULL;

  RecordDecl *RecDecl = type->getAsRecordType()->getDecl();
  const ASTRecordLayout &RL = M->getContext().getASTRecordLayout(RecDecl);

  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Line = SM.getLogicalLineNumber(RecDecl->getLocation());

  std::vector<llvm::DebugInfoDesc *> &Elements = RecType->getElements();

  // Add the members.
  int NumMembers = RecDecl->getNumMembers();
  for (int i = 0; i < NumMembers; i++) {
    FieldDecl *Member = RecDecl->getMember(i);
    llvm::TypeDesc *MemberTy = getOrCreateType(Member->getType(), Unit);
    MemberTy->setOffset(RL.getFieldOffset(i));
    Elements.push_back(MemberTy);
  }

  // Fill in the blanks.
  if(RecType) {
    RecType->setContext(Unit);
    RecType->setName(RecDecl->getName());
    RecType->setFile(getOrCreateCompileUnit(RecDecl->getLocation()));
    RecType->setLine(Line);
    RecType->setSize(RL.getSize());
    RecType->setAlign(RL.getAlignment());
    RecType->setOffset(0);
  }
  return(RecType);
}

/// getOrCreateEnumType - get Enum type.
llvm::TypeDesc *
CGDebugInfo::getOrCreateEnumType(QualType type, llvm::CompileUnitDesc *Unit)
{
  llvm::CompositeTypeDesc *EnumTy 
    = new llvm::CompositeTypeDesc(llvm::dwarf::DW_TAG_enumeration_type);

  EnumType *EType = dyn_cast<EnumType>(type);
  if (!EType) return(NULL);

  EnumDecl *EDecl = EType->getDecl();
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Line = SM.getLogicalLineNumber(EDecl->getLocation());

  // Size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(type);
  uint64_t Align = M->getContext().getTypeAlign(type);
  
  // Create descriptors for enum members.
  std::vector<llvm::DebugInfoDesc *> &Elements = EnumTy->getElements();
  EnumConstantDecl *ElementList = EDecl->getEnumConstantList();
  while (ElementList) {
    llvm::EnumeratorDesc *EnumDesc = new llvm::EnumeratorDesc();
    // push it to the enum desc list so that we can free it later.
    EnumDescList.push_back(EnumDesc);

    const char *ElementName = ElementList->getName();
    uint64_t Value = ElementList->getInitVal().getZExtValue();

    EnumDesc->setName(ElementName);
    EnumDesc->setValue(Value);
    Elements.push_back(EnumDesc);
    if (ElementList->getNextDeclarator())
      ElementList 
        = dyn_cast<EnumConstantDecl>(ElementList->getNextDeclarator());
    else
      break;
  }

  // Fill in the blanks.
  if (EnumTy) {
    EnumTy->setContext(Unit);
    EnumTy->setName(EDecl->getName());
    EnumTy->setSize(Size);
    EnumTy->setAlign(Align);    
    EnumTy->setOffset(0);
    EnumTy->setFile(getOrCreateCompileUnit(EDecl->getLocation()));
    EnumTy->setLine(Line);
  }
  return EnumTy;
}

/// getOrCreateArrayType - get or create array types.
llvm::TypeDesc *
CGDebugInfo::getOrCreateArrayType(QualType type, llvm::CompileUnitDesc *Unit)
{
  llvm::CompositeTypeDesc *ArrayTy 
    = new llvm::CompositeTypeDesc(llvm::dwarf::DW_TAG_array_type);

  // Size, align and offset of the type.
  uint64_t Size = M->getContext().getTypeSize(type);
  uint64_t Align = M->getContext().getTypeAlign(type);

  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Line = SM.getLogicalLineNumber(CurLoc);

  // Add the dimensions of the array.
  std::vector<llvm::DebugInfoDesc *> &Elements = ArrayTy->getElements();
  do {
    const ArrayType *AT = M->getContext().getAsArrayType(type);
    llvm::SubrangeDesc *Subrange = new llvm::SubrangeDesc();

    // push it back on the subrange desc list so that we can free it later.
    SubrangeDescList.push_back(Subrange);

    uint64_t Upper = 0;
    if (const ConstantArrayType *ConstArrTy = dyn_cast<ConstantArrayType>(AT)) {
      Upper = ConstArrTy->getSize().getZExtValue() - 1;
    }
    Subrange->setLo(0);
    Subrange->setHi(Upper);
    Elements.push_back(Subrange);
    type = AT->getElementType();
  } while (type->isArrayType());

  ArrayTy->setFromType(getOrCreateType(type, Unit));

  if (ArrayTy) {
    ArrayTy->setContext(Unit);
    ArrayTy->setSize(Size);
    ArrayTy->setAlign(Align);
    ArrayTy->setOffset(0);
    ArrayTy->setFile(getOrCreateCompileUnit(CurLoc));
    ArrayTy->setLine(Line);
  }
  return ArrayTy;
}


/// getOrCreateTaggedType - get or create structure/union/Enum type.
llvm::TypeDesc *
CGDebugInfo::getOrCreateTaggedType(QualType type, llvm::CompileUnitDesc *Unit)
{
  if (type->isStructureType() || type->isUnionType())
    return getOrCreateRecordType(type, Unit);
  else if (type->isEnumeralType())
    return getOrCreateEnumType(type, Unit);
  else
    return NULL;
}
  
/// getOrCreateType - Get the type from the cache or create a new
/// one if necessary.
llvm::TypeDesc *
CGDebugInfo::getOrCreateType(QualType type, llvm::CompileUnitDesc *Unit)
{
  if (type.isNull())
    return NULL;

  // Check to see if the compile unit already has created this type.
  llvm::TypeDesc *&Slot = TypeCache[type.getAsOpaquePtr()];
  if (Slot) return Slot;

  // We need to check for the CVR qualifiers as the first thing.
  if (type.getCVRQualifiers()) {
    Slot = getOrCreateCVRType (type, Unit);
    return Slot;
  }

  // Work out details of type.
  switch(type->getTypeClass()) {
    case Type::Complex:
    case Type::Reference:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ASQual:
    case Type::ObjCInterface:
    case Type::ObjCQualifiedInterface:
    case Type::ObjCQualifiedId:
    case Type::TypeOfExp:
    case Type::TypeOfTyp:
    default:
    {
      return NULL;
    }

    case Type::TypeName:
      Slot = getOrCreateTypedefType(type, Unit);
      break;

    case Type::FunctionProto:
    case Type::FunctionNoProto:
      Slot = getOrCreateFunctionType(type, Unit);
      break;

    case Type::Builtin:
      Slot = getOrCreateBuiltinType(type, Unit);
      break;

    case Type::Pointer:
      Slot = getOrCreatePointerType(type, Unit);
      break;

    case Type::Tagged:
      Slot = getOrCreateTaggedType(type, Unit);
      break;

    case Type::ConstantArray:
    case Type::VariableArray:
    case Type::IncompleteArray:
      Slot = getOrCreateArrayType(type, Unit);
      break;
  }

  return Slot;
}

/// EmitFunctionStart - Constructs the debug code for entering a function -
/// "llvm.dbg.func.start.".
void CGDebugInfo::EmitFunctionStart(const FunctionDecl *FnDecl,
                                    llvm::Function *Fn,
                                    llvm::IRBuilder &Builder)
{
  // Create subprogram descriptor.
  Subprogram = new llvm::SubprogramDesc();

  // Make sure we have an anchor.
  if (!SubprogramAnchor) {
    SubprogramAnchor = new llvm::AnchorDesc(Subprogram);
  }

  // Get name information.
  Subprogram->setName(FnDecl->getName());
  Subprogram->setFullName(FnDecl->getName());

  // Gather location information.
  llvm::CompileUnitDesc *Unit = getOrCreateCompileUnit(CurLoc);
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Loc = SM.getLogicalLineNumber(CurLoc);

  // Get Function Type.
  QualType type = FnDecl->getResultType();
  llvm::TypeDesc *SPTy = getOrCreateType(type, Unit);

  Subprogram->setAnchor(SubprogramAnchor);
  Subprogram->setContext(Unit);
  Subprogram->setFile(Unit);
  Subprogram->setLine(Loc);
  Subprogram->setType(SPTy);
  Subprogram->setIsStatic(Fn->hasInternalLinkage());
  Subprogram->setIsDefinition(true);

  // Lazily construct llvm.dbg.func.start.
  if (!FuncStartFn)
    FuncStartFn = llvm::Intrinsic::getDeclaration(&M->getModule(),
                    llvm::Intrinsic::dbg_func_start);

  // Call llvm.dbg.func.start which also implicitly calls llvm.dbg.stoppoint.
  Builder.CreateCall(FuncStartFn, getCastValueFor(Subprogram), "");

  // Push function on region stack.
  RegionStack.push_back(Subprogram);
}


void 
CGDebugInfo::EmitStopPoint(llvm::Function *Fn, llvm::IRBuilder &Builder) 
{
  if (CurLoc.isInvalid() || CurLoc.isMacroID()) return;
  
  // Don't bother if things are the same as last time.
  SourceManager &SM = M->getContext().getSourceManager();
  if (CurLoc == PrevLoc 
       || (SM.getLineNumber(CurLoc) == SM.getLineNumber(PrevLoc)
           && SM.isFromSameFile(CurLoc, PrevLoc)))
    return;

  // Update last state.
  PrevLoc = CurLoc;

  // Get the appropriate compile unit.
  llvm::CompileUnitDesc *Unit = getOrCreateCompileUnit(CurLoc);

  // Lazily construct llvm.dbg.stoppoint function.
  if (!StopPointFn)
    StopPointFn = llvm::Intrinsic::getDeclaration(&M->getModule(), 
                                        llvm::Intrinsic::dbg_stoppoint);

  uint64_t CurLineNo = SM.getLogicalLineNumber(CurLoc);
  uint64_t ColumnNo = SM.getLogicalColumnNumber(CurLoc);

  // Invoke llvm.dbg.stoppoint
  Builder.CreateCall3(StopPointFn, 
                      llvm::ConstantInt::get(llvm::Type::Int32Ty, CurLineNo),
                      llvm::ConstantInt::get(llvm::Type::Int32Ty, ColumnNo),
                      getCastValueFor(Unit), "");
}

/// EmitRegionStart- Constructs the debug code for entering a declarative
/// region - "llvm.dbg.region.start.".
void CGDebugInfo::EmitRegionStart(llvm::Function *Fn, llvm::IRBuilder &Builder) 
{
  llvm::BlockDesc *Block = new llvm::BlockDesc();
  if (RegionStack.size() > 0)
    Block->setContext(RegionStack.back());
  RegionStack.push_back(Block);

  // Lazily construct llvm.dbg.region.start function.
  if (!RegionStartFn)
    RegionStartFn = llvm::Intrinsic::getDeclaration(&M->getModule(), 
                                llvm::Intrinsic::dbg_region_start);

  // Call llvm.dbg.func.start.
  Builder.CreateCall(RegionStartFn, getCastValueFor(Block), "");
}

/// EmitRegionEnd - Constructs the debug code for exiting a declarative
/// region - "llvm.dbg.region.end."
void CGDebugInfo::EmitRegionEnd(llvm::Function *Fn, llvm::IRBuilder &Builder) 
{
  // Lazily construct llvm.dbg.region.end function.
  if (!RegionEndFn)
    RegionEndFn =llvm::Intrinsic::getDeclaration(&M->getModule(), 
                                llvm::Intrinsic::dbg_region_end);

  // Provide an region stop point.
  EmitStopPoint(Fn, Builder);
  
  // Call llvm.dbg.func.end.
  llvm::DebugInfoDesc *DID = RegionStack.back();
  Builder.CreateCall(RegionEndFn, getCastValueFor(DID), "");
  RegionStack.pop_back();
}

/// EmitDeclare - Emit local variable declaration debug info.
void CGDebugInfo::EmitDeclare(const VarDecl *decl, unsigned Tag,
                              llvm::Value *AI,
                              llvm::IRBuilder &Builder)
{
  // FIXME: If it is a compiler generated temporary then return.

  // Construct llvm.dbg.declare function.
  if (!DeclareFn)
    DeclareFn = llvm::Intrinsic::getDeclaration(&M->getModule(), 
                        llvm::Intrinsic::dbg_declare);

  // Get type information.
  llvm::CompileUnitDesc *Unit = getOrCreateCompileUnit(CurLoc);
  llvm::TypeDesc *TyDesc = getOrCreateType(decl->getType(), Unit);

  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Loc = SM.getLogicalLineNumber(CurLoc);

  // Construct variable.
  llvm::VariableDesc *Variable = new llvm::VariableDesc(Tag);
  Variable->setContext(RegionStack.back());
  Variable->setName(decl->getName());
  Variable->setFile(Unit);
  Variable->setLine(Loc);
  Variable->setType(TyDesc);

  // Push it onto the list so that we can free it.
  VariableDescList.push_back(Variable);

  // Cast the AllocA result to a {}* for the call to llvm.dbg.declare.
  // These bit cast instructions will get freed when the basic block is
  // deleted. So do not need to free them explicity.
  const llvm::PointerType *EmpPtr = SR->getEmptyStructPtrType();
  llvm::Value *AllocACast =  new llvm::BitCastInst(AI, EmpPtr, decl->getName(),
                               Builder.GetInsertBlock());

  // Call llvm.dbg.declare.
  Builder.CreateCall2(DeclareFn, AllocACast, getCastValueFor(Variable), "");
}

/// EmitGlobalVariable - Emit information about a global variable.
void CGDebugInfo::EmitGlobalVariable(llvm::GlobalVariable *GV, 
                                     const VarDecl *decl)
{
  // Create global variable debug descriptor.
  llvm::GlobalVariableDesc *Global = new llvm::GlobalVariableDesc();

  // Push it onto the list so that we can free it.
  GlobalVarDescList.push_back(Global);

  // Make sure we have an anchor.
  if (!GlobalVariableAnchor)
    GlobalVariableAnchor = new llvm::AnchorDesc(Global);

  // Get name information.
  Global->setName(decl->getName());
  Global->setFullName(decl->getName());

  llvm::CompileUnitDesc *Unit = getOrCreateCompileUnit(CurLoc);
  SourceManager &SM = M->getContext().getSourceManager();
  uint64_t Loc = SM.getLogicalLineNumber(CurLoc);

  llvm::TypeDesc *TyD = getOrCreateType(decl->getType(), Unit);

  // Fill in the Global information.
  Global->setAnchor(GlobalVariableAnchor);
  Global->setContext(Unit);
  Global->setFile(Unit);
  Global->setLine(Loc);
  Global->setType(TyD);
  Global->setIsDefinition(true);
  Global->setIsStatic(GV->hasInternalLinkage());
  Global->setGlobalVariable(GV);

  // Make sure global is created if needed.
  getValueFor(Global);
}

