//===--- DebugInfo.cpp - Debug Information Helper Classes -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the helper classes used to build and interpret debug
// information in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/Support/Streams.h"

using namespace llvm;
using namespace llvm::dwarf;

//===----------------------------------------------------------------------===//
// DIDescriptor
//===----------------------------------------------------------------------===//

/// ValidDebugInfo - Return true if V represents valid debug info value.
bool DIDescriptor::ValidDebugInfo(Value *V, CodeGenOpt::Level OptLevel) {
  if (!V)
    return false;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(V->stripPointerCasts());
  if (!GV)
    return false;

  if (!GV->hasInternalLinkage () && !GV->hasLinkOnceLinkage())
    return false;

  DIDescriptor DI(GV);

  // Check current version. Allow Version6 for now.
  unsigned Version = DI.getVersion();
  if (Version != LLVMDebugVersion && Version != LLVMDebugVersion6)
    return false;

  unsigned Tag = DI.getTag();
  switch (Tag) {
  case DW_TAG_variable:
    assert(DIVariable(GV).Verify() && "Invalid DebugInfo value");
    break;
  case DW_TAG_compile_unit:
    assert(DICompileUnit(GV).Verify() && "Invalid DebugInfo value");
    break;
  case DW_TAG_subprogram:
    assert(DISubprogram(GV).Verify() && "Invalid DebugInfo value");
    break;
  case DW_TAG_lexical_block:
    // FIXME: This interfers with the quality of generated code during
    // optimization.
    if (OptLevel != CodeGenOpt::None)
      return false;
    // FALLTHROUGH
  default:
    break;
  }

  return true;
}

DIDescriptor::DIDescriptor(GlobalVariable *GV, unsigned RequiredTag) {
  DbgGV = GV;
  
  // If this is non-null, check to see if the Tag matches. If not, set to null.
  if (GV && getTag() != RequiredTag)
    DbgGV = 0;
}

const std::string &
DIDescriptor::getStringField(unsigned Elt, std::string &Result) const {
  if (DbgGV == 0) {
    Result.clear();
    return Result;
  }

  Constant *C = DbgGV->getInitializer();
  if (C == 0 || Elt >= C->getNumOperands()) {
    Result.clear();
    return Result;
  }

  // Fills in the string if it succeeds
  if (!GetConstantStringInfo(C->getOperand(Elt), Result))
    Result.clear();

  return Result;
}

uint64_t DIDescriptor::getUInt64Field(unsigned Elt) const {
  if (DbgGV == 0) return 0;
  if (!DbgGV->hasInitializer()) return 0;

  Constant *C = DbgGV->getInitializer();
  if (C == 0 || Elt >= C->getNumOperands())
    return 0;

  if (ConstantInt *CI = dyn_cast<ConstantInt>(C->getOperand(Elt)))
    return CI->getZExtValue();
  return 0;
}

DIDescriptor DIDescriptor::getDescriptorField(unsigned Elt) const {
  if (DbgGV == 0) return DIDescriptor();

  Constant *C = DbgGV->getInitializer();
  if (C == 0 || Elt >= C->getNumOperands())
    return DIDescriptor();

  C = C->getOperand(Elt);
  return DIDescriptor(dyn_cast<GlobalVariable>(C->stripPointerCasts()));
}

GlobalVariable *DIDescriptor::getGlobalVariableField(unsigned Elt) const {
  if (DbgGV == 0) return 0;

  Constant *C = DbgGV->getInitializer();
  if (C == 0 || Elt >= C->getNumOperands())
    return 0;

  C = C->getOperand(Elt);
  return dyn_cast<GlobalVariable>(C->stripPointerCasts());
}

//===----------------------------------------------------------------------===//
// Simple Descriptor Constructors and other Methods
//===----------------------------------------------------------------------===//

// Needed by DIVariable::getType().
DIType::DIType(GlobalVariable *GV) : DIDescriptor(GV) {
  if (!GV) return;
  unsigned tag = getTag();
  if (tag != dwarf::DW_TAG_base_type && !DIDerivedType::isDerivedType(tag) &&
      !DICompositeType::isCompositeType(tag))
    DbgGV = 0;
}

/// isDerivedType - Return true if the specified tag is legal for
/// DIDerivedType.
bool DIType::isDerivedType(unsigned Tag) {
  switch (Tag) {
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_pointer_type:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_member:
  case dwarf::DW_TAG_inheritance:
    return true;
  default:
    // FIXME: Even though it doesn't make sense, CompositeTypes are current
    // modelled as DerivedTypes, this should return true for them as well.
    return false;
  }
}

/// isCompositeType - Return true if the specified tag is legal for
/// DICompositeType.
bool DIType::isCompositeType(unsigned TAG) {
  switch (TAG) {
  case dwarf::DW_TAG_array_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_vector_type:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_class_type:
    return true;
  default:
    return false;
  }
}

/// isVariable - Return true if the specified tag is legal for DIVariable.
bool DIVariable::isVariable(unsigned Tag) {
  switch (Tag) {
  case dwarf::DW_TAG_auto_variable:
  case dwarf::DW_TAG_arg_variable:
  case dwarf::DW_TAG_return_variable:
    return true;
  default:
    return false;
  }
}

unsigned DIArray::getNumElements() const {
  assert (DbgGV && "Invalid DIArray");
  Constant *C = DbgGV->getInitializer();
  assert (C && "Invalid DIArray initializer");
  return C->getNumOperands();
}

/// replaceAllUsesWith - Replace all uses of debug info referenced by
/// this descriptor. After this completes, the current debug info value
/// is erased.
void DIDerivedType::replaceAllUsesWith(DIDescriptor &D) {
  if (isNull())
    return;

  assert (!D.isNull() && "Can not replace with null");
  getGV()->replaceAllUsesWith(D.getGV());
  getGV()->eraseFromParent();
}

/// Verify - Verify that a compile unit is well formed.
bool DICompileUnit::Verify() const {
  if (isNull()) 
    return false;
  std::string Res;
  if (getFilename(Res).empty()) 
    return false;
  // It is possible that directory and produce string is empty.
  return true;
}

/// Verify - Verify that a type descriptor is well formed.
bool DIType::Verify() const {
  if (isNull()) 
    return false;
  if (getContext().isNull()) 
    return false;

  DICompileUnit CU = getCompileUnit();
  if (!CU.isNull() && !CU.Verify()) 
    return false;
  return true;
}

/// Verify - Verify that a composite type descriptor is well formed.
bool DICompositeType::Verify() const {
  if (isNull()) 
    return false;
  if (getContext().isNull()) 
    return false;

  DICompileUnit CU = getCompileUnit();
  if (!CU.isNull() && !CU.Verify()) 
    return false;
  return true;
}

/// Verify - Verify that a subprogram descriptor is well formed.
bool DISubprogram::Verify() const {
  if (isNull())
    return false;
  
  if (getContext().isNull())
    return false;

  DICompileUnit CU = getCompileUnit();
  if (!CU.Verify()) 
    return false;

  DICompositeType Ty = getType();
  if (!Ty.isNull() && !Ty.Verify())
    return false;
  return true;
}

/// Verify - Verify that a global variable descriptor is well formed.
bool DIGlobalVariable::Verify() const {
  if (isNull())
    return false;
  
  if (getContext().isNull())
    return false;

  DICompileUnit CU = getCompileUnit();
  if (!CU.isNull() && !CU.Verify()) 
    return false;

  DIType Ty = getType();
  if (!Ty.Verify())
    return false;

  if (!getGlobal())
    return false;

  return true;
}

/// Verify - Verify that a variable descriptor is well formed.
bool DIVariable::Verify() const {
  if (isNull())
    return false;
  
  if (getContext().isNull())
    return false;

  DIType Ty = getType();
  if (!Ty.Verify())
    return false;

  return true;
}

/// getOriginalTypeSize - If this type is derived from a base type then
/// return base type size.
uint64_t DIDerivedType::getOriginalTypeSize() const {
  if (getTag() != dwarf::DW_TAG_member)
    return getSizeInBits();
  DIType BT = getTypeDerivedFrom();
  if (BT.getTag() != dwarf::DW_TAG_base_type)
    return getSizeInBits();
  return BT.getSizeInBits();
}

/// describes - Return true if this subprogram provides debugging
/// information for the function F.
bool DISubprogram::describes(const Function *F) {
  assert (F && "Invalid function");
  std::string Name;
  getLinkageName(Name);
  if (Name.empty())
    getName(Name);
  if (F->getName() == Name)
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// DIDescriptor: dump routines for all descriptors.
//===----------------------------------------------------------------------===//


/// dump - Print descriptor.
void DIDescriptor::dump() const {
  cerr << "[" << dwarf::TagString(getTag()) << "] ";
  cerr << std::hex << "[GV:" << DbgGV << "]" << std::dec;
}

/// dump - Print compile unit.
void DICompileUnit::dump() const {
  if (getLanguage())
    cerr << " [" << dwarf::LanguageString(getLanguage()) << "] ";

  std::string Res1, Res2;
  cerr << " [" << getDirectory(Res1) << "/" << getFilename(Res2) << " ]";
}

/// dump - Print type.
void DIType::dump() const {
  if (isNull()) return;

  std::string Res;
  if (!getName(Res).empty())
    cerr << " [" << Res << "] ";

  unsigned Tag = getTag();
  cerr << " [" << dwarf::TagString(Tag) << "] ";

  // TODO : Print context
  getCompileUnit().dump();
  cerr << " [" 
       << getLineNumber() << ", " 
       << getSizeInBits() << ", "
       << getAlignInBits() << ", "
       << getOffsetInBits() 
       << "] ";

  if (isPrivate()) 
    cerr << " [private] ";
  else if (isProtected())
    cerr << " [protected] ";

  if (isForwardDecl())
    cerr << " [fwd] ";

  if (isBasicType(Tag))
    DIBasicType(DbgGV).dump();
  else if (isDerivedType(Tag))
    DIDerivedType(DbgGV).dump();
  else if (isCompositeType(Tag))
    DICompositeType(DbgGV).dump();
  else {
    cerr << "Invalid DIType\n";
    return;
  }

  cerr << "\n";
}

/// dump - Print basic type.
void DIBasicType::dump() const {
  cerr << " [" << dwarf::AttributeEncodingString(getEncoding()) << "] ";
}

/// dump - Print derived type.
void DIDerivedType::dump() const {
  cerr << "\n\t Derived From: "; getTypeDerivedFrom().dump();
}

/// dump - Print composite type.
void DICompositeType::dump() const {
  DIArray A = getTypeArray();
  if (A.isNull())
    return;
  cerr << " [" << A.getNumElements() << " elements]";
}

/// dump - Print global.
void DIGlobal::dump() const {
  std::string Res;
  if (!getName(Res).empty())
    cerr << " [" << Res << "] ";

  unsigned Tag = getTag();
  cerr << " [" << dwarf::TagString(Tag) << "] ";

  // TODO : Print context
  getCompileUnit().dump();
  cerr << " [" << getLineNumber() << "] ";

  if (isLocalToUnit())
    cerr << " [local] ";

  if (isDefinition())
    cerr << " [def] ";

  if (isGlobalVariable(Tag))
    DIGlobalVariable(DbgGV).dump();

  cerr << "\n";
}

/// dump - Print subprogram.
void DISubprogram::dump() const {
  DIGlobal::dump();
}

/// dump - Print global variable.
void DIGlobalVariable::dump() const {
  cerr << " ["; getGlobal()->dump(); cerr << "] ";
}

/// dump - Print variable.
void DIVariable::dump() const {
  std::string Res;
  if (!getName(Res).empty())
    cerr << " [" << Res << "] ";

  getCompileUnit().dump();
  cerr << " [" << getLineNumber() << "] ";
  getType().dump();
  cerr << "\n";
}

//===----------------------------------------------------------------------===//
// DIFactory: Basic Helpers
//===----------------------------------------------------------------------===//

DIFactory::DIFactory(Module &m)
  : M(m), VMContext(M.getContext()), StopPointFn(0), FuncStartFn(0), 
    RegionStartFn(0), RegionEndFn(0),
    DeclareFn(0) {
  EmptyStructPtr = PointerType::getUnqual(StructType::get(VMContext));
}

/// getCastToEmpty - Return this descriptor as a Constant* with type '{}*'.
/// This is only valid when the descriptor is non-null.
Constant *DIFactory::getCastToEmpty(DIDescriptor D) {
  if (D.isNull()) return llvm::Constant::getNullValue(EmptyStructPtr);
  return ConstantExpr::getBitCast(D.getGV(), EmptyStructPtr);
}

Constant *DIFactory::GetTagConstant(unsigned TAG) {
  assert((TAG & LLVMDebugVersionMask) == 0 &&
         "Tag too large for debug encoding!");
  return ConstantInt::get(Type::Int32Ty, TAG | LLVMDebugVersion);
}

Constant *DIFactory::GetStringConstant(const std::string &String) {
  // Check string cache for previous edition.
  Constant *&Slot = StringCache[String];
  
  // Return Constant if previously defined.
  if (Slot) return Slot;
  
  const PointerType *DestTy = PointerType::getUnqual(Type::Int8Ty);
  
  // If empty string then use a i8* null instead.
  if (String.empty())
    return Slot = ConstantPointerNull::get(DestTy);

  // Construct string as an llvm constant.
  Constant *ConstStr = ConstantArray::get(String);
    
  // Otherwise create and return a new string global.
  GlobalVariable *StrGV = new GlobalVariable(M, ConstStr->getType(), true,
                                             GlobalVariable::InternalLinkage,
                                             ConstStr, ".str");
  StrGV->setSection("llvm.metadata");
  return Slot = ConstantExpr::getBitCast(StrGV, DestTy);
}

//===----------------------------------------------------------------------===//
// DIFactory: Primary Constructors
//===----------------------------------------------------------------------===//

/// GetOrCreateArray - Create an descriptor for an array of descriptors. 
/// This implicitly uniques the arrays created.
DIArray DIFactory::GetOrCreateArray(DIDescriptor *Tys, unsigned NumTys) {
  SmallVector<Constant*, 16> Elts;
  
  for (unsigned i = 0; i != NumTys; ++i)
    Elts.push_back(getCastToEmpty(Tys[i]));
  
  Constant *Init = ConstantArray::get(ArrayType::get(EmptyStructPtr,
                                                     Elts.size()),
                                      Elts.data(), Elts.size());
  // If we already have this array, just return the uniqued version.
  DIDescriptor &Entry = SimpleConstantCache[Init];
  if (!Entry.isNull()) return DIArray(Entry.getGV());

  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.array");
  GV->setSection("llvm.metadata");
  Entry = DIDescriptor(GV);
  return DIArray(GV);
}

/// GetOrCreateSubrange - Create a descriptor for a value range.  This
/// implicitly uniques the values returned.
DISubrange DIFactory::GetOrCreateSubrange(int64_t Lo, int64_t Hi) {
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_subrange_type),
    ConstantInt::get(Type::Int64Ty, Lo),
    ConstantInt::get(Type::Int64Ty, Hi)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));

  // If we already have this range, just return the uniqued version.
  DIDescriptor &Entry = SimpleConstantCache[Init];
  if (!Entry.isNull()) return DISubrange(Entry.getGV());
  
  M.addTypeName("llvm.dbg.subrange.type", Init->getType());

  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.subrange");
  GV->setSection("llvm.metadata");
  Entry = DIDescriptor(GV);
  return DISubrange(GV);
}



/// CreateCompileUnit - Create a new descriptor for the specified compile
/// unit.  Note that this does not unique compile units within the module.
DICompileUnit DIFactory::CreateCompileUnit(unsigned LangID,
                                           const std::string &Filename,
                                           const std::string &Directory,
                                           const std::string &Producer,
                                           bool isMain,
                                           bool isOptimized,
                                           const char *Flags,
                                           unsigned RunTimeVer) {
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_compile_unit),
    llvm::Constant::getNullValue(EmptyStructPtr),
    ConstantInt::get(Type::Int32Ty, LangID),
    GetStringConstant(Filename),
    GetStringConstant(Directory),
    GetStringConstant(Producer),
    ConstantInt::get(Type::Int1Ty, isMain),
    ConstantInt::get(Type::Int1Ty, isOptimized),
    GetStringConstant(Flags),
    ConstantInt::get(Type::Int32Ty, RunTimeVer)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.compile_unit.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::LinkOnceAnyLinkage,
                                          Init, "llvm.dbg.compile_unit");
  GV->setSection("llvm.metadata");
  return DICompileUnit(GV);
}

/// CreateEnumerator - Create a single enumerator value.
DIEnumerator DIFactory::CreateEnumerator(const std::string &Name, uint64_t Val){
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_enumerator),
    GetStringConstant(Name),
    ConstantInt::get(Type::Int64Ty, Val)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.enumerator.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.enumerator");
  GV->setSection("llvm.metadata");
  return DIEnumerator(GV);
}


/// CreateBasicType - Create a basic type like int, float, etc.
DIBasicType DIFactory::CreateBasicType(DIDescriptor Context,
                                      const std::string &Name,
                                       DICompileUnit CompileUnit,
                                       unsigned LineNumber,
                                       uint64_t SizeInBits,
                                       uint64_t AlignInBits,
                                       uint64_t OffsetInBits, unsigned Flags,
                                       unsigned Encoding) {
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_base_type),
    getCastToEmpty(Context),
    GetStringConstant(Name),
    getCastToEmpty(CompileUnit),
    ConstantInt::get(Type::Int32Ty, LineNumber),
    ConstantInt::get(Type::Int64Ty, SizeInBits),
    ConstantInt::get(Type::Int64Ty, AlignInBits),
    ConstantInt::get(Type::Int64Ty, OffsetInBits),
    ConstantInt::get(Type::Int32Ty, Flags),
    ConstantInt::get(Type::Int32Ty, Encoding)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.basictype.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.basictype");
  GV->setSection("llvm.metadata");
  return DIBasicType(GV);
}

/// CreateDerivedType - Create a derived type like const qualified type,
/// pointer, typedef, etc.
DIDerivedType DIFactory::CreateDerivedType(unsigned Tag,
                                           DIDescriptor Context,
                                           const std::string &Name,
                                           DICompileUnit CompileUnit,
                                           unsigned LineNumber,
                                           uint64_t SizeInBits,
                                           uint64_t AlignInBits,
                                           uint64_t OffsetInBits,
                                           unsigned Flags,
                                           DIType DerivedFrom) {
  Constant *Elts[] = {
    GetTagConstant(Tag),
    getCastToEmpty(Context),
    GetStringConstant(Name),
    getCastToEmpty(CompileUnit),
    ConstantInt::get(Type::Int32Ty, LineNumber),
    ConstantInt::get(Type::Int64Ty, SizeInBits),
    ConstantInt::get(Type::Int64Ty, AlignInBits),
    ConstantInt::get(Type::Int64Ty, OffsetInBits),
    ConstantInt::get(Type::Int32Ty, Flags),
    getCastToEmpty(DerivedFrom)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.derivedtype.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.derivedtype");
  GV->setSection("llvm.metadata");
  return DIDerivedType(GV);
}

/// CreateCompositeType - Create a composite type like array, struct, etc.
DICompositeType DIFactory::CreateCompositeType(unsigned Tag,
                                               DIDescriptor Context,
                                               const std::string &Name,
                                               DICompileUnit CompileUnit,
                                               unsigned LineNumber,
                                               uint64_t SizeInBits,
                                               uint64_t AlignInBits,
                                               uint64_t OffsetInBits,
                                               unsigned Flags,
                                               DIType DerivedFrom,
                                               DIArray Elements,
                                               unsigned RuntimeLang) {

  Constant *Elts[] = {
    GetTagConstant(Tag),
    getCastToEmpty(Context),
    GetStringConstant(Name),
    getCastToEmpty(CompileUnit),
    ConstantInt::get(Type::Int32Ty, LineNumber),
    ConstantInt::get(Type::Int64Ty, SizeInBits),
    ConstantInt::get(Type::Int64Ty, AlignInBits),
    ConstantInt::get(Type::Int64Ty, OffsetInBits),
    ConstantInt::get(Type::Int32Ty, Flags),
    getCastToEmpty(DerivedFrom),
    getCastToEmpty(Elements),
    ConstantInt::get(Type::Int32Ty, RuntimeLang)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.composite.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.composite");
  GV->setSection("llvm.metadata");
  return DICompositeType(GV);
}


/// CreateSubprogram - Create a new descriptor for the specified subprogram.
/// See comments in DISubprogram for descriptions of these fields.  This
/// method does not unique the generated descriptors.
DISubprogram DIFactory::CreateSubprogram(DIDescriptor Context, 
                                         const std::string &Name,
                                         const std::string &DisplayName,
                                         const std::string &LinkageName,
                                         DICompileUnit CompileUnit,
                                         unsigned LineNo, DIType Type,
                                         bool isLocalToUnit,
                                         bool isDefinition) {

  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_subprogram),
    llvm::Constant::getNullValue(EmptyStructPtr),
    getCastToEmpty(Context),
    GetStringConstant(Name),
    GetStringConstant(DisplayName),
    GetStringConstant(LinkageName),
    getCastToEmpty(CompileUnit),
    ConstantInt::get(Type::Int32Ty, LineNo),
    getCastToEmpty(Type),
    ConstantInt::get(Type::Int1Ty, isLocalToUnit),
    ConstantInt::get(Type::Int1Ty, isDefinition)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.subprogram.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::LinkOnceAnyLinkage,
                                          Init, "llvm.dbg.subprogram");
  GV->setSection("llvm.metadata");
  return DISubprogram(GV);
}

/// CreateGlobalVariable - Create a new descriptor for the specified global.
DIGlobalVariable
DIFactory::CreateGlobalVariable(DIDescriptor Context, const std::string &Name,
                                const std::string &DisplayName,
                                const std::string &LinkageName,
                                DICompileUnit CompileUnit,
                                unsigned LineNo, DIType Type,bool isLocalToUnit,
                                bool isDefinition, llvm::GlobalVariable *Val) {
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_variable),
    llvm::Constant::getNullValue(EmptyStructPtr),
    getCastToEmpty(Context),
    GetStringConstant(Name),
    GetStringConstant(DisplayName),
    GetStringConstant(LinkageName),
    getCastToEmpty(CompileUnit),
    ConstantInt::get(Type::Int32Ty, LineNo),
    getCastToEmpty(Type),
    ConstantInt::get(Type::Int1Ty, isLocalToUnit),
    ConstantInt::get(Type::Int1Ty, isDefinition),
    ConstantExpr::getBitCast(Val, EmptyStructPtr)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.global_variable.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::LinkOnceAnyLinkage,
                                          Init, "llvm.dbg.global_variable");
  GV->setSection("llvm.metadata");
  return DIGlobalVariable(GV);
}


/// CreateVariable - Create a new descriptor for the specified variable.
DIVariable DIFactory::CreateVariable(unsigned Tag, DIDescriptor Context,
                                     const std::string &Name,
                                     DICompileUnit CompileUnit, unsigned LineNo,
                                     DIType Type) {
  Constant *Elts[] = {
    GetTagConstant(Tag),
    getCastToEmpty(Context),
    GetStringConstant(Name),
    getCastToEmpty(CompileUnit),
    ConstantInt::get(Type::Int32Ty, LineNo),
    getCastToEmpty(Type)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.variable.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.variable");
  GV->setSection("llvm.metadata");
  return DIVariable(GV);
}


/// CreateBlock - This creates a descriptor for a lexical block with the
/// specified parent VMContext.
DIBlock DIFactory::CreateBlock(DIDescriptor Context) {
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_lexical_block),
    getCastToEmpty(Context)
  };
  
  Constant *Init = ConstantStruct::get(VMContext, Elts,
                                       sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.block.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(M, Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.block");
  GV->setSection("llvm.metadata");
  return DIBlock(GV);
}


//===----------------------------------------------------------------------===//
// DIFactory: Routines for inserting code into a function
//===----------------------------------------------------------------------===//

/// InsertStopPoint - Create a new llvm.dbg.stoppoint intrinsic invocation,
/// inserting it at the end of the specified basic block.
void DIFactory::InsertStopPoint(DICompileUnit CU, unsigned LineNo,
                                unsigned ColNo, BasicBlock *BB) {
  
  // Lazily construct llvm.dbg.stoppoint function.
  if (!StopPointFn)
    StopPointFn = llvm::Intrinsic::getDeclaration(&M, 
                                              llvm::Intrinsic::dbg_stoppoint);
  
  // Invoke llvm.dbg.stoppoint
  Value *Args[] = {
    ConstantInt::get(llvm::Type::Int32Ty, LineNo),
    ConstantInt::get(llvm::Type::Int32Ty, ColNo),
    getCastToEmpty(CU)
  };
  CallInst::Create(StopPointFn, Args, Args+3, "", BB);
}

/// InsertSubprogramStart - Create a new llvm.dbg.func.start intrinsic to
/// mark the start of the specified subprogram.
void DIFactory::InsertSubprogramStart(DISubprogram SP, BasicBlock *BB) {
  // Lazily construct llvm.dbg.func.start.
  if (!FuncStartFn)
    FuncStartFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_func_start);
  
  // Call llvm.dbg.func.start which also implicitly sets a stoppoint.
  CallInst::Create(FuncStartFn, getCastToEmpty(SP), "", BB);
}

/// InsertRegionStart - Insert a new llvm.dbg.region.start intrinsic call to
/// mark the start of a region for the specified scoping descriptor.
void DIFactory::InsertRegionStart(DIDescriptor D, BasicBlock *BB) {
  // Lazily construct llvm.dbg.region.start function.
  if (!RegionStartFn)
    RegionStartFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_region_start);

  // Call llvm.dbg.func.start.
  CallInst::Create(RegionStartFn, getCastToEmpty(D), "", BB);
}

/// InsertRegionEnd - Insert a new llvm.dbg.region.end intrinsic call to
/// mark the end of a region for the specified scoping descriptor.
void DIFactory::InsertRegionEnd(DIDescriptor D, BasicBlock *BB) {
  // Lazily construct llvm.dbg.region.end function.
  if (!RegionEndFn)
    RegionEndFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_region_end);

  // Call llvm.dbg.region.end.
  CallInst::Create(RegionEndFn, getCastToEmpty(D), "", BB);
}

/// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
void DIFactory::InsertDeclare(Value *Storage, DIVariable D, BasicBlock *BB) {
  // Cast the storage to a {}* for the call to llvm.dbg.declare.
  Storage = new BitCastInst(Storage, EmptyStructPtr, "", BB);
  
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  Value *Args[] = { Storage, getCastToEmpty(D) };
  CallInst::Create(DeclareFn, Args, Args+2, "", BB);
}

//===----------------------------------------------------------------------===//
// DebugInfoFinder implementations.
//===----------------------------------------------------------------------===//

/// processModule - Process entire module and collect debug info.
void DebugInfoFinder::processModule(Module &M) {
  
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    for (Function::iterator FI = (*I).begin(), FE = (*I).end(); FI != FE; ++FI)
      for (BasicBlock::iterator BI = (*FI).begin(), BE = (*FI).end(); BI != BE;
           ++BI) {
        if (DbgStopPointInst *SPI = dyn_cast<DbgStopPointInst>(BI))
          processStopPoint(SPI);
        else if (DbgFuncStartInst *FSI = dyn_cast<DbgFuncStartInst>(BI))
          processFuncStart(FSI);
        else if (DbgRegionStartInst *DRS = dyn_cast<DbgRegionStartInst>(BI))
          processRegionStart(DRS);
        else if (DbgRegionEndInst *DRE = dyn_cast<DbgRegionEndInst>(BI))
          processRegionEnd(DRE);
        else if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(BI))
          processDeclare(DDI);
      }
  
  for (Module::global_iterator GVI = M.global_begin(), GVE = M.global_end();
       GVI != GVE; ++GVI) {
    GlobalVariable *GV = GVI;
    if (!GV->hasName() || !GV->isConstant() 
        || strcmp(GV->getName().data(), "llvm.dbg.global_variable")
        || !GV->hasInitializer())
      continue;
    DIGlobalVariable DIG(GV);
    if (addGlobalVariable(DIG)) {
      addCompileUnit(DIG.getCompileUnit());
      processType(DIG.getType());
    }
  }
}
    
/// processType - Process DIType.
void DebugInfoFinder::processType(DIType DT) {
  if (DT.isNull())
    return;
  if (!NodesSeen.insert(DT.getGV()))
    return;

  addCompileUnit(DT.getCompileUnit());
  if (DT.isCompositeType(DT.getTag())) {
    DICompositeType DCT(DT.getGV());
    processType(DCT.getTypeDerivedFrom());
    DIArray DA = DCT.getTypeArray();
    if (!DA.isNull())
      for (unsigned i = 0, e = DA.getNumElements(); i != e; ++i) {
        DIDescriptor D = DA.getElement(i);
        DIType TypeE = DIType(D.getGV());
        if (!TypeE.isNull())
          processType(TypeE);
        else 
          processSubprogram(DISubprogram(D.getGV()));
      }
  } else if (DT.isDerivedType(DT.getTag())) {
    DIDerivedType DDT(DT.getGV());
    if (!DDT.isNull()) 
      processType(DDT.getTypeDerivedFrom());
  }
}

/// processSubprogram - Process DISubprogram.
void DebugInfoFinder::processSubprogram(DISubprogram SP) {
  if (SP.isNull())
    return;
  if (!addSubprogram(SP))
    return;
  addCompileUnit(SP.getCompileUnit());
  processType(SP.getType());
}

/// processStopPoint - Process DbgStopPointInst.
void DebugInfoFinder::processStopPoint(DbgStopPointInst *SPI) {
  GlobalVariable *Context = dyn_cast<GlobalVariable>(SPI->getContext());
  addCompileUnit(DICompileUnit(Context));
}

/// processFuncStart - Process DbgFuncStartInst.
void DebugInfoFinder::processFuncStart(DbgFuncStartInst *FSI) {
  GlobalVariable *SP = dyn_cast<GlobalVariable>(FSI->getSubprogram());
  processSubprogram(DISubprogram(SP));
}

/// processRegionStart - Process DbgRegionStart.
void DebugInfoFinder::processRegionStart(DbgRegionStartInst *DRS) {
  GlobalVariable *SP = dyn_cast<GlobalVariable>(DRS->getContext());
  processSubprogram(DISubprogram(SP));
}

/// processRegionEnd - Process DbgRegionEnd.
void DebugInfoFinder::processRegionEnd(DbgRegionEndInst *DRE) {
  GlobalVariable *SP = dyn_cast<GlobalVariable>(DRE->getContext());
  processSubprogram(DISubprogram(SP));
}

/// processDeclare - Process DbgDeclareInst.
void DebugInfoFinder::processDeclare(DbgDeclareInst *DDI) {
  DIVariable DV(cast<GlobalVariable>(DDI->getVariable()));
  if (DV.isNull())
    return;

  if (!NodesSeen.insert(DV.getGV()))
    return;

  addCompileUnit(DV.getCompileUnit());
  processType(DV.getType());
}

/// addCompileUnit - Add compile unit into CUs.
bool DebugInfoFinder::addCompileUnit(DICompileUnit CU) {
  if (CU.isNull())
    return false;

  if (!NodesSeen.insert(CU.getGV()))
    return false;

  CUs.push_back(CU.getGV());
  return true;
}
    
/// addGlobalVariable - Add global variable into GVs.
bool DebugInfoFinder::addGlobalVariable(DIGlobalVariable DIG) {
  if (DIG.isNull())
    return false;

  if (!NodesSeen.insert(DIG.getGV()))
    return false;

  GVs.push_back(DIG.getGV());
  return true;
}

// addSubprogram - Add subprgoram into SPs.
bool DebugInfoFinder::addSubprogram(DISubprogram SP) {
  if (SP.isNull())
    return false;
  
  if (!NodesSeen.insert(SP.getGV()))
    return false;

  SPs.push_back(SP.getGV());
  return true;
}

namespace llvm {
  /// findStopPoint - Find the stoppoint coressponding to this instruction, that
  /// is the stoppoint that dominates this instruction.
  const DbgStopPointInst *findStopPoint(const Instruction *Inst) {
    if (const DbgStopPointInst *DSI = dyn_cast<DbgStopPointInst>(Inst))
      return DSI;

    const BasicBlock *BB = Inst->getParent();
    BasicBlock::const_iterator I = Inst, B;
    while (BB) {
      B = BB->begin();

      // A BB consisting only of a terminator can't have a stoppoint.
      while (I != B) {
        --I;
        if (const DbgStopPointInst *DSI = dyn_cast<DbgStopPointInst>(I))
          return DSI;
      }

      // This BB didn't have a stoppoint: if there is only one predecessor, look
      // for a stoppoint there. We could use getIDom(), but that would require
      // dominator info.
      BB = I->getParent()->getUniquePredecessor();
      if (BB)
        I = BB->getTerminator();
    }

    return 0;
  }

  /// findBBStopPoint - Find the stoppoint corresponding to first real
  /// (non-debug intrinsic) instruction in this Basic Block, and return the
  /// stoppoint for it.
  const DbgStopPointInst *findBBStopPoint(const BasicBlock *BB) {
    for(BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (const DbgStopPointInst *DSI = dyn_cast<DbgStopPointInst>(I))
        return DSI;

    // Fallback to looking for stoppoint of unique predecessor. Useful if this
    // BB contains no stoppoints, but unique predecessor does.
    BB = BB->getUniquePredecessor();
    if (BB)
      return findStopPoint(BB->getTerminator());

    return 0;
  }

  Value *findDbgGlobalDeclare(GlobalVariable *V) {
    const Module *M = V->getParent();
    
    const Type *Ty = M->getTypeByName("llvm.dbg.global_variable.type");
    if (!Ty) return 0;

    Ty = PointerType::get(Ty, 0);

    Value *Val = V->stripPointerCasts();
    for (Value::use_iterator I = Val->use_begin(), E = Val->use_end();
         I != E; ++I) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(I)) {
        if (CE->getOpcode() == Instruction::BitCast) {
          Value *VV = CE;

          while (VV->hasOneUse())
            VV = *VV->use_begin();

          if (VV->getType() == Ty)
            return VV;
        }
      }
    }
    
    if (Val->getType() == Ty)
      return Val;

    return 0;
  }

  /// Finds the llvm.dbg.declare intrinsic corresponding to this value if any.
  /// It looks through pointer casts too.
  const DbgDeclareInst *findDbgDeclare(const Value *V, bool stripCasts) {
    if (stripCasts) {
      V = V->stripPointerCasts();

      // Look for the bitcast.
      for (Value::use_const_iterator I = V->use_begin(), E =V->use_end();
            I != E; ++I)
        if (isa<BitCastInst>(I))
          return findDbgDeclare(*I, false);

      return 0;
    }

    // Find llvm.dbg.declare among uses of the instruction.
    for (Value::use_const_iterator I = V->use_begin(), E =V->use_end();
          I != E; ++I)
      if (const DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(I))
        return DDI;

    return 0;
  }

  bool getLocationInfo(const Value *V, std::string &DisplayName,
                       std::string &Type, unsigned &LineNo, std::string &File,
                       std::string &Dir) {
    DICompileUnit Unit;
    DIType TypeD;

    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(const_cast<Value*>(V))) {
      Value *DIGV = findDbgGlobalDeclare(GV);
      if (!DIGV) return false;
      DIGlobalVariable Var(cast<GlobalVariable>(DIGV));

      Var.getDisplayName(DisplayName);
      LineNo = Var.getLineNumber();
      Unit = Var.getCompileUnit();
      TypeD = Var.getType();
    } else {
      const DbgDeclareInst *DDI = findDbgDeclare(V);
      if (!DDI) return false;
      DIVariable Var(cast<GlobalVariable>(DDI->getVariable()));

      Var.getName(DisplayName);
      LineNo = Var.getLineNumber();
      Unit = Var.getCompileUnit();
      TypeD = Var.getType();
    }

    TypeD.getName(Type);
    Unit.getFilename(File);
    Unit.getDirectory(Dir);
    return true;
  }

  /// isValidDebugInfoIntrinsic - Return true if SPI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgStopPointInst &SPI, 
                                 CodeGenOpt::Level OptLev) {
    return DIDescriptor::ValidDebugInfo(SPI.getContext(), OptLev);
  }

  /// isValidDebugInfoIntrinsic - Return true if FSI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgFuncStartInst &FSI,
                                 CodeGenOpt::Level OptLev) {
    return DIDescriptor::ValidDebugInfo(FSI.getSubprogram(), OptLev);
  }

  /// isValidDebugInfoIntrinsic - Return true if RSI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgRegionStartInst &RSI,
                                 CodeGenOpt::Level OptLev) {
    return DIDescriptor::ValidDebugInfo(RSI.getContext(), OptLev);
  }

  /// isValidDebugInfoIntrinsic - Return true if REI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgRegionEndInst &REI,
                                 CodeGenOpt::Level OptLev) {
    return DIDescriptor::ValidDebugInfo(REI.getContext(), OptLev);
  }


  /// isValidDebugInfoIntrinsic - Return true if DI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgDeclareInst &DI,
                                 CodeGenOpt::Level OptLev) {
    return DIDescriptor::ValidDebugInfo(DI.getVariable(), OptLev);
  }

  /// ExtractDebugLocation - Extract debug location information 
  /// from llvm.dbg.stoppoint intrinsic.
  DebugLoc ExtractDebugLocation(DbgStopPointInst &SPI,
                                DebugLocTracker &DebugLocInfo) {
    DebugLoc DL;
    Value *Context = SPI.getContext();

    // If this location is already tracked then use it.
    DebugLocTuple Tuple(cast<GlobalVariable>(Context), SPI.getLine(), 
                        SPI.getColumn());
    DenseMap<DebugLocTuple, unsigned>::iterator II
      = DebugLocInfo.DebugIdMap.find(Tuple);
    if (II != DebugLocInfo.DebugIdMap.end())
      return DebugLoc::get(II->second);

    // Add a new location entry.
    unsigned Id = DebugLocInfo.DebugLocations.size();
    DebugLocInfo.DebugLocations.push_back(Tuple);
    DebugLocInfo.DebugIdMap[Tuple] = Id;
    
    return DebugLoc::get(Id);
  }

  /// ExtractDebugLocation - Extract debug location information 
  /// from llvm.dbg.func_start intrinsic.
  DebugLoc ExtractDebugLocation(DbgFuncStartInst &FSI,
                                DebugLocTracker &DebugLocInfo) {
    DebugLoc DL;
    Value *SP = FSI.getSubprogram();

    DISubprogram Subprogram(cast<GlobalVariable>(SP));
    unsigned Line = Subprogram.getLineNumber();
    DICompileUnit CU(Subprogram.getCompileUnit());

    // If this location is already tracked then use it.
    DebugLocTuple Tuple(CU.getGV(), Line, /* Column */ 0);
    DenseMap<DebugLocTuple, unsigned>::iterator II
      = DebugLocInfo.DebugIdMap.find(Tuple);
    if (II != DebugLocInfo.DebugIdMap.end())
      return DebugLoc::get(II->second);

    // Add a new location entry.
    unsigned Id = DebugLocInfo.DebugLocations.size();
    DebugLocInfo.DebugLocations.push_back(Tuple);
    DebugLocInfo.DebugIdMap[Tuple] = Id;
    
    return DebugLoc::get(Id);
  }

  /// isInlinedFnStart - Return true if FSI is starting an inlined function.
  bool isInlinedFnStart(DbgFuncStartInst &FSI, const Function *CurrentFn) {
    DISubprogram Subprogram(cast<GlobalVariable>(FSI.getSubprogram()));
    if (Subprogram.describes(CurrentFn))
      return false;

    return true;
  }

  /// isInlinedFnEnd - Return true if REI is ending an inlined function.
  bool isInlinedFnEnd(DbgRegionEndInst &REI, const Function *CurrentFn) {
    DISubprogram Subprogram(cast<GlobalVariable>(REI.getContext()));
    if (Subprogram.isNull() || Subprogram.describes(CurrentFn))
      return false;

    return true;
  }

}
