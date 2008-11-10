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
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Dwarf.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// DIDescriptor
//===----------------------------------------------------------------------===//

DIDescriptor::DIDescriptor(GlobalVariable *gv, unsigned RequiredTag) {
  GV = gv;
  
  // If this is non-null, check to see if the Tag matches.  If not, set to null.
  if (GV && getTag() != RequiredTag)
    GV = 0;
}


std::string DIDescriptor::getStringField(unsigned Elt) const {
  if (GV == 0) return "";
  Constant *C = GV->getInitializer();
  if (C == 0 || Elt >= C->getNumOperands())
    return "";
  
  std::string Result;
  // Fills in the string if it succeeds
  if (!GetConstantStringInfo(C->getOperand(Elt), Result))
    Result.clear();
  return Result;
}

uint64_t DIDescriptor::getUInt64Field(unsigned Elt) const {
  if (GV == 0) return 0;
  Constant *C = GV->getInitializer();
  if (C == 0 || Elt >= C->getNumOperands())
    return 0;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(C->getOperand(Elt)))
    return CI->getZExtValue();
  return 0;
}


DIDescriptor DIDescriptor::getDescriptorField(unsigned Elt) const {
  if (GV == 0) return DIDescriptor();
  Constant *C = GV->getInitializer();
  if (C == 0 || Elt >= C->getNumOperands())
    return DIDescriptor();
  C = C->getOperand(Elt);
  return DIDescriptor(dyn_cast<GlobalVariable>(C->stripPointerCasts()));
}

GlobalVariable *DIDescriptor::getGlobalVariableField(unsigned Elt) const {
  if (GV == 0) return 0;
  Constant *C = GV->getInitializer();
  if (C == 0 || Elt >= C->getNumOperands())
    return 0;
  C = C->getOperand(Elt);
  
  return dyn_cast<GlobalVariable>(C->stripPointerCasts());
}



/// getCastToEmpty - Return this descriptor as a Constant* with type '{}*'.
/// This is only valid when the descriptor is non-null.
Constant *DIDescriptor::getCastToEmpty() const {
  const Type *DestTy = PointerType::getUnqual(StructType::get(NULL, NULL));
  if (isNull()) return Constant::getNullValue(DestTy);
  return ConstantExpr::getBitCast(GV, DestTy);
}

//===----------------------------------------------------------------------===//
// Simple Descriptor Constructors and other Methods
//===----------------------------------------------------------------------===//

DIAnchor::DIAnchor(GlobalVariable *GV)
  : DIDescriptor(GV, dwarf::DW_TAG_anchor) {}
DIEnumerator::DIEnumerator(GlobalVariable *GV)
  : DIDescriptor(GV, dwarf::DW_TAG_enumerator) {}
DISubrange::DISubrange(GlobalVariable *GV)
  : DIDescriptor(GV, dwarf::DW_TAG_subrange_type) {}
DICompileUnit::DICompileUnit(GlobalVariable *GV)
  : DIDescriptor(GV, dwarf::DW_TAG_compile_unit) {}
DIBasicType::DIBasicType(GlobalVariable *GV)
  : DIType(GV, dwarf::DW_TAG_base_type) {}
DISubprogram::DISubprogram(GlobalVariable *GV)
  : DIGlobal(GV, dwarf::DW_TAG_subprogram) {}
DIGlobalVariable::DIGlobalVariable(GlobalVariable *GV)
  : DIGlobal(GV, dwarf::DW_TAG_variable) {}
DIBlock::DIBlock(GlobalVariable *GV)
  : DIDescriptor(GV, dwarf::DW_TAG_lexical_block) {}

/// isDerivedType - Return true if the specified tag is legal for
/// DIDerivedType.
bool DIDerivedType::isDerivedType(unsigned Tag) {
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

DIDerivedType::DIDerivedType(GlobalVariable *GV) : DIType(GV, true, true) {
  if (GV && !isDerivedType(getTag()))
    GV = 0;
}

/// isCompositeType - Return true if the specified tag is legal for
/// DICompositeType.
bool DICompositeType::isCompositeType(unsigned TAG) {
  switch (TAG) {
  case dwarf::DW_TAG_array_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_vector_type:
  case dwarf::DW_TAG_subroutine_type:
    return true;
  default:
    return false;
  }
}

DICompositeType::DICompositeType(GlobalVariable *GV)
  : DIDerivedType(GV, true, true) {
  if (GV && !isCompositeType(getTag()))
    GV = 0;
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

DIVariable::DIVariable(GlobalVariable *GV) : DIDescriptor(GV) {
  if (GV && !isVariable(getTag()))
    GV = 0;
}



//===----------------------------------------------------------------------===//
// DIFactory: Basic Helpers
//===----------------------------------------------------------------------===//

Constant *DIFactory::GetTagConstant(unsigned TAG) {
  assert((TAG & DIDescriptor::VersionMask) == 0 &&
         "Tag too large for debug encoding!");
  return ConstantInt::get(Type::Int32Ty, TAG | DIDescriptor::Version6);
}

Constant *DIFactory::GetStringConstant(const std::string &String) {
  // Check string cache for previous edition.
  Constant *&Slot = StringCache[String];
  
  // Return Constant if previously defined.
  if (Slot) return Slot;
  
  const PointerType *DestTy = PointerType::getUnqual(Type::Int8Ty);
  
  // If empty string then use a sbyte* null instead.
  if (String.empty())
    return Slot = ConstantPointerNull::get(DestTy);

  // Construct string as an llvm constant.
  Constant *ConstStr = ConstantArray::get(String);
    
  // Otherwise create and return a new string global.
  GlobalVariable *StrGV = new GlobalVariable(ConstStr->getType(), true,
                                             GlobalVariable::InternalLinkage,
                                             ConstStr, ".str", &M);
  StrGV->setSection("llvm.metadata");
  return Slot = ConstantExpr::getBitCast(StrGV, DestTy);
}

/// GetOrCreateAnchor - Look up an anchor for the specified tag and name.  If it
/// already exists, return it.  If not, create a new one and return it.
DIAnchor DIFactory::GetOrCreateAnchor(unsigned TAG, const char *Name) {
  const Type *EltTy = StructType::get(Type::Int32Ty,  Type::Int32Ty, NULL);
  
  // Otherwise, create the global or return it if already in the module.
  Constant *C = M.getOrInsertGlobal(Name, EltTy);
  assert(isa<GlobalVariable>(C) && "Incorrectly typed anchor?");
  GlobalVariable *GV = cast<GlobalVariable>(C);
  
  // If it has an initializer, it is already in the module.
  if (GV->hasInitializer()) 
    return SubProgramAnchor = DIAnchor(GV);
  
  GV->setLinkage(GlobalValue::LinkOnceLinkage);
  GV->setSection("llvm.metadata");
  GV->setConstant(true);
  M.addTypeName("llvm.dbg.anchor.type", EltTy);
  
  // Otherwise, set the initializer.
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_anchor),
    ConstantInt::get(Type::Int32Ty, TAG)
  };
  
  GV->setInitializer(ConstantStruct::get(Elts, 2));
  return DIAnchor(GV);
}



//===----------------------------------------------------------------------===//
// DIFactory: Primary Constructors
//===----------------------------------------------------------------------===//

/// GetOrCreateCompileUnitAnchor - Return the anchor for compile units,
/// creating a new one if there isn't already one in the module.
DIAnchor DIFactory::GetOrCreateCompileUnitAnchor() {
  // If we already created one, just return it.
  if (!CompileUnitAnchor.isNull())
    return CompileUnitAnchor;
  return CompileUnitAnchor = GetOrCreateAnchor(dwarf::DW_TAG_compile_unit,
                                               "llvm.dbg.compile_units");
}

/// GetOrCreateSubprogramAnchor - Return the anchor for subprograms,
/// creating a new one if there isn't already one in the module.
DIAnchor DIFactory::GetOrCreateSubprogramAnchor() {
  // If we already created one, just return it.
  if (!SubProgramAnchor.isNull())
    return SubProgramAnchor;
  return SubProgramAnchor = GetOrCreateAnchor(dwarf::DW_TAG_subprogram,
                                              "llvm.dbg.subprograms");
}

/// GetOrCreateGlobalVariableAnchor - Return the anchor for globals,
/// creating a new one if there isn't already one in the module.
DIAnchor DIFactory::GetOrCreateGlobalVariableAnchor() {
  // If we already created one, just return it.
  if (!GlobalVariableAnchor.isNull())
    return GlobalVariableAnchor;
  return GlobalVariableAnchor = GetOrCreateAnchor(dwarf::DW_TAG_variable,
                                                  "llvm.dbg.global_variables");
}

/// GetOrCreateArray - Create an descriptor for an array of descriptors. 
/// This implicitly uniques the arrays created.
DIArray DIFactory::GetOrCreateArray(DIDescriptor *Tys, unsigned NumTys) {
  SmallVector<Constant*, 16> Elts;
  
  for (unsigned i = 0; i != NumTys; ++i)
    Elts.push_back(Tys[i].getCastToEmpty());
  
  const Type *EltTy = PointerType::getUnqual(StructType::get(NULL,NULL));
  
  Constant *Init = ConstantArray::get(ArrayType::get(EltTy, Elts.size()),
                                      &Elts[0], Elts.size());
  // If we already have this array, just return the uniqued version.
  DIDescriptor &Entry = SimpleConstantCache[Init];
  if (!Entry.isNull()) return DIArray(Entry.getGV());
  
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.array", &M);
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
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));

  // If we already have this range, just return the uniqued version.
  DIDescriptor &Entry = SimpleConstantCache[Init];
  if (!Entry.isNull()) return DISubrange(Entry.getGV());
  
  M.addTypeName("llvm.dbg.subrange.type", Init->getType());

  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.subrange", &M);
  GV->setSection("llvm.metadata");
  Entry = DIDescriptor(GV);
  return DISubrange(GV);
}



/// CreateCompileUnit - Create a new descriptor for the specified compile
/// unit.  Note that this does not unique compile units within the module.
DICompileUnit DIFactory::CreateCompileUnit(unsigned LangID,
                                           const std::string &Filename,
                                           const std::string &Directory,
                                           const std::string &Producer) {
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_compile_unit),
    GetOrCreateCompileUnitAnchor().getCastToEmpty(),
    ConstantInt::get(Type::Int32Ty, LangID),
    GetStringConstant(Filename),
    GetStringConstant(Directory),
    GetStringConstant(Producer)
  };
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.compile_unit.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.compile_unit", &M);
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
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.enumerator.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.enumerator", &M);
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
    Context.getCastToEmpty(),
    GetStringConstant(Name),
    CompileUnit.getCastToEmpty(),
    ConstantInt::get(Type::Int32Ty, LineNumber),
    ConstantInt::get(Type::Int64Ty, SizeInBits),
    ConstantInt::get(Type::Int64Ty, AlignInBits),
    ConstantInt::get(Type::Int64Ty, OffsetInBits),
    ConstantInt::get(Type::Int32Ty, Flags),
    ConstantInt::get(Type::Int32Ty, Encoding)
  };
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.basictype.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.basictype", &M);
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
    Context.getCastToEmpty(),
    GetStringConstant(Name),
    CompileUnit.getCastToEmpty(),
    ConstantInt::get(Type::Int32Ty, LineNumber),
    ConstantInt::get(Type::Int64Ty, SizeInBits),
    ConstantInt::get(Type::Int64Ty, AlignInBits),
    ConstantInt::get(Type::Int64Ty, OffsetInBits),
    ConstantInt::get(Type::Int32Ty, Flags),
    DerivedFrom.getCastToEmpty()
  };
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.derivedtype.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.derivedtype", &M);
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
                                               DIArray Elements) {
  Constant *Elts[] = {
    GetTagConstant(Tag),
    Context.getCastToEmpty(),
    GetStringConstant(Name),
    CompileUnit.getCastToEmpty(),
    ConstantInt::get(Type::Int32Ty, LineNumber),
    ConstantInt::get(Type::Int64Ty, SizeInBits),
    ConstantInt::get(Type::Int64Ty, AlignInBits),
    ConstantInt::get(Type::Int64Ty, OffsetInBits),
    ConstantInt::get(Type::Int32Ty, Flags),
    DerivedFrom.getCastToEmpty(),
    Elements.getCastToEmpty()
  };
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.composite.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.composite", &M);
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
    GetOrCreateSubprogramAnchor().getCastToEmpty(),
    Context.getCastToEmpty(),
    GetStringConstant(Name),
    GetStringConstant(DisplayName),
    GetStringConstant(LinkageName),
    CompileUnit.getCastToEmpty(),
    ConstantInt::get(Type::Int32Ty, LineNo),
    Type.getCastToEmpty(),
    ConstantInt::get(Type::Int1Ty, isLocalToUnit),
    ConstantInt::get(Type::Int1Ty, isDefinition)
  };
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.subprogram.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.subprogram", &M);
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
    GetOrCreateGlobalVariableAnchor().getCastToEmpty(),
    Context.getCastToEmpty(),
    GetStringConstant(Name),
    GetStringConstant(DisplayName),
    GetStringConstant(LinkageName),
    CompileUnit.getCastToEmpty(),
    ConstantInt::get(Type::Int32Ty, LineNo),
    Type.getCastToEmpty(),
    ConstantInt::get(Type::Int1Ty, isLocalToUnit),
    ConstantInt::get(Type::Int1Ty, isDefinition),
    ConstantExpr::getBitCast(Val, 
                             PointerType::getUnqual(StructType::get(NULL,NULL)))
  };
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.global_variable.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.global_variable", &M);
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
    Context.getCastToEmpty(),
    GetStringConstant(Name),
    CompileUnit.getCastToEmpty(),
    ConstantInt::get(Type::Int32Ty, LineNo),
    Type.getCastToEmpty(),
  };
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.variable.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.variable", &M);
  GV->setSection("llvm.metadata");
  return DIVariable(GV);
}


/// CreateBlock - This creates a descriptor for a lexical block with the
/// specified parent context.
DIBlock DIFactory::CreateBlock(DIDescriptor Context) {
  Constant *Elts[] = {
    GetTagConstant(dwarf::DW_TAG_lexical_block),
    Context.getCastToEmpty()
  };
  
  Constant *Init = ConstantStruct::get(Elts, sizeof(Elts)/sizeof(Elts[0]));
  
  M.addTypeName("llvm.dbg.block.type", Init->getType());
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true,
                                          GlobalValue::InternalLinkage,
                                          Init, "llvm.dbg.block", &M);
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
    llvm::ConstantInt::get(llvm::Type::Int32Ty, LineNo),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, ColNo),
    CU.getCastToEmpty()
  };
  CallInst::Create(StopPointFn, Args, Args+3, "", BB);
}

/// InsertSubprogramStart - Create a new llvm.dbg.func.start intrinsic to
/// mark the start of the specified subprogram.
void DIFactory::InsertSubprogramStart(DISubprogram SP, BasicBlock *BB) {
  // Lazily construct llvm.dbg.func.start.
  if (!FuncStartFn)
    FuncStartFn = llvm::Intrinsic::getDeclaration(&M, 
                                              llvm::Intrinsic::dbg_func_start);
  
  // Call llvm.dbg.func.start which also implicitly sets a stoppoint.
  CallInst::Create(FuncStartFn, SP.getCastToEmpty(), "", BB);
}

/// InsertRegionStart - Insert a new llvm.dbg.region.start intrinsic call to
/// mark the start of a region for the specified scoping descriptor.
void DIFactory::InsertRegionStart(DIDescriptor D, BasicBlock *BB) {
  // Lazily construct llvm.dbg.region.start function.
  if (!RegionStartFn)
    RegionStartFn = llvm::Intrinsic::getDeclaration(&M, 
                                            llvm::Intrinsic::dbg_region_start);
  // Call llvm.dbg.func.start.
  CallInst::Create(RegionStartFn, D.getCastToEmpty(), "", BB);
}


/// InsertRegionEnd - Insert a new llvm.dbg.region.end intrinsic call to
/// mark the end of a region for the specified scoping descriptor.
void DIFactory::InsertRegionEnd(DIDescriptor D, BasicBlock *BB) {
  // Lazily construct llvm.dbg.region.end function.
  if (!RegionEndFn)
    RegionEndFn = llvm::Intrinsic::getDeclaration(&M,
                                               llvm::Intrinsic::dbg_region_end);
  
  CallInst::Create(RegionEndFn, D.getCastToEmpty(), "", BB);
}

/// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
void DIFactory::InsertDeclare(llvm::Value *Storage, DIVariable D,
                              BasicBlock *BB) {
  // Cast the storage to a {}* for the call to llvm.dbg.declare.
  const Type *DestTy = PointerType::getUnqual(StructType::get(NULL, NULL));
  Storage = new llvm::BitCastInst(Storage, DestTy, Storage->getName(), BB);
  
  if (!DeclareFn)
    DeclareFn = llvm::Intrinsic::getDeclaration(&M,
                                                llvm::Intrinsic::dbg_declare);
  Value *Args[] = { Storage, D.getCastToEmpty() };
  CallInst::Create(DeclareFn, Args, Args+2, "", BB);
}
