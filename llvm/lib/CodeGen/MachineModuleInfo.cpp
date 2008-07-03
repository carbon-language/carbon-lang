//===-- llvm/CodeGen/MachineModuleInfo.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfo.h"

#include "llvm/Constants.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Streams.h"
using namespace llvm;
using namespace llvm::dwarf;

// Handle the Pass registration stuff necessary to use TargetData's.
static RegisterPass<MachineModuleInfo>
X("machinemoduleinfo", "Module Information");
char MachineModuleInfo::ID = 0;

//===----------------------------------------------------------------------===//

/// getGlobalVariablesUsing - Return all of the GlobalVariables which have the
/// specified value in their initializer somewhere.
static void
getGlobalVariablesUsing(Value *V, std::vector<GlobalVariable*> &Result) {
  // Scan though value users.
  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(*I)) {
      // If the user is a GlobalVariable then add to result.
      Result.push_back(GV);
    } else if (Constant *C = dyn_cast<Constant>(*I)) {
      // If the user is a constant variable then scan its users
      getGlobalVariablesUsing(C, Result);
    }
  }
}

/// getGlobalVariablesUsing - Return all of the GlobalVariables that use the
/// named GlobalVariable.
static std::vector<GlobalVariable*>
getGlobalVariablesUsing(Module &M, const std::string &RootName) {
  std::vector<GlobalVariable*> Result;  // GlobalVariables matching criteria.
  
  std::vector<const Type*> FieldTypes;
  FieldTypes.push_back(Type::Int32Ty);
  FieldTypes.push_back(Type::Int32Ty);

  // Get the GlobalVariable root.
  GlobalVariable *UseRoot = M.getGlobalVariable(RootName,
                                                StructType::get(FieldTypes));

  // If present and linkonce then scan for users.
  if (UseRoot && UseRoot->hasLinkOnceLinkage()) {
    getGlobalVariablesUsing(UseRoot, Result);
  }
  
  return Result;
}
  
/// isStringValue - Return true if the given value can be coerced to a string.
///
static bool isStringValue(Value *V) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    if (GV->hasInitializer() && isa<ConstantArray>(GV->getInitializer())) {
      ConstantArray *Init = cast<ConstantArray>(GV->getInitializer());
      return Init->isString();
    }
  } else if (Constant *C = dyn_cast<Constant>(V)) {
    if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
      return isStringValue(GV);
    else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        if (CE->getNumOperands() == 3 &&
            cast<Constant>(CE->getOperand(1))->isNullValue() &&
            isa<ConstantInt>(CE->getOperand(2))) {
          return isStringValue(CE->getOperand(0));
        }
      }
    }
  }
  return false;
}

/// getGlobalVariable - Return either a direct or cast Global value.
///
static GlobalVariable *getGlobalVariable(Value *V) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    return GV;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::BitCast) {
      return dyn_cast<GlobalVariable>(CE->getOperand(0));
    } else if (CE->getOpcode() == Instruction::GetElementPtr) {
      for (unsigned int i=1; i<CE->getNumOperands(); i++) {
        if (!CE->getOperand(i)->isNullValue())
          return NULL;
      }
      return dyn_cast<GlobalVariable>(CE->getOperand(0));
    }
  }
  return NULL;
}

/// isGlobalVariable - Return true if the given value can be coerced to a
/// GlobalVariable.
static bool isGlobalVariable(Value *V) {
  if (isa<GlobalVariable>(V) || isa<ConstantPointerNull>(V)) {
    return true;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::BitCast) {
      return isa<GlobalVariable>(CE->getOperand(0));
    } else if (CE->getOpcode() == Instruction::GetElementPtr) {
      for (unsigned int i=1; i<CE->getNumOperands(); i++) {
        if (!CE->getOperand(i)->isNullValue())
          return false;
      }
      return isa<GlobalVariable>(CE->getOperand(0));
    }
  }
  return false;
}

/// getUIntOperand - Return ith operand if it is an unsigned integer.
///
static ConstantInt *getUIntOperand(GlobalVariable *GV, unsigned i) {
  // Make sure the GlobalVariable has an initializer.
  if (!GV->hasInitializer()) return NULL;
  
  // Get the initializer constant.
  ConstantStruct *CI = dyn_cast<ConstantStruct>(GV->getInitializer());
  if (!CI) return NULL;
  
  // Check if there is at least i + 1 operands.
  unsigned N = CI->getNumOperands();
  if (i >= N) return NULL;

  // Check constant.
  return dyn_cast<ConstantInt>(CI->getOperand(i));
}

//===----------------------------------------------------------------------===//

/// ApplyToFields - Target the visitor to each field of the debug information
/// descriptor.
void DIVisitor::ApplyToFields(DebugInfoDesc *DD) {
  DD->ApplyToFields(this);
}

namespace {

//===----------------------------------------------------------------------===//
/// DICountVisitor - This DIVisitor counts all the fields in the supplied debug
/// the supplied DebugInfoDesc.
class DICountVisitor : public DIVisitor {
private:
  unsigned Count;                       // Running count of fields.
  
public:
  DICountVisitor() : DIVisitor(), Count(0) {}
  
  // Accessors.
  unsigned getCount() const { return Count; }
  
  /// Apply - Count each of the fields.
  ///
  virtual void Apply(int &Field)             { ++Count; }
  virtual void Apply(unsigned &Field)        { ++Count; }
  virtual void Apply(int64_t &Field)         { ++Count; }
  virtual void Apply(uint64_t &Field)        { ++Count; }
  virtual void Apply(bool &Field)            { ++Count; }
  virtual void Apply(std::string &Field)     { ++Count; }
  virtual void Apply(DebugInfoDesc *&Field)  { ++Count; }
  virtual void Apply(GlobalVariable *&Field) { ++Count; }
  virtual void Apply(std::vector<DebugInfoDesc *> &Field) {
    ++Count;
  }
};

//===----------------------------------------------------------------------===//
/// DIDeserializeVisitor - This DIVisitor deserializes all the fields in the
/// supplied DebugInfoDesc.
class DIDeserializeVisitor : public DIVisitor {
private:
  DIDeserializer &DR;                   // Active deserializer.
  unsigned I;                           // Current operand index.
  ConstantStruct *CI;                   // GlobalVariable constant initializer.

public:
  DIDeserializeVisitor(DIDeserializer &D, GlobalVariable *GV)
    : DIVisitor(), DR(D), I(0), CI(cast<ConstantStruct>(GV->getInitializer()))
  {}
  
  /// Apply - Set the value of each of the fields.
  ///
  virtual void Apply(int &Field) {
    Constant *C = CI->getOperand(I++);
    Field = cast<ConstantInt>(C)->getSExtValue();
  }
  virtual void Apply(unsigned &Field) {
    Constant *C = CI->getOperand(I++);
    Field = cast<ConstantInt>(C)->getZExtValue();
  }
  virtual void Apply(int64_t &Field) {
    Constant *C = CI->getOperand(I++);
    Field = cast<ConstantInt>(C)->getSExtValue();
  }
  virtual void Apply(uint64_t &Field) {
    Constant *C = CI->getOperand(I++);
    Field = cast<ConstantInt>(C)->getZExtValue();
  }
  virtual void Apply(bool &Field) {
    Constant *C = CI->getOperand(I++);
    Field = cast<ConstantInt>(C)->getZExtValue();
  }
  virtual void Apply(std::string &Field) {
    Constant *C = CI->getOperand(I++);
    // Fills in the string if it succeeds
    if (!GetConstantStringInfo(C, Field))
      Field.clear();
  }
  virtual void Apply(DebugInfoDesc *&Field) {
    Constant *C = CI->getOperand(I++);
    Field = DR.Deserialize(C);
  }
  virtual void Apply(GlobalVariable *&Field) {
    Constant *C = CI->getOperand(I++);
    Field = getGlobalVariable(C);
  }
  virtual void Apply(std::vector<DebugInfoDesc *> &Field) {
    Field.resize(0);
    Constant *C = CI->getOperand(I++);
    GlobalVariable *GV = getGlobalVariable(C);
    if (GV->hasInitializer()) {
      if (ConstantArray *CA = dyn_cast<ConstantArray>(GV->getInitializer())) {
        for (unsigned i = 0, N = CA->getNumOperands(); i < N; ++i) {
          GlobalVariable *GVE = getGlobalVariable(CA->getOperand(i));
          DebugInfoDesc *DE = DR.Deserialize(GVE);
          Field.push_back(DE);
        }
      } else if (GV->getInitializer()->isNullValue()) {
        if (const ArrayType *T =
            dyn_cast<ArrayType>(GV->getType()->getElementType())) {
          Field.resize(T->getNumElements());
        }
      }
    }
  }
};

//===----------------------------------------------------------------------===//
/// DISerializeVisitor - This DIVisitor serializes all the fields in
/// the supplied DebugInfoDesc.
class DISerializeVisitor : public DIVisitor {
private:
  DISerializer &SR;                     // Active serializer.
  std::vector<Constant*> &Elements;     // Element accumulator.
  
public:
  DISerializeVisitor(DISerializer &S, std::vector<Constant*> &E)
  : DIVisitor()
  , SR(S)
  , Elements(E)
  {}
  
  /// Apply - Set the value of each of the fields.
  ///
  virtual void Apply(int &Field) {
    Elements.push_back(ConstantInt::get(Type::Int32Ty, int32_t(Field)));
  }
  virtual void Apply(unsigned &Field) {
    Elements.push_back(ConstantInt::get(Type::Int32Ty, uint32_t(Field)));
  }
  virtual void Apply(int64_t &Field) {
    Elements.push_back(ConstantInt::get(Type::Int64Ty, int64_t(Field)));
  }
  virtual void Apply(uint64_t &Field) {
    Elements.push_back(ConstantInt::get(Type::Int64Ty, uint64_t(Field)));
  }
  virtual void Apply(bool &Field) {
    Elements.push_back(ConstantInt::get(Type::Int1Ty, Field));
  }
  virtual void Apply(std::string &Field) {
    Elements.push_back(SR.getString(Field));
  }
  virtual void Apply(DebugInfoDesc *&Field) {
    GlobalVariable *GV = NULL;
    
    // If non-NULL then convert to global.
    if (Field) GV = SR.Serialize(Field);
    
    // FIXME - At some point should use specific type.
    const PointerType *EmptyTy = SR.getEmptyStructPtrType();
    
    if (GV) {
      // Set to pointer to global.
      Elements.push_back(ConstantExpr::getBitCast(GV, EmptyTy));
    } else {
      // Use NULL.
      Elements.push_back(ConstantPointerNull::get(EmptyTy));
    }
  }
  virtual void Apply(GlobalVariable *&Field) {
    const PointerType *EmptyTy = SR.getEmptyStructPtrType();
    if (Field) {
      Elements.push_back(ConstantExpr::getBitCast(Field, EmptyTy));
    } else {
      Elements.push_back(ConstantPointerNull::get(EmptyTy));
    }
  }
  virtual void Apply(std::vector<DebugInfoDesc *> &Field) {
    const PointerType *EmptyTy = SR.getEmptyStructPtrType();
    unsigned N = Field.size();
    ArrayType *AT = ArrayType::get(EmptyTy, N);
    std::vector<Constant *> ArrayElements;

    for (unsigned i = 0, N = Field.size(); i < N; ++i) {
      if (DebugInfoDesc *Element = Field[i]) {
        GlobalVariable *GVE = SR.Serialize(Element);
        Constant *CE = ConstantExpr::getBitCast(GVE, EmptyTy);
        ArrayElements.push_back(cast<Constant>(CE));
      } else {
        ArrayElements.push_back(ConstantPointerNull::get(EmptyTy));
      }
    }
    
    Constant *CA = ConstantArray::get(AT, ArrayElements);
    GlobalVariable *CAGV = new GlobalVariable(AT, true,
                                              GlobalValue::InternalLinkage,
                                              CA, "llvm.dbg.array",
                                              SR.getModule());
    CAGV->setSection("llvm.metadata");
    Constant *CAE = ConstantExpr::getBitCast(CAGV, EmptyTy);
    Elements.push_back(CAE);
  }
};

//===----------------------------------------------------------------------===//
/// DIGetTypesVisitor - This DIVisitor gathers all the field types in
/// the supplied DebugInfoDesc.
class DIGetTypesVisitor : public DIVisitor {
private:
  DISerializer &SR;                     // Active serializer.
  std::vector<const Type*> &Fields;     // Type accumulator.
  
public:
  DIGetTypesVisitor(DISerializer &S, std::vector<const Type*> &F)
  : DIVisitor()
  , SR(S)
  , Fields(F)
  {}
  
  /// Apply - Set the value of each of the fields.
  ///
  virtual void Apply(int &Field) {
    Fields.push_back(Type::Int32Ty);
  }
  virtual void Apply(unsigned &Field) {
    Fields.push_back(Type::Int32Ty);
  }
  virtual void Apply(int64_t &Field) {
    Fields.push_back(Type::Int64Ty);
  }
  virtual void Apply(uint64_t &Field) {
    Fields.push_back(Type::Int64Ty);
  }
  virtual void Apply(bool &Field) {
    Fields.push_back(Type::Int1Ty);
  }
  virtual void Apply(std::string &Field) {
    Fields.push_back(SR.getStrPtrType());
  }
  virtual void Apply(DebugInfoDesc *&Field) {
    // FIXME - At some point should use specific type.
    const PointerType *EmptyTy = SR.getEmptyStructPtrType();
    Fields.push_back(EmptyTy);
  }
  virtual void Apply(GlobalVariable *&Field) {
    const PointerType *EmptyTy = SR.getEmptyStructPtrType();
    Fields.push_back(EmptyTy);
  }
  virtual void Apply(std::vector<DebugInfoDesc *> &Field) {
    const PointerType *EmptyTy = SR.getEmptyStructPtrType();
    Fields.push_back(EmptyTy);
  }
};

//===----------------------------------------------------------------------===//
/// DIVerifyVisitor - This DIVisitor verifies all the field types against
/// a constant initializer.
class DIVerifyVisitor : public DIVisitor {
private:
  DIVerifier &VR;                       // Active verifier.
  bool IsValid;                         // Validity status.
  unsigned I;                           // Current operand index.
  ConstantStruct *CI;                   // GlobalVariable constant initializer.
  
public:
  DIVerifyVisitor(DIVerifier &V, GlobalVariable *GV)
  : DIVisitor()
  , VR(V)
  , IsValid(true)
  , I(0)
  , CI(cast<ConstantStruct>(GV->getInitializer()))
  {
  }
  
  // Accessors.
  bool isValid() const { return IsValid; }
  
  /// Apply - Set the value of each of the fields.
  ///
  virtual void Apply(int &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isa<ConstantInt>(C);
  }
  virtual void Apply(unsigned &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isa<ConstantInt>(C);
  }
  virtual void Apply(int64_t &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isa<ConstantInt>(C);
  }
  virtual void Apply(uint64_t &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isa<ConstantInt>(C);
  }
  virtual void Apply(bool &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isa<ConstantInt>(C) && C->getType() == Type::Int1Ty;
  }
  virtual void Apply(std::string &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid &&
              (!C || isStringValue(C) || C->isNullValue());
  }
  virtual void Apply(DebugInfoDesc *&Field) {
    // FIXME - Prepare the correct descriptor.
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isGlobalVariable(C);
  }
  virtual void Apply(GlobalVariable *&Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isGlobalVariable(C);
  }
  virtual void Apply(std::vector<DebugInfoDesc *> &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isGlobalVariable(C);
    if (!IsValid) return;

    GlobalVariable *GV = getGlobalVariable(C);
    IsValid = IsValid && GV && GV->hasInitializer();
    if (!IsValid) return;
    
    ConstantArray *CA = dyn_cast<ConstantArray>(GV->getInitializer());
    IsValid = IsValid && CA;
    if (!IsValid) return;

    for (unsigned i = 0, N = CA->getNumOperands(); IsValid && i < N; ++i) {
      IsValid = IsValid && isGlobalVariable(CA->getOperand(i));
      if (!IsValid) return;
    
      GlobalVariable *GVE = getGlobalVariable(CA->getOperand(i));
      VR.Verify(GVE);
    }
  }
};

}

//===----------------------------------------------------------------------===//

/// TagFromGlobal - Returns the tag number from a debug info descriptor
/// GlobalVariable.   Return DIIValid if operand is not an unsigned int. 
unsigned DebugInfoDesc::TagFromGlobal(GlobalVariable *GV) {
  ConstantInt *C = getUIntOperand(GV, 0);
  return C ? ((unsigned)C->getZExtValue() & ~LLVMDebugVersionMask) :
             (unsigned)DW_TAG_invalid;
}

/// VersionFromGlobal - Returns the version number from a debug info
/// descriptor GlobalVariable.  Return DIIValid if operand is not an unsigned
/// int.
unsigned  DebugInfoDesc::VersionFromGlobal(GlobalVariable *GV) {
  ConstantInt *C = getUIntOperand(GV, 0);
  return C ? ((unsigned)C->getZExtValue() & LLVMDebugVersionMask) :
             (unsigned)DW_TAG_invalid;
}

/// DescFactory - Create an instance of debug info descriptor based on Tag.
/// Return NULL if not a recognized Tag.
DebugInfoDesc *DebugInfoDesc::DescFactory(unsigned Tag) {
  switch (Tag) {
  case DW_TAG_anchor:           return new AnchorDesc();
  case DW_TAG_compile_unit:     return new CompileUnitDesc();
  case DW_TAG_variable:         return new GlobalVariableDesc();
  case DW_TAG_subprogram:       return new SubprogramDesc();
  case DW_TAG_lexical_block:    return new BlockDesc();
  case DW_TAG_base_type:        return new BasicTypeDesc();
  case DW_TAG_typedef:
  case DW_TAG_pointer_type:        
  case DW_TAG_reference_type:
  case DW_TAG_const_type:
  case DW_TAG_volatile_type:        
  case DW_TAG_restrict_type:
  case DW_TAG_member:
  case DW_TAG_inheritance:      return new DerivedTypeDesc(Tag);
  case DW_TAG_array_type:
  case DW_TAG_structure_type:
  case DW_TAG_union_type:
  case DW_TAG_enumeration_type:
  case DW_TAG_vector_type:
  case DW_TAG_subroutine_type:  return new CompositeTypeDesc(Tag);
  case DW_TAG_subrange_type:    return new SubrangeDesc();
  case DW_TAG_enumerator:       return new EnumeratorDesc();
  case DW_TAG_return_variable:
  case DW_TAG_arg_variable:
  case DW_TAG_auto_variable:    return new VariableDesc(Tag);
  default: break;
  }
  return NULL;
}

/// getLinkage - get linkage appropriate for this type of descriptor.
///
GlobalValue::LinkageTypes DebugInfoDesc::getLinkage() const {
  return GlobalValue::InternalLinkage;
}

/// ApplyToFields - Target the vistor to the fields of the descriptor.
///
void DebugInfoDesc::ApplyToFields(DIVisitor *Visitor) {
  Visitor->Apply(Tag);
}

//===----------------------------------------------------------------------===//

AnchorDesc::AnchorDesc()
: DebugInfoDesc(DW_TAG_anchor)
, AnchorTag(0)
{}
AnchorDesc::AnchorDesc(AnchoredDesc *D)
: DebugInfoDesc(DW_TAG_anchor)
, AnchorTag(D->getTag())
{}

// Implement isa/cast/dyncast.
bool AnchorDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_anchor;
}
  
/// getLinkage - get linkage appropriate for this type of descriptor.
///
GlobalValue::LinkageTypes AnchorDesc::getLinkage() const {
  return GlobalValue::LinkOnceLinkage;
}

/// ApplyToFields - Target the visitor to the fields of the TransUnitDesc.
///
void AnchorDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  
  Visitor->Apply(AnchorTag);
}

/// getDescString - Return a string used to compose global names and labels. A
/// A global variable name needs to be defined for each debug descriptor that is
/// anchored. NOTE: that each global variable named here also needs to be added
/// to the list of names left external in the internalizer.
///   ExternalNames.insert("llvm.dbg.compile_units");
///   ExternalNames.insert("llvm.dbg.global_variables");
///   ExternalNames.insert("llvm.dbg.subprograms");
const char *AnchorDesc::getDescString() const {
  switch (AnchorTag) {
  case DW_TAG_compile_unit: return CompileUnitDesc::AnchorString;
  case DW_TAG_variable:     return GlobalVariableDesc::AnchorString;
  case DW_TAG_subprogram:   return SubprogramDesc::AnchorString;
  default: break;
  }

  assert(0 && "Tag does not have a case for anchor string");
  return "";
}

/// getTypeString - Return a string used to label this descriptors type.
///
const char *AnchorDesc::getTypeString() const {
  return "llvm.dbg.anchor.type";
}

#ifndef NDEBUG
void AnchorDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "AnchorTag(" << AnchorTag << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

AnchoredDesc::AnchoredDesc(unsigned T)
: DebugInfoDesc(T)
, Anchor(NULL)
{}

/// ApplyToFields - Target the visitor to the fields of the AnchoredDesc.
///
void AnchoredDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);

  Visitor->Apply(Anchor);
}

//===----------------------------------------------------------------------===//

CompileUnitDesc::CompileUnitDesc()
: AnchoredDesc(DW_TAG_compile_unit)
, Language(0)
, FileName("")
, Directory("")
, Producer("")
{}

// Implement isa/cast/dyncast.
bool CompileUnitDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_compile_unit;
}

/// ApplyToFields - Target the visitor to the fields of the CompileUnitDesc.
///
void CompileUnitDesc::ApplyToFields(DIVisitor *Visitor) {
  AnchoredDesc::ApplyToFields(Visitor);
  
  // Handle cases out of sync with compiler.
  if (getVersion() == 0) {
    unsigned DebugVersion;
    Visitor->Apply(DebugVersion);
  }

  Visitor->Apply(Language);
  Visitor->Apply(FileName);
  Visitor->Apply(Directory);
  Visitor->Apply(Producer);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *CompileUnitDesc::getDescString() const {
  return "llvm.dbg.compile_unit";
}

/// getTypeString - Return a string used to label this descriptors type.
///
const char *CompileUnitDesc::getTypeString() const {
  return "llvm.dbg.compile_unit.type";
}

/// getAnchorString - Return a string used to label this descriptor's anchor.
///
const char *const CompileUnitDesc::AnchorString = "llvm.dbg.compile_units";
const char *CompileUnitDesc::getAnchorString() const {
  return AnchorString;
}

#ifndef NDEBUG
void CompileUnitDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Anchor(" << getAnchor() << "), "
       << "Language(" << Language << "), "
       << "FileName(\"" << FileName << "\"), "
       << "Directory(\"" << Directory << "\"), "
       << "Producer(\"" << Producer << "\")\n";
}
#endif

//===----------------------------------------------------------------------===//

TypeDesc::TypeDesc(unsigned T)
: DebugInfoDesc(T)
, Context(NULL)
, Name("")
, File(NULL)
, Line(0)
, Size(0)
, Align(0)
, Offset(0)
, Flags(0)
{}

/// ApplyToFields - Target the visitor to the fields of the TypeDesc.
///
void TypeDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  
  Visitor->Apply(Context);
  Visitor->Apply(Name);
  Visitor->Apply(File);
  Visitor->Apply(Line);
  Visitor->Apply(Size);
  Visitor->Apply(Align);
  Visitor->Apply(Offset);
  if (getVersion() > LLVMDebugVersion4) Visitor->Apply(Flags);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *TypeDesc::getDescString() const {
  return "llvm.dbg.type";
}

/// getTypeString - Return a string used to label this descriptor's type.
///
const char *TypeDesc::getTypeString() const {
  return "llvm.dbg.type.type";
}

#ifndef NDEBUG
void TypeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << Context << "), "
       << "Name(\"" << Name << "\"), "
       << "File(" << File << "), "
       << "Line(" << Line << "), "
       << "Size(" << Size << "), "
       << "Align(" << Align << "), "
       << "Offset(" << Offset << "), "
       << "Flags(" << Flags << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

BasicTypeDesc::BasicTypeDesc()
: TypeDesc(DW_TAG_base_type)
, Encoding(0)
{}

// Implement isa/cast/dyncast.
bool BasicTypeDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_base_type;
}

/// ApplyToFields - Target the visitor to the fields of the BasicTypeDesc.
///
void BasicTypeDesc::ApplyToFields(DIVisitor *Visitor) {
  TypeDesc::ApplyToFields(Visitor);
  
  Visitor->Apply(Encoding);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *BasicTypeDesc::getDescString() const {
  return "llvm.dbg.basictype";
}

/// getTypeString - Return a string used to label this descriptor's type.
///
const char *BasicTypeDesc::getTypeString() const {
  return "llvm.dbg.basictype.type";
}

#ifndef NDEBUG
void BasicTypeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << getContext() << "), "
       << "Name(\"" << getName() << "\"), "
       << "Size(" << getSize() << "), "
       << "Encoding(" << Encoding << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

DerivedTypeDesc::DerivedTypeDesc(unsigned T)
: TypeDesc(T)
, FromType(NULL)
{}

// Implement isa/cast/dyncast.
bool DerivedTypeDesc::classof(const DebugInfoDesc *D) {
  unsigned T =  D->getTag();
  switch (T) {
  case DW_TAG_typedef:
  case DW_TAG_pointer_type:
  case DW_TAG_reference_type:
  case DW_TAG_const_type:
  case DW_TAG_volatile_type:
  case DW_TAG_restrict_type:
  case DW_TAG_member:
  case DW_TAG_inheritance:
    return true;
  default: break;
  }
  return false;
}

/// ApplyToFields - Target the visitor to the fields of the DerivedTypeDesc.
///
void DerivedTypeDesc::ApplyToFields(DIVisitor *Visitor) {
  TypeDesc::ApplyToFields(Visitor);
  
  Visitor->Apply(FromType);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *DerivedTypeDesc::getDescString() const {
  return "llvm.dbg.derivedtype";
}

/// getTypeString - Return a string used to label this descriptor's type.
///
const char *DerivedTypeDesc::getTypeString() const {
  return "llvm.dbg.derivedtype.type";
}

#ifndef NDEBUG
void DerivedTypeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << getContext() << "), "
       << "Name(\"" << getName() << "\"), "
       << "Size(" << getSize() << "), "
       << "File(" << getFile() << "), "
       << "Line(" << getLine() << "), "
       << "FromType(" << FromType << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

CompositeTypeDesc::CompositeTypeDesc(unsigned T)
: DerivedTypeDesc(T)
, Elements()
{}
  
// Implement isa/cast/dyncast.
bool CompositeTypeDesc::classof(const DebugInfoDesc *D) {
  unsigned T =  D->getTag();
  switch (T) {
  case DW_TAG_array_type:
  case DW_TAG_structure_type:
  case DW_TAG_union_type:
  case DW_TAG_enumeration_type:
  case DW_TAG_vector_type:
  case DW_TAG_subroutine_type:
    return true;
  default: break;
  }
  return false;
}

/// ApplyToFields - Target the visitor to the fields of the CompositeTypeDesc.
///
void CompositeTypeDesc::ApplyToFields(DIVisitor *Visitor) {
  DerivedTypeDesc::ApplyToFields(Visitor);  

  Visitor->Apply(Elements);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *CompositeTypeDesc::getDescString() const {
  return "llvm.dbg.compositetype";
}

/// getTypeString - Return a string used to label this descriptor's type.
///
const char *CompositeTypeDesc::getTypeString() const {
  return "llvm.dbg.compositetype.type";
}

#ifndef NDEBUG
void CompositeTypeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << getContext() << "), "
       << "Name(\"" << getName() << "\"), "
       << "Size(" << getSize() << "), "
       << "File(" << getFile() << "), "
       << "Line(" << getLine() << "), "
       << "FromType(" << getFromType() << "), "
       << "Elements.size(" << Elements.size() << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

SubrangeDesc::SubrangeDesc()
: DebugInfoDesc(DW_TAG_subrange_type)
, Lo(0)
, Hi(0)
{}

// Implement isa/cast/dyncast.
bool SubrangeDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_subrange_type;
}

/// ApplyToFields - Target the visitor to the fields of the SubrangeDesc.
///
void SubrangeDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);

  Visitor->Apply(Lo);
  Visitor->Apply(Hi);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *SubrangeDesc::getDescString() const {
  return "llvm.dbg.subrange";
}
  
/// getTypeString - Return a string used to label this descriptor's type.
///
const char *SubrangeDesc::getTypeString() const {
  return "llvm.dbg.subrange.type";
}

#ifndef NDEBUG
void SubrangeDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Lo(" << Lo << "), "
       << "Hi(" << Hi << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

EnumeratorDesc::EnumeratorDesc()
: DebugInfoDesc(DW_TAG_enumerator)
, Name("")
, Value(0)
{}

// Implement isa/cast/dyncast.
bool EnumeratorDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_enumerator;
}

/// ApplyToFields - Target the visitor to the fields of the EnumeratorDesc.
///
void EnumeratorDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);

  Visitor->Apply(Name);
  Visitor->Apply(Value);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *EnumeratorDesc::getDescString() const {
  return "llvm.dbg.enumerator";
}
  
/// getTypeString - Return a string used to label this descriptor's type.
///
const char *EnumeratorDesc::getTypeString() const {
  return "llvm.dbg.enumerator.type";
}

#ifndef NDEBUG
void EnumeratorDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Name(" << Name << "), "
       << "Value(" << Value << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

VariableDesc::VariableDesc(unsigned T)
: DebugInfoDesc(T)
, Context(NULL)
, Name("")
, File(NULL)
, Line(0)
, TyDesc(0)
{}

// Implement isa/cast/dyncast.
bool VariableDesc::classof(const DebugInfoDesc *D) {
  unsigned T =  D->getTag();
  switch (T) {
  case DW_TAG_auto_variable:
  case DW_TAG_arg_variable:
  case DW_TAG_return_variable:
    return true;
  default: break;
  }
  return false;
}

/// ApplyToFields - Target the visitor to the fields of the VariableDesc.
///
void VariableDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);
  
  Visitor->Apply(Context);
  Visitor->Apply(Name);
  Visitor->Apply(File);
  Visitor->Apply(Line);
  Visitor->Apply(TyDesc);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *VariableDesc::getDescString() const {
  return "llvm.dbg.variable";
}

/// getTypeString - Return a string used to label this descriptor's type.
///
const char *VariableDesc::getTypeString() const {
  return "llvm.dbg.variable.type";
}

#ifndef NDEBUG
void VariableDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Context(" << Context << "), "
       << "Name(\"" << Name << "\"), "
       << "File(" << File << "), "
       << "Line(" << Line << "), "
       << "TyDesc(" << TyDesc << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

GlobalDesc::GlobalDesc(unsigned T)
: AnchoredDesc(T)
, Context(0)
, Name("")
, FullName("")
, LinkageName("")
, File(NULL)
, Line(0)
, TyDesc(NULL)
, IsStatic(false)
, IsDefinition(false)
{}

/// ApplyToFields - Target the visitor to the fields of the global.
///
void GlobalDesc::ApplyToFields(DIVisitor *Visitor) {
  AnchoredDesc::ApplyToFields(Visitor);

  Visitor->Apply(Context);
  Visitor->Apply(Name);
  Visitor->Apply(FullName);
  Visitor->Apply(LinkageName);
  Visitor->Apply(File);
  Visitor->Apply(Line);
  Visitor->Apply(TyDesc);
  Visitor->Apply(IsStatic);
  Visitor->Apply(IsDefinition);
}

//===----------------------------------------------------------------------===//

GlobalVariableDesc::GlobalVariableDesc()
: GlobalDesc(DW_TAG_variable)
, Global(NULL)
{}

// Implement isa/cast/dyncast.
bool GlobalVariableDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_variable; 
}

/// ApplyToFields - Target the visitor to the fields of the GlobalVariableDesc.
///
void GlobalVariableDesc::ApplyToFields(DIVisitor *Visitor) {
  GlobalDesc::ApplyToFields(Visitor);

  Visitor->Apply(Global);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *GlobalVariableDesc::getDescString() const {
  return "llvm.dbg.global_variable";
}

/// getTypeString - Return a string used to label this descriptors type.
///
const char *GlobalVariableDesc::getTypeString() const {
  return "llvm.dbg.global_variable.type";
}

/// getAnchorString - Return a string used to label this descriptor's anchor.
///
const char *const GlobalVariableDesc::AnchorString = "llvm.dbg.global_variables";
const char *GlobalVariableDesc::getAnchorString() const {
  return AnchorString;
}

#ifndef NDEBUG
void GlobalVariableDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Anchor(" << getAnchor() << "), "
       << "Name(\"" << getName() << "\"), "
       << "FullName(\"" << getFullName() << "\"), "
       << "LinkageName(\"" << getLinkageName() << "\"), "
       << "File(" << getFile() << "),"
       << "Line(" << getLine() << "),"
       << "Type(" << getType() << "), "
       << "IsStatic(" << (isStatic() ? "true" : "false") << "), "
       << "IsDefinition(" << (isDefinition() ? "true" : "false") << "), "
       << "Global(" << Global << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

SubprogramDesc::SubprogramDesc()
: GlobalDesc(DW_TAG_subprogram)
{}

// Implement isa/cast/dyncast.
bool SubprogramDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_subprogram;
}

/// ApplyToFields - Target the visitor to the fields of the
/// SubprogramDesc.
void SubprogramDesc::ApplyToFields(DIVisitor *Visitor) {
  GlobalDesc::ApplyToFields(Visitor);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *SubprogramDesc::getDescString() const {
  return "llvm.dbg.subprogram";
}

/// getTypeString - Return a string used to label this descriptors type.
///
const char *SubprogramDesc::getTypeString() const {
  return "llvm.dbg.subprogram.type";
}

/// getAnchorString - Return a string used to label this descriptor's anchor.
///
const char *const SubprogramDesc::AnchorString = "llvm.dbg.subprograms";
const char *SubprogramDesc::getAnchorString() const {
  return AnchorString;
}

#ifndef NDEBUG
void SubprogramDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "), "
       << "Anchor(" << getAnchor() << "), "
       << "Name(\"" << getName() << "\"), "
       << "FullName(\"" << getFullName() << "\"), "
       << "LinkageName(\"" << getLinkageName() << "\"), "
       << "File(" << getFile() << "),"
       << "Line(" << getLine() << "),"
       << "Type(" << getType() << "), "
       << "IsStatic(" << (isStatic() ? "true" : "false") << "), "
       << "IsDefinition(" << (isDefinition() ? "true" : "false") << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

BlockDesc::BlockDesc()
: DebugInfoDesc(DW_TAG_lexical_block)
, Context(NULL)
{}

// Implement isa/cast/dyncast.
bool BlockDesc::classof(const DebugInfoDesc *D) {
  return D->getTag() == DW_TAG_lexical_block;
}

/// ApplyToFields - Target the visitor to the fields of the BlockDesc.
///
void BlockDesc::ApplyToFields(DIVisitor *Visitor) {
  DebugInfoDesc::ApplyToFields(Visitor);

  Visitor->Apply(Context);
}

/// getDescString - Return a string used to compose global names and labels.
///
const char *BlockDesc::getDescString() const {
  return "llvm.dbg.block";
}

/// getTypeString - Return a string used to label this descriptors type.
///
const char *BlockDesc::getTypeString() const {
  return "llvm.dbg.block.type";
}

#ifndef NDEBUG
void BlockDesc::dump() {
  cerr << getDescString() << " "
       << "Version(" << getVersion() << "), "
       << "Tag(" << getTag() << "),"
       << "Context(" << Context << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

DebugInfoDesc *DIDeserializer::Deserialize(Value *V) {
  return Deserialize(getGlobalVariable(V));
}
DebugInfoDesc *DIDeserializer::Deserialize(GlobalVariable *GV) {
  // Handle NULL.
  if (!GV) return NULL;

  // Check to see if it has been already deserialized.
  DebugInfoDesc *&Slot = GlobalDescs[GV];
  if (Slot) return Slot;

  // Get the Tag from the global.
  unsigned Tag = DebugInfoDesc::TagFromGlobal(GV);
  
  // Create an empty instance of the correct sort.
  Slot = DebugInfoDesc::DescFactory(Tag);
  
  // If not a user defined descriptor.
  if (Slot) {
    // Deserialize the fields.
    DIDeserializeVisitor DRAM(*this, GV);
    DRAM.ApplyToFields(Slot);
  }
  
  return Slot;
}

//===----------------------------------------------------------------------===//

/// getStrPtrType - Return a "sbyte *" type.
///
const PointerType *DISerializer::getStrPtrType() {
  // If not already defined.
  if (!StrPtrTy) {
    // Construct the pointer to signed bytes.
    StrPtrTy = PointerType::getUnqual(Type::Int8Ty);
  }
  
  return StrPtrTy;
}

/// getEmptyStructPtrType - Return a "{ }*" type.
///
const PointerType *DISerializer::getEmptyStructPtrType() {
  // If not already defined.
  if (EmptyStructPtrTy) return EmptyStructPtrTy;

  // Construct the pointer to empty structure type.
  const StructType *EmptyStructTy =
    StructType::get(std::vector<const Type*>());

  // Construct the pointer to empty structure type.
  EmptyStructPtrTy = PointerType::getUnqual(EmptyStructTy);
  return EmptyStructPtrTy;
}

/// getTagType - Return the type describing the specified descriptor (via tag.)
///
const StructType *DISerializer::getTagType(DebugInfoDesc *DD) {
  // Attempt to get the previously defined type.
  StructType *&Ty = TagTypes[DD->getTag()];
  
  // If not already defined.
  if (!Ty) {
    // Set up fields vector.
    std::vector<const Type*> Fields;
    // Get types of fields.
    DIGetTypesVisitor GTAM(*this, Fields);
    GTAM.ApplyToFields(DD);

    // Construct structured type.
    Ty = StructType::get(Fields);
    
    // Register type name with module.
    M->addTypeName(DD->getTypeString(), Ty);
  }
  
  return Ty;
}

/// getString - Construct the string as constant string global.
///
Constant *DISerializer::getString(const std::string &String) {
  // Check string cache for previous edition.
  Constant *&Slot = StringCache[String.c_str()];

  // Return Constant if previously defined.
  if (Slot) return Slot;

  // If empty string then use a sbyte* null instead.
  if (String.empty()) {
    Slot = ConstantPointerNull::get(getStrPtrType());
  } else {
    // Construct string as an llvm constant.
    Constant *ConstStr = ConstantArray::get(String);

    // Otherwise create and return a new string global.
    GlobalVariable *StrGV = new GlobalVariable(ConstStr->getType(), true,
                                               GlobalVariable::InternalLinkage,
                                               ConstStr, ".str", M);
    StrGV->setSection("llvm.metadata");

    // Convert to generic string pointer.
    Slot = ConstantExpr::getBitCast(StrGV, getStrPtrType());
  }

  return Slot;
  
}

/// Serialize - Recursively cast the specified descriptor into a GlobalVariable
/// so that it can be serialized to a .bc or .ll file.
GlobalVariable *DISerializer::Serialize(DebugInfoDesc *DD) {
  // Check if the DebugInfoDesc is already in the map.
  GlobalVariable *&Slot = DescGlobals[DD];
  
  // See if DebugInfoDesc exists, if so return prior GlobalVariable.
  if (Slot) return Slot;
  
  // Get the type associated with the Tag.
  const StructType *Ty = getTagType(DD);

  // Create the GlobalVariable early to prevent infinite recursion.
  GlobalVariable *GV = new GlobalVariable(Ty, true, DD->getLinkage(),
                                          NULL, DD->getDescString(), M);
  GV->setSection("llvm.metadata");

  // Insert new GlobalVariable in DescGlobals map.
  Slot = GV;
 
  // Set up elements vector
  std::vector<Constant*> Elements;
  // Add fields.
  DISerializeVisitor SRAM(*this, Elements);
  SRAM.ApplyToFields(DD);

  // Set the globals initializer.
  GV->setInitializer(ConstantStruct::get(Ty, Elements));
  
  return GV;
}

/// addDescriptor - Directly connect DD with existing GV.
void DISerializer::addDescriptor(DebugInfoDesc *DD,
                                 GlobalVariable *GV) {
  DescGlobals[DD] = GV;
}

//===----------------------------------------------------------------------===//

/// Verify - Return true if the GlobalVariable appears to be a valid
/// serialization of a DebugInfoDesc.
bool DIVerifier::Verify(Value *V) {
  return !V || Verify(getGlobalVariable(V));
}
bool DIVerifier::Verify(GlobalVariable *GV) {
  // NULLs are valid.
  if (!GV) return true;
  
  // Check prior validity.
  unsigned &ValiditySlot = Validity[GV];
  
  // If visited before then use old state.
  if (ValiditySlot) return ValiditySlot == Valid;
  
  // Assume validity for the time being (recursion.)
  ValiditySlot = Valid;
  
  // Make sure the global is internal or link once (anchor.)
  if (GV->getLinkage() != GlobalValue::InternalLinkage &&
      GV->getLinkage() != GlobalValue::LinkOnceLinkage) {
    ValiditySlot = Invalid;
    return false;
  }

  // Get the Tag.
  unsigned Tag = DebugInfoDesc::TagFromGlobal(GV);
  
  // Check for user defined descriptors.
  if (Tag == DW_TAG_invalid) {
    ValiditySlot = Valid;
    return true;
  }
  
  // Get the Version.
  unsigned Version = DebugInfoDesc::VersionFromGlobal(GV);
  
  // Check for version mismatch.
  if (Version != LLVMDebugVersion) {
    ValiditySlot = Invalid;
    return false;
  }

  // Construct an empty DebugInfoDesc.
  DebugInfoDesc *DD = DebugInfoDesc::DescFactory(Tag);
  
  // Allow for user defined descriptors.
  if (!DD) return true;
  
  // Get the initializer constant.
  ConstantStruct *CI = cast<ConstantStruct>(GV->getInitializer());
  
  // Get the operand count.
  unsigned N = CI->getNumOperands();
  
  // Get the field count.
  unsigned &CountSlot = Counts[Tag];
  if (!CountSlot) {
    // Check the operand count to the field count
    DICountVisitor CTAM;
    CTAM.ApplyToFields(DD);
    CountSlot = CTAM.getCount();
  }
  
  // Field count must be at most equal operand count.
  if (CountSlot >  N) {
    delete DD;
    ValiditySlot = Invalid;
    return false;
  }
  
  // Check each field for valid type.
  DIVerifyVisitor VRAM(*this, GV);
  VRAM.ApplyToFields(DD);
  
  // Release empty DebugInfoDesc.
  delete DD;
  
  // If fields are not valid.
  if (!VRAM.isValid()) {
    ValiditySlot = Invalid;
    return false;
  }
  
  return true;
}

/// isVerified - Return true if the specified GV has already been
/// verified as a debug information descriptor.
bool DIVerifier::isVerified(GlobalVariable *GV) {
  unsigned &ValiditySlot = Validity[GV];
  if (ValiditySlot) return ValiditySlot == Valid;
  return false;
}

//===----------------------------------------------------------------------===//

DebugScope::~DebugScope() {
  for (unsigned i = 0, N = Scopes.size(); i < N; ++i) delete Scopes[i];
  for (unsigned j = 0, M = Variables.size(); j < M; ++j) delete Variables[j];
}

//===----------------------------------------------------------------------===//

MachineModuleInfo::MachineModuleInfo()
: ImmutablePass((intptr_t)&ID)
, DR()
, VR()
, CompileUnits()
, Directories()
, SourceFiles()
, Lines()
, LabelIDList()
, ScopeMap()
, RootScope(NULL)
, FrameMoves()
, LandingPads()
, Personalities()
, CallsEHReturn(0)
, CallsUnwindInit(0)
{
  // Always emit "no personality" info
  Personalities.push_back(NULL);
}
MachineModuleInfo::~MachineModuleInfo() {

}

/// doInitialization - Initialize the state for a new module.
///
bool MachineModuleInfo::doInitialization() {
  return false;
}

/// doFinalization - Tear down the state after completion of a module.
///
bool MachineModuleInfo::doFinalization() {
  return false;
}

/// BeginFunction - Begin gathering function meta information.
///
void MachineModuleInfo::BeginFunction(MachineFunction *MF) {
  // Coming soon.
}

/// EndFunction - Discard function meta information.
///
void MachineModuleInfo::EndFunction() {
  // Clean up scope information.
  if (RootScope) {
    delete RootScope;
    ScopeMap.clear();
    RootScope = NULL;
  }
  
  // Clean up line info.
  Lines.clear();

  // Clean up frame info.
  FrameMoves.clear();
  
  // Clean up exception info.
  LandingPads.clear();
  TypeInfos.clear();
  FilterIds.clear();
  FilterEnds.clear();
  CallsEHReturn = 0;
  CallsUnwindInit = 0;
}

/// getDescFor - Convert a Value to a debug information descriptor.
///
// FIXME - use new Value type when available.
DebugInfoDesc *MachineModuleInfo::getDescFor(Value *V) {
  return DR.Deserialize(V);
}

/// AnalyzeModule - Scan the module for global debug information.
///
void MachineModuleInfo::AnalyzeModule(Module &M) {
  SetupCompileUnits(M);

  // Insert functions in the llvm.used array into UsedFunctions.
  GlobalVariable *GV = M.getGlobalVariable("llvm.used");
  if (!GV || !GV->hasInitializer()) return;

  // Should be an array of 'i8*'.
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (InitList == 0) return;

  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(InitList->getOperand(i)))
      if (CE->getOpcode() == Instruction::BitCast)
        if (Function *F = dyn_cast<Function>(CE->getOperand(0)))
          UsedFunctions.insert(F);
  }
}

/// SetupCompileUnits - Set up the unique vector of compile units.
///
void MachineModuleInfo::SetupCompileUnits(Module &M) {
  std::vector<CompileUnitDesc *>CU = getAnchoredDescriptors<CompileUnitDesc>(M);
  
  for (unsigned i = 0, N = CU.size(); i < N; i++) {
    CompileUnits.insert(CU[i]);
  }
}

/// getCompileUnits - Return a vector of debug compile units.
///
const UniqueVector<CompileUnitDesc *> MachineModuleInfo::getCompileUnits()const{
  return CompileUnits;
}

/// getGlobalVariablesUsing - Return all of the GlobalVariables that use the
/// named GlobalVariable.
std::vector<GlobalVariable*>
MachineModuleInfo::getGlobalVariablesUsing(Module &M,
                                           const std::string &RootName) {
  return ::getGlobalVariablesUsing(M, RootName);
}

/// RecordSourceLine - Records location information and associates it with a
/// debug label.  Returns a unique label ID used to generate a label and 
/// provide correspondence to the source line list.
unsigned MachineModuleInfo::RecordSourceLine(unsigned Line, unsigned Column,
                                             unsigned Source) {
  unsigned ID = NextLabelID();
  Lines.push_back(SourceLineInfo(Line, Column, Source, ID));
  return ID;
}

/// RecordSource - Register a source file with debug info. Returns an source
/// ID.
unsigned MachineModuleInfo::RecordSource(const std::string &Directory,
                                         const std::string &Source) {
  unsigned DirectoryID = Directories.insert(Directory);
  return SourceFiles.insert(SourceFileInfo(DirectoryID, Source));
}
unsigned MachineModuleInfo::RecordSource(const CompileUnitDesc *CompileUnit) {
  return RecordSource(CompileUnit->getDirectory(),
                      CompileUnit->getFileName());
}

/// RecordRegionStart - Indicate the start of a region.
///
unsigned MachineModuleInfo::RecordRegionStart(Value *V) {
  // FIXME - need to be able to handle split scopes because of bb cloning.
  DebugInfoDesc *ScopeDesc = DR.Deserialize(V);
  DebugScope *Scope = getOrCreateScope(ScopeDesc);
  unsigned ID = NextLabelID();
  if (!Scope->getStartLabelID()) Scope->setStartLabelID(ID);
  return ID;
}

/// RecordRegionEnd - Indicate the end of a region.
///
unsigned MachineModuleInfo::RecordRegionEnd(Value *V) {
  // FIXME - need to be able to handle split scopes because of bb cloning.
  DebugInfoDesc *ScopeDesc = DR.Deserialize(V);
  DebugScope *Scope = getOrCreateScope(ScopeDesc);
  unsigned ID = NextLabelID();
  Scope->setEndLabelID(ID);
  return ID;
}

/// RecordVariable - Indicate the declaration of  a local variable.
///
void MachineModuleInfo::RecordVariable(GlobalValue *GV, unsigned FrameIndex) {
  VariableDesc *VD = cast<VariableDesc>(DR.Deserialize(GV));
  DebugScope *Scope = getOrCreateScope(VD->getContext());
  DebugVariable *DV = new DebugVariable(VD, FrameIndex);
  Scope->AddVariable(DV);
}

/// getOrCreateScope - Returns the scope associated with the given descriptor.
///
DebugScope *MachineModuleInfo::getOrCreateScope(DebugInfoDesc *ScopeDesc) {
  DebugScope *&Slot = ScopeMap[ScopeDesc];
  if (!Slot) {
    // FIXME - breaks down when the context is an inlined function.
    DebugInfoDesc *ParentDesc = NULL;
    if (BlockDesc *Block = dyn_cast<BlockDesc>(ScopeDesc)) {
      ParentDesc = Block->getContext();
    }
    DebugScope *Parent = ParentDesc ? getOrCreateScope(ParentDesc) : NULL;
    Slot = new DebugScope(Parent, ScopeDesc);
    if (Parent) {
      Parent->AddScope(Slot);
    } else if (RootScope) {
      // FIXME - Add inlined function scopes to the root so we can delete
      // them later.  Long term, handle inlined functions properly.
      RootScope->AddScope(Slot);
    } else {
      // First function is top level function.
      RootScope = Slot;
    }
  }
  return Slot;
}

//===-EH-------------------------------------------------------------------===//

/// getOrCreateLandingPadInfo - Find or create an LandingPadInfo for the
/// specified MachineBasicBlock.
LandingPadInfo &MachineModuleInfo::getOrCreateLandingPadInfo
    (MachineBasicBlock *LandingPad) {
  unsigned N = LandingPads.size();
  for (unsigned i = 0; i < N; ++i) {
    LandingPadInfo &LP = LandingPads[i];
    if (LP.LandingPadBlock == LandingPad)
      return LP;
  }
  
  LandingPads.push_back(LandingPadInfo(LandingPad));
  return LandingPads[N];
}

/// addInvoke - Provide the begin and end labels of an invoke style call and
/// associate it with a try landing pad block.
void MachineModuleInfo::addInvoke(MachineBasicBlock *LandingPad,
                                  unsigned BeginLabel, unsigned EndLabel) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.BeginLabels.push_back(BeginLabel);
  LP.EndLabels.push_back(EndLabel);
}

/// addLandingPad - Provide the label of a try LandingPad block.
///
unsigned MachineModuleInfo::addLandingPad(MachineBasicBlock *LandingPad) {
  unsigned LandingPadLabel = NextLabelID();
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.LandingPadLabel = LandingPadLabel;  
  return LandingPadLabel;
}

/// addPersonality - Provide the personality function for the exception
/// information.
void MachineModuleInfo::addPersonality(MachineBasicBlock *LandingPad,
                                       Function *Personality) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.Personality = Personality;

  for (unsigned i = 0; i < Personalities.size(); ++i)
    if (Personalities[i] == Personality)
      return;
  
  Personalities.push_back(Personality);
}

/// addCatchTypeInfo - Provide the catch typeinfo for a landing pad.
///
void MachineModuleInfo::addCatchTypeInfo(MachineBasicBlock *LandingPad,
                                        std::vector<GlobalVariable *> &TyInfo) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  for (unsigned N = TyInfo.size(); N; --N)
    LP.TypeIds.push_back(getTypeIDFor(TyInfo[N - 1]));
}

/// addFilterTypeInfo - Provide the filter typeinfo for a landing pad.
///
void MachineModuleInfo::addFilterTypeInfo(MachineBasicBlock *LandingPad,
                                        std::vector<GlobalVariable *> &TyInfo) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  std::vector<unsigned> IdsInFilter (TyInfo.size());
  for (unsigned I = 0, E = TyInfo.size(); I != E; ++I)
    IdsInFilter[I] = getTypeIDFor(TyInfo[I]);
  LP.TypeIds.push_back(getFilterIDFor(IdsInFilter));
}

/// addCleanup - Add a cleanup action for a landing pad.
///
void MachineModuleInfo::addCleanup(MachineBasicBlock *LandingPad) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.TypeIds.push_back(0);
}

/// TidyLandingPads - Remap landing pad labels and remove any deleted landing
/// pads.
void MachineModuleInfo::TidyLandingPads() {
  for (unsigned i = 0; i != LandingPads.size(); ) {
    LandingPadInfo &LandingPad = LandingPads[i];
    LandingPad.LandingPadLabel = MappedLabel(LandingPad.LandingPadLabel);

    // Special case: we *should* emit LPs with null LP MBB. This indicates
    // "nounwind" case.
    if (!LandingPad.LandingPadLabel && LandingPad.LandingPadBlock) {
      LandingPads.erase(LandingPads.begin() + i);
      continue;
    }

    for (unsigned j=0; j != LandingPads[i].BeginLabels.size(); ) {
      unsigned BeginLabel = MappedLabel(LandingPad.BeginLabels[j]);
      unsigned EndLabel = MappedLabel(LandingPad.EndLabels[j]);

      if (!BeginLabel || !EndLabel) {
        LandingPad.BeginLabels.erase(LandingPad.BeginLabels.begin() + j);
        LandingPad.EndLabels.erase(LandingPad.EndLabels.begin() + j);
        continue;
      }

      LandingPad.BeginLabels[j] = BeginLabel;
      LandingPad.EndLabels[j] = EndLabel;
      ++j;
    }

    // Remove landing pads with no try-ranges.
    if (LandingPads[i].BeginLabels.empty()) {
      LandingPads.erase(LandingPads.begin() + i);
      continue;
    }

    // If there is no landing pad, ensure that the list of typeids is empty.
    // If the only typeid is a cleanup, this is the same as having no typeids.
    if (!LandingPad.LandingPadBlock ||
        (LandingPad.TypeIds.size() == 1 && !LandingPad.TypeIds[0]))
      LandingPad.TypeIds.clear();

    ++i;
  }
}

/// getTypeIDFor - Return the type id for the specified typeinfo.  This is 
/// function wide.
unsigned MachineModuleInfo::getTypeIDFor(GlobalVariable *TI) {
  for (unsigned i = 0, N = TypeInfos.size(); i != N; ++i)
    if (TypeInfos[i] == TI) return i + 1;

  TypeInfos.push_back(TI);
  return TypeInfos.size();
}

/// getFilterIDFor - Return the filter id for the specified typeinfos.  This is
/// function wide.
int MachineModuleInfo::getFilterIDFor(std::vector<unsigned> &TyIds) {
  // If the new filter coincides with the tail of an existing filter, then
  // re-use the existing filter.  Folding filters more than this requires
  // re-ordering filters and/or their elements - probably not worth it.
  for (std::vector<unsigned>::iterator I = FilterEnds.begin(),
       E = FilterEnds.end(); I != E; ++I) {
    unsigned i = *I, j = TyIds.size();

    while (i && j)
      if (FilterIds[--i] != TyIds[--j])
        goto try_next;

    if (!j)
      // The new filter coincides with range [i, end) of the existing filter.
      return -(1 + i);

try_next:;
  }

  // Add the new filter.
  int FilterID = -(1 + FilterIds.size());
  FilterIds.reserve(FilterIds.size() + TyIds.size() + 1);
  for (unsigned I = 0, N = TyIds.size(); I != N; ++I)
    FilterIds.push_back(TyIds[I]);
  FilterEnds.push_back(FilterIds.size());
  FilterIds.push_back(0); // terminator
  return FilterID;
}

/// getPersonality - Return the personality function for the current function.
Function *MachineModuleInfo::getPersonality() const {
  // FIXME: Until PR1414 will be fixed, we're using 1 personality function per
  // function
  return !LandingPads.empty() ? LandingPads[0].Personality : NULL;
}

/// getPersonalityIndex - Return unique index for current personality
/// function. NULL personality function should always get zero index.
unsigned MachineModuleInfo::getPersonalityIndex() const {
  const Function* Personality = NULL;
  
  // Scan landing pads. If there is at least one non-NULL personality - use it.
  for (unsigned i = 0; i != LandingPads.size(); ++i)
    if (LandingPads[i].Personality) {
      Personality = LandingPads[i].Personality;
      break;
    }
  
  for (unsigned i = 0; i < Personalities.size(); ++i) {
    if (Personalities[i] == Personality)
      return i;
  }

  // This should never happen
  assert(0 && "Personality function should be set!");
  return 0;
}

//===----------------------------------------------------------------------===//
/// DebugLabelFolding pass - This pass prunes out redundant labels.  This allows
/// a info consumer to determine if the range of two labels is empty, by seeing
/// if the labels map to the same reduced label.

namespace llvm {

struct DebugLabelFolder : public MachineFunctionPass {
  static char ID;
  DebugLabelFolder() : MachineFunctionPass((intptr_t)&ID) {}

  virtual bool runOnMachineFunction(MachineFunction &MF);
  virtual const char *getPassName() const { return "Label Folder"; }
};

char DebugLabelFolder::ID = 0;

bool DebugLabelFolder::runOnMachineFunction(MachineFunction &MF) {
  // Get machine module info.
  MachineModuleInfo *MMI = getAnalysisToUpdate<MachineModuleInfo>();
  if (!MMI) return false;
  
  // Track if change is made.
  bool MadeChange = false;
  // No prior label to begin.
  unsigned PriorLabel = 0;
  
  // Iterate through basic blocks.
  for (MachineFunction::iterator BB = MF.begin(), E = MF.end();
       BB != E; ++BB) {
    // Iterate through instructions.
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      // Is it a label.
      if (I->isDebugLabel()) {
        // The label ID # is always operand #0, an immediate.
        unsigned NextLabel = I->getOperand(0).getImm();
        
        // If there was an immediate prior label.
        if (PriorLabel) {
          // Remap the current label to prior label.
          MMI->RemapLabel(NextLabel, PriorLabel);
          // Delete the current label.
          I = BB->erase(I);
          // Indicate a change has been made.
          MadeChange = true;
          continue;
        } else {
          // Start a new round.
          PriorLabel = NextLabel;
        }
       } else {
        // No consecutive labels.
        PriorLabel = 0;
      }
      
      ++I;
    }
  }
  
  return MadeChange;
}

FunctionPass *createDebugLabelFoldingPass() { return new DebugLabelFolder(); }

}

