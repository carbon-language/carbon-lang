//===-- llvm/CodeGen/MachineDebugInfo.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDebugInfo.h"

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/Dwarf.h"

#include <iostream>

using namespace llvm;

// Handle the Pass registration stuff necessary to use TargetData's.
namespace {
  RegisterPass<MachineDebugInfo> X("machinedebuginfo", "Debug Information");
}

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

  // Get the GlobalVariable root.
  GlobalVariable *UseRoot = M.getGlobalVariable(RootName,
                                   StructType::get(std::vector<const Type*>()));

  // If present and linkonce then scan for users.
  if (UseRoot && UseRoot->hasLinkOnceLinkage()) {
    getGlobalVariablesUsing(UseRoot, Result);
  }
  
  return Result;
}
  
/// getStringValue - Turn an LLVM constant pointer that eventually points to a
/// global into a string value.  Return an empty string if we can't do it.
///
static const std::string getStringValue(Value *V, unsigned Offset = 0) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    if (GV->hasInitializer() && isa<ConstantArray>(GV->getInitializer())) {
      ConstantArray *Init = cast<ConstantArray>(GV->getInitializer());
      if (Init->isString()) {
        std::string Result = Init->getAsString();
        if (Offset < Result.size()) {
          // If we are pointing INTO The string, erase the beginning...
          Result.erase(Result.begin(), Result.begin()+Offset);

          // Take off the null terminator, and any string fragments after it.
          std::string::size_type NullPos = Result.find_first_of((char)0);
          if (NullPos != std::string::npos)
            Result.erase(Result.begin()+NullPos, Result.end());
          return Result;
        }
      }
    }
  } else if (Constant *C = dyn_cast<Constant>(V)) {
    if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
      return getStringValue(GV, Offset);
    else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        // Turn a gep into the specified offset.
        if (CE->getNumOperands() == 3 &&
            cast<Constant>(CE->getOperand(1))->isNullValue() &&
            isa<ConstantInt>(CE->getOperand(2))) {
          return getStringValue(CE->getOperand(0),
                   Offset+cast<ConstantInt>(CE->getOperand(2))->getRawValue());
        }
      }
    }
  }
  return "";
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
    if (CE->getOpcode() == Instruction::Cast) {
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
    if (CE->getOpcode() == Instruction::Cast) {
      return isa<GlobalVariable>(CE->getOperand(0));
    }
  }
  return false;
}

/// getUIntOperand - Return ith operand if it is an unsigned integer.
///
static ConstantUInt *getUIntOperand(GlobalVariable *GV, unsigned i) {
  // Make sure the GlobalVariable has an initializer.
  if (!GV->hasInitializer()) return NULL;
  
  // Get the initializer constant.
  ConstantStruct *CI = dyn_cast<ConstantStruct>(GV->getInitializer());
  if (!CI) return NULL;
  
  // Check if there is at least i + 1 operands.
  unsigned N = CI->getNumOperands();
  if (i >= N) return NULL;

  // Check constant.
  return dyn_cast<ConstantUInt>(CI->getOperand(i));
}

//===----------------------------------------------------------------------===//

/// TagFromGlobal - Returns the Tag number from a debug info descriptor
/// GlobalVariable.  
unsigned DebugInfoDesc::TagFromGlobal(GlobalVariable *GV) {
  ConstantUInt *C = getUIntOperand(GV, 0);
  return C ? (unsigned)C->getValue() : (unsigned)DIInvalid;
}

/// DescFactory - Create an instance of debug info descriptor based on Tag.
/// Return NULL if not a recognized Tag.
DebugInfoDesc *DebugInfoDesc::DescFactory(unsigned Tag) {
  switch (Tag) {
  case DI_TAG_compile_unit:    return new CompileUnitDesc();
  case DI_TAG_global_variable: return new GlobalVariableDesc();
  case DI_TAG_subprogram:      return new SubprogramDesc();
  default: break;
  }
  return NULL;
}

//===----------------------------------------------------------------------===//

/// ApplyToFields - Target the visitor to each field of the debug information
/// descriptor.
void DIVisitor::ApplyToFields(DebugInfoDesc *DD) {
  DD->ApplyToFields(this);
}

//===----------------------------------------------------------------------===//
/// DICountVisitor - This DIVisitor counts all the fields in the supplied debug
/// the supplied DebugInfoDesc.
class DICountVisitor : public DIVisitor {
private:
  unsigned Count;                       // Running count of fields.
  
public:
  DICountVisitor() : DIVisitor(), Count(1) {}
  
  // Accessors.
  unsigned getCount() const { return Count; }
  
  /// Apply - Count each of the fields.
  ///
  virtual void Apply(int &Field)             { ++Count; }
  virtual void Apply(unsigned &Field)        { ++Count; }
  virtual void Apply(bool &Field)            { ++Count; }
  virtual void Apply(std::string &Field)     { ++Count; }
  virtual void Apply(DebugInfoDesc *&Field)  { ++Count; }
  virtual void Apply(GlobalVariable *&Field) { ++Count; }
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
  : DIVisitor()
  , DR(D)
  , I(1)
  , CI(cast<ConstantStruct>(GV->getInitializer()))
  {}
  
  /// Apply - Set the value of each of the fields.
  ///
  virtual void Apply(int &Field) {
    Constant *C = CI->getOperand(I++);
    Field = cast<ConstantSInt>(C)->getValue();
  }
  virtual void Apply(unsigned &Field) {
    Constant *C = CI->getOperand(I++);
    Field = cast<ConstantUInt>(C)->getValue();
  }
  virtual void Apply(bool &Field) {
    Constant *C = CI->getOperand(I++);
    Field = cast<ConstantBool>(C)->getValue();
  }
  virtual void Apply(std::string &Field) {
    Constant *C = CI->getOperand(I++);
    Field = getStringValue(C);
  }
  virtual void Apply(DebugInfoDesc *&Field) {
    Constant *C = CI->getOperand(I++);
    Field = DR.Deserialize(C);
  }
  virtual void Apply(GlobalVariable *&Field) {
    Constant *C = CI->getOperand(I++);
    Field = getGlobalVariable(C);
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
    Elements.push_back(ConstantUInt::get(Type::IntTy, Field));
  }
  virtual void Apply(unsigned &Field) {
    Elements.push_back(ConstantUInt::get(Type::UIntTy, Field));
  }
  virtual void Apply(bool &Field) {
    Elements.push_back(ConstantBool::get(Field));
  }
  virtual void Apply(std::string &Field) {
    Elements.push_back(SR.getString(Field));
  }
  virtual void Apply(DebugInfoDesc *&Field) {
    GlobalVariable *GV = NULL;
    
    // If non-NULL the convert to global.
    if (Field) GV = SR.Serialize(Field);
    
    // FIXME - At some point should use specific type.
    const PointerType *EmptyTy = SR.getEmptyStructPtrType();
    
    if (GV) {
      // Set to pointer to global.
      Elements.push_back(ConstantExpr::getCast(GV, EmptyTy));
    } else {
      // Use NULL.
      Elements.push_back(ConstantPointerNull::get(EmptyTy));
    }
  }
  virtual void Apply(GlobalVariable *&Field) {
    const PointerType *EmptyTy = SR.getEmptyStructPtrType();
    Elements.push_back(ConstantExpr::getCast(Field, EmptyTy));
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
    Fields.push_back(Type::IntTy);
  }
  virtual void Apply(unsigned &Field) {
    Fields.push_back(Type::UIntTy);
  }
  virtual void Apply(bool &Field) {
    Fields.push_back(Type::BoolTy);
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
  , I(1)
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
  virtual void Apply(bool &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isa<ConstantBool>(C);
  }
  virtual void Apply(std::string &Field) {
    Constant *C = CI->getOperand(I++);
    IsValid = IsValid && isStringValue(C);
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
};

//===----------------------------------------------------------------------===//

/// DebugVersionFromGlobal - Returns the version number from a compile unit
/// GlobalVariable.
unsigned CompileUnitDesc::DebugVersionFromGlobal(GlobalVariable *GV) {
  ConstantUInt *C = getUIntOperand(GV, 1);
  return C ? (unsigned)C->getValue() : (unsigned)DIInvalid;
}
  
/// ApplyToFields - Target the visitor to the fields of the CompileUnitDesc.
///
void CompileUnitDesc::ApplyToFields(DIVisitor *Visitor) {
  Visitor->Apply(DebugVersion);
  Visitor->Apply(Language);
  Visitor->Apply(FileName);
  Visitor->Apply(Directory);
  Visitor->Apply(Producer);
  Visitor->Apply(TransUnit);
}

/// TypeString - Return a string used to compose globalnames and labels.
///
const char *CompileUnitDesc::TypeString() const {
  return "compile_unit";
}

#ifndef NDEBUG
void CompileUnitDesc::dump() {
  std::cerr << TypeString() << " "
            << "Tag(" << getTag() << "), "
            << "Language(" << Language << "), "
            << "FileName(\"" << FileName << "\"), "
            << "Directory(\"" << Directory << "\"), "
            << "Producer(\"" << Producer << "\")\n";
}
#endif

//===----------------------------------------------------------------------===//

/// ApplyToFields - Target the visitor to the fields of the GlobalVariableDesc.
///
void GlobalVariableDesc::ApplyToFields(DIVisitor *Visitor) {
  Visitor->Apply(Context);
  Visitor->Apply(Name);
  Visitor->Apply(TransUnit);
  Visitor->Apply(TyDesc);
  Visitor->Apply(IsStatic);
  Visitor->Apply(IsDefinition);
  Visitor->Apply(Global);
}

/// TypeString - Return a string used to compose globalnames and labels.
///
const char *GlobalVariableDesc::TypeString() const {
  return "global_variable";
}

#ifndef NDEBUG
void GlobalVariableDesc::dump() {
  std::cerr << TypeString() << " "
            << "Tag(" << getTag() << "), "
            << "Name(\"" << Name << "\"), "
            << "Type(" << TyDesc << "), "
            << "IsStatic(" << (IsStatic ? "true" : "false") << "), "
            << "IsDefinition(" << (IsDefinition ? "true" : "false") << "), "
            << "Global(" << Global << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

/// ApplyToFields - Target the visitor to the fields of the
/// SubprogramDesc.
void SubprogramDesc::ApplyToFields(DIVisitor *Visitor) {
  Visitor->Apply(Context);
  Visitor->Apply(Name);
  Visitor->Apply(TransUnit);
  Visitor->Apply(TyDesc);
  Visitor->Apply(IsStatic);
  Visitor->Apply(IsDefinition);
  
  // FIXME - Temp variable until restructured.
  GlobalVariable *Tmp;
  Visitor->Apply(Tmp);
}

/// TypeString - Return a string used to compose globalnames and labels.
///
const char *SubprogramDesc::TypeString() const {
  return "subprogram";
}

#ifndef NDEBUG
void SubprogramDesc::dump() {
  std::cerr << TypeString() << " "
            << "Tag(" << getTag() << "), "
            << "Name(\"" << Name << "\"), "
            << "Type(" << TyDesc << "), "
            << "IsStatic(" << (IsStatic ? "true" : "false") << "), "
            << "IsDefinition(" << (IsDefinition ? "true" : "false") << ")\n";
}
#endif

//===----------------------------------------------------------------------===//

DebugInfoDesc *DIDeserializer::Deserialize(Value *V) {
  return Deserialize(cast<GlobalVariable>(V));
}
DebugInfoDesc *DIDeserializer::Deserialize(GlobalVariable *GV) {
  // Check to see if it has been already deserialized.
  DebugInfoDesc *&Slot = GlobalDescs[GV];
  if (Slot) return Slot;

  // Get the Tag from the global.
  unsigned Tag = DebugInfoDesc::TagFromGlobal(GV);
  
  // Get the debug version if a compile unit.
  if (Tag == DI_TAG_compile_unit) {
    DebugVersion = CompileUnitDesc::DebugVersionFromGlobal(GV);
  }
  
  // Create an empty instance of the correct sort.
  Slot = DebugInfoDesc::DescFactory(Tag);
  assert(Slot && "Unknown Tag");
  
  // Deserialize the fields.
  DIDeserializeVisitor DRAM(*this, GV);
  DRAM.ApplyToFields(Slot);
  
  return Slot;
}

//===----------------------------------------------------------------------===//

/// getStrPtrType - Return a "sbyte *" type.
///
const PointerType *DISerializer::getStrPtrType() {
  // If not already defined.
  if (!StrPtrTy) {
    // Construct the pointer to signed bytes.
    StrPtrTy = PointerType::get(Type::SByteTy);
  }
  
  return StrPtrTy;
}

/// getEmptyStructPtrType - Return a "{ }*" type.
///
const PointerType *DISerializer::getEmptyStructPtrType() {
  // If not already defined.
  if (!EmptyStructPtrTy) {
    // Construct the empty structure type.
    const StructType *EmptyStructTy =
                                    StructType::get(std::vector<const Type*>());
    // Construct the pointer to empty structure type.
    EmptyStructPtrTy = PointerType::get(EmptyStructTy);
  }
  
  return EmptyStructPtrTy;
}

/// getTagType - Return the type describing the specified descriptor (via tag.)
///
const StructType *DISerializer::getTagType(DebugInfoDesc *DD) {
  // Attempt to get the previously defined type.
  StructType *&Ty = TagTypes[DD->getTag()];
  
  // If not already defined.
  if (!Ty) {
    // Get descriptor type name.
    const char *TS = DD->TypeString();
    
    // Set up fields vector.
    std::vector<const Type*> Fields;
    // Add tag field.
    Fields.push_back(Type::UIntTy);
    // Get types of remaining fields.
    DIGetTypesVisitor GTAM(*this, Fields);
    GTAM.ApplyToFields(DD);

    // Construct structured type.
    Ty = StructType::get(Fields);
    
    // Construct a name for the type.
    const std::string Name = std::string("lldb.") + DD->TypeString() + ".type";

    // Register type name with module.
    M->addTypeName(Name, Ty);
  }
  
  return Ty;
}

/// getString - Construct the string as constant string global.
///
GlobalVariable *DISerializer::getString(const std::string &String) {
  // Check string cache for previous edition.
  GlobalVariable *&Slot = StringCache[String];
  // return GlobalVariable if previously defined.
  if (Slot) return Slot;
  // Construct strings as an llvm constant.
  Constant *ConstStr = ConstantArray::get(String);
  // Otherwise create and return a new string global.
  return Slot = new GlobalVariable(ConstStr->getType(), true,
                                   GlobalVariable::InternalLinkage,
                                   ConstStr, "str", M);
}

/// Serialize - Recursively cast the specified descriptor into a GlobalVariable
/// so that it can be serialized to a .bc or .ll file.
GlobalVariable *DISerializer::Serialize(DebugInfoDesc *DD) {
  // Check if the DebugInfoDesc is already in the map.
  GlobalVariable *&Slot = DescGlobals[DD];
  
  // See if DebugInfoDesc exists, if so return prior GlobalVariable.
  if (Slot) return Slot;
  
  // Get DebugInfoDesc type Tag.
  unsigned Tag = DD->getTag();
  
  // Construct name.
  const std::string Name = std::string("lldb.") +
                           DD->TypeString();
  
  // Get the type associated with the Tag.
  const StructType *Ty = getTagType(DD);

  // Create the GlobalVariable early to prevent infinite recursion.
  GlobalVariable *GV = new GlobalVariable(Ty, true,
                                          GlobalValue::InternalLinkage,
                                          NULL, Name, M);

  // Insert new GlobalVariable in DescGlobals map.
  Slot = GV;
 
  // Set up elements vector
  std::vector<Constant*> Elements;
  // Add Tag value.
  Elements.push_back(ConstantUInt::get(Type::UIntTy, Tag));
  // Add remaining fields.
  DISerializeVisitor SRAM(*this, Elements);
  SRAM.ApplyToFields(DD);
  
  // Set the globals initializer.
  GV->setInitializer(ConstantStruct::get(Ty, Elements));
  
  return GV;
}

//===----------------------------------------------------------------------===//

/// markVisited - Return true if the GlobalVariable hase been "seen" before.
/// Mark visited otherwise.
bool DIVerifier::markVisited(GlobalVariable *GV) {
  // Check if the GlobalVariable is already in the Visited set.
  std::set<GlobalVariable *>::iterator VI = Visited.lower_bound(GV);
  
  // See if GlobalVariable exists.
  bool Exists = VI != Visited.end() && *VI == GV;

  // Insert in set.
  if (!Exists) Visited.insert(VI, GV);
  
  return Exists;
}

/// Verify - Return true if the GlobalVariable appears to be a valid
/// serialization of a DebugInfoDesc.
bool DIVerifier::Verify(GlobalVariable *GV) {
  // Check if seen before.
  if (markVisited(GV)) return true;
  
  // Get the Tag
  unsigned Tag = DebugInfoDesc::TagFromGlobal(GV);
  if (Tag == DIInvalid) return false;

  // If a compile unit we need the debug version.
  if (Tag == DI_TAG_compile_unit) {
    DebugVersion = CompileUnitDesc::DebugVersionFromGlobal(GV);
    if (DebugVersion == DIInvalid) return false;
  }

  // Construct an empty DebugInfoDesc.
  DebugInfoDesc *DD = DebugInfoDesc::DescFactory(Tag);
  if (!DD) return false;
  
  // Get the initializer constant.
  ConstantStruct *CI = cast<ConstantStruct>(GV->getInitializer());
  
  // Get the operand count.
  unsigned N = CI->getNumOperands();
  
  // Get the field count.
  unsigned &Slot = Counts[Tag];
  if (!Slot) {
    // Check the operand count to the field count
    DICountVisitor CTAM;
    CTAM.ApplyToFields(DD);
    Slot = CTAM.getCount();
  }
  
  // Field count must equal operand count.
  if (Slot != N) {
    delete DD;
    return false;
  }
  
  // Check each field for valid type.
  DIVerifyVisitor VRAM(*this, GV);
  VRAM.ApplyToFields(DD);
  
  // Release empty DebugInfoDesc.
  delete DD;
  
  // Return result of field tests.
  return VRAM.isValid();
}

//===----------------------------------------------------------------------===//


MachineDebugInfo::MachineDebugInfo()
: SR()
, DR()
, VR()
, CompileUnits()
, Directories()
, SourceFiles()
, Lines()
{
  
}
MachineDebugInfo::~MachineDebugInfo() {

}

/// doInitialization - Initialize the debug state for a new module.
///
bool MachineDebugInfo::doInitialization() {
  return false;
}

/// doFinalization - Tear down the debug state after completion of a module.
///
bool MachineDebugInfo::doFinalization() {
  return false;
}

/// AnalyzeModule - Scan the module for global debug information.
///
void MachineDebugInfo::AnalyzeModule(Module &M) {
  SR.setModule(&M);
  DR.setModule(&M);
  SetupCompileUnits(M);
}

/// SetupCompileUnits - Set up the unique vector of compile units.
///
void MachineDebugInfo::SetupCompileUnits(Module &M) {
  SR.setModule(&M);
  DR.setModule(&M);
  // Get vector of all debug compile units.
  std::vector<GlobalVariable*> Globals =
                       getGlobalVariablesUsing(M, "llvm.dbg.translation_units");
  
  // Scan all compile unit globals.
  for (unsigned i = 0, N = Globals.size(); i < N; ++i) {
    // Add compile unit to result.
    CompileUnits.insert(
                    static_cast<CompileUnitDesc *>(DR.Deserialize(Globals[i])));
  }
}

/// getCompileUnits - Return a vector of debug compile units.
///
const UniqueVector<CompileUnitDesc *> MachineDebugInfo::getCompileUnits()const{
  return CompileUnits;
}

/// getGlobalVariables - Return a vector of debug GlobalVariables.
///
std::vector<GlobalVariableDesc *>
MachineDebugInfo::getGlobalVariables(Module &M) {
  SR.setModule(&M);
  DR.setModule(&M);
  // Get vector of all debug global objects.
  std::vector<GlobalVariable*> Globals =
                                 getGlobalVariablesUsing(M, "llvm.dbg.globals");
  
  // Accumulation of GlobalVariables.
  std::vector<GlobalVariableDesc *> GlobalVariables;

  // Scan all globals.
  for (unsigned i = 0, N = Globals.size(); i < N; ++i) {
    GlobalVariable *GV = Globals[i];
    if (DebugInfoDesc::TagFromGlobal(GV) == DI_TAG_global_variable) {
      GlobalVariableDesc *GVD =
                          static_cast<GlobalVariableDesc *>(DR.Deserialize(GV));
      GlobalVariables.push_back(GVD);
    }
  }

  return GlobalVariables;
}

