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
#include "llvm/Intrinsics.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;

// Handle the Pass registration stuff necessary to use TargetData's.
namespace {
  RegisterPass<MachineDebugInfo> X("machinedebuginfo", "Debug Information");
}

//===----------------------------------------------------------------------===//

/// getGlobalVariablesUsing - Return all of the global variables which have the
/// specified value in their initializer somewhere.
static void
getGlobalVariablesUsing(Value *V, std::vector<GlobalVariable*> &Result) {
  // Scan though value users.
  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(*I)) {
      // If the user is a global variable then add to result.
      Result.push_back(GV);
    } else if (Constant *C = dyn_cast<Constant>(*I)) {
      // If the user is a constant variable then scan its users
      getGlobalVariablesUsing(C, Result);
    }
  }
}

/// getGlobalVariablesUsing - Return all of the global variables that use the
/// named global variable.
static std::vector<GlobalVariable*>
getGlobalVariablesUsing(Module &M, const std::string &RootName) {
  std::vector<GlobalVariable*> Result;  // Global variables matching criteria.

  // Get the global variable root.
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
const static std::string getStringValue(Value *V, unsigned Offset = 0) {
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

/// getGlobalValue - Return either a direct or cast Global value.
///
static GlobalVariable *getGlobalValue(Value *V) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    return GV;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    return CE->getOpcode() == Instruction::Cast ?  dyn_cast<GlobalVariable>(V)
                                                :  NULL;
  }
  return NULL;
}

  
//===----------------------------------------------------------------------===//

DebugInfoWrapper::DebugInfoWrapper(GlobalVariable *G)
: GV(G)
, IC(dyn_cast<ConstantStruct>(GV->getInitializer())) {
  assert(IC && "llvm.db.global is missing structured constant");
}
  
//===----------------------------------------------------------------------===//

CompileUnitWrapper::CompileUnitWrapper(GlobalVariable *G)
: DebugInfoWrapper(G)
{
  // FIXME - should probably ease up on the number of operands (version.)
  assert(IC->getNumOperands() == N_op &&
         "Compile unit does not have correct number of operands");
}

/// getTag - Return the compile unit's tag number.  Currently should be 
/// DW_TAG_variable.
unsigned CompileUnitWrapper::getTag() const {
  return cast<ConstantUInt>(IC->getOperand(Tag_op))->getValue();
}

/// isCorrectDebugVersion - Return true if is the correct llvm debug version.
/// Currently the value is 0 (zero.)  If the value is is not correct then
/// ignore all debug information.
bool CompileUnitWrapper::isCorrectDebugVersion() const {
  return cast<ConstantUInt>(IC->getOperand(Version_op))->getValue();
}

/// getLanguage - Return the compile unit's language number (ex. DW_LANG_C89.)
///
unsigned CompileUnitWrapper::getLanguage() const {
  return cast<ConstantUInt>(IC->getOperand(Language_op))->getValue();
}

/// getFileName - Return the compile unit's file name.
///
const std::string CompileUnitWrapper::getFileName() const {
  return getStringValue(IC->getOperand(FileName_op));
}

/// getDirectory - Return the compile unit's file directory.
///
const std::string CompileUnitWrapper::getDirectory() const {
  return getStringValue(IC->getOperand(Directory_op));
}
  
/// getProducer - Return the compile unit's generator name.
///
const std::string CompileUnitWrapper::getProducer() const {
  return getStringValue(IC->getOperand(Producer_op));
}

//===----------------------------------------------------------------------===//

GlobalWrapper::GlobalWrapper(GlobalVariable *G)
: DebugInfoWrapper(G)
{
  // FIXME - should probably ease up on the number of operands (version.)
  assert(IC->getNumOperands() == N_op &&
         "Global does not have correct number of operands");
}

/// getTag - Return the global's tag number.  Currently should be 
/// DW_TAG_variable or DW_TAG_subprogram.
unsigned GlobalWrapper::getTag() const {
  return cast<ConstantUInt>(IC->getOperand(Tag_op))->getValue();
}

/// getContext - Return the "lldb.compile_unit" context global.
///
GlobalVariable *GlobalWrapper::getContext() const {
  return getGlobalValue(IC->getOperand(Context_op));
}

/// getName - Return the name of the global.
///
const std::string GlobalWrapper::getName() const {
  return getStringValue(IC->getOperand(Name_op));
}

/// getType - Return the type of the global.
///
const GlobalVariable *GlobalWrapper::getType() const {
  return getGlobalValue(IC->getOperand(Type_op));
}

/// isStatic - Return true if the global is static.
///
bool GlobalWrapper::isStatic() const {
  return cast<ConstantBool>(IC->getOperand(Static_op))->getValue();
}

/// isDefinition - Return true if the global is a definition.
///
bool GlobalWrapper::isDefinition() const {
  return dyn_cast<ConstantBool>(IC->getOperand(Definition_op))->getValue();
}

/// getGlobalVariable - Return the global variable (tag == DW_TAG_variable.)
///
GlobalVariable *GlobalWrapper::getGlobalVariable() const {
  return getGlobalValue(IC->getOperand(GlobalVariable_op));
}

//===----------------------------------------------------------------------===//


MachineDebugInfo::MachineDebugInfo()
: CompileUnits()
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
  SetupCompileUnits(M);
}

/// SetupCompileUnits - Set up the unique vector of compile units.
///
void MachineDebugInfo::SetupCompileUnits(Module &M) {
  // Get vector of all debug compile units.
  std::vector<GlobalVariable*> Globals =
                       getGlobalVariablesUsing(M, "llvm.dbg.translation_units");
  
  // Scan all compile unit globals.
  for (unsigned i = 0, N = Globals.size(); i < N; ++i) {
    // Create wrapper for compile unit.
    CompileUnitWrapper CUI(Globals[i]);
    // Add to result.
    if (CUI.isCorrectDebugVersion()) CompileUnits.insert(CUI);
  }
  
  // If there any bad compile units then suppress debug information
  if (CompileUnits.size() != Globals.size()) CompileUnits.reset();
}

/// getCompileUnits - Return a vector of debug compile units.
///
const UniqueVector<CompileUnitWrapper> MachineDebugInfo::getCompileUnits()const{
  return CompileUnits;
}

/// getGlobalVariables - Return a vector of debug global variables.
///
std::vector<GlobalWrapper> MachineDebugInfo::getGlobalVariables(Module &M) {
  // Get vector of all debug global objects.
  std::vector<GlobalVariable*> Globals =
                                 getGlobalVariablesUsing(M, "llvm.dbg.globals");
  
  // Accumulation of global variables.
  std::vector<GlobalWrapper> GlobalVariables;

// FIXME - skip until globals have new format
#if 0
  // Scan all globals.
  for (unsigned i = 0, N = Globals.size(); i < N; ++i) {
    // Create wrapper for global.
    GlobalWrapper GW(Globals[i]);
    // If the global is a variable then add to result.
    if (GW.getTag() == DW_TAG_variable) GlobalVariables.push_back(GW);
  }
#endif

  return GlobalVariables;
}

