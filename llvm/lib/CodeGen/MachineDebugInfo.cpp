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
  assert(IC->getNumOperands() == 7 &&
         "Compile unit does not have correct number of operands");
}

/// getTag - Return the compile unit's tag number.  Currently should be 
/// DW_TAG_variable.
unsigned CompileUnitWrapper::getTag() const {
  ConstantUInt *CI = dyn_cast<ConstantUInt>(IC->getOperand(0));
  assert(CI && "Compile unit tag not an unsigned integer");
  return CI->getValue();
}

/// isCorrectDebugVersion - Return true if is the correct llvm debug version.
/// Currently the value is 0 (zero.)  If the value is is not correct then
/// ignore all debug information.
bool CompileUnitWrapper::isCorrectDebugVersion() const {
  ConstantUInt *CI = dyn_cast<ConstantUInt>(IC->getOperand(1));
  assert(CI && "Compile unit debug version not an unsigned integer");
  return CI->getValue() == 0;
}

/// getLanguage - Return the compile unit's language number (ex. DW_LANG_C89.)
///
unsigned CompileUnitWrapper::getLanguage() const {
  ConstantUInt *CI = dyn_cast<ConstantUInt>(IC->getOperand(2));
  assert(CI && "Compile unit language number not an unsigned integer");
  return CI->getValue();
}

/// getFileName - Return the compile unit's file name.
///
const std::string CompileUnitWrapper::getFileName() const {
  return getStringValue(IC->getOperand(3));
}

/// getDirectory - Return the compile unit's file directory.
///
const std::string CompileUnitWrapper::getDirectory() const {
  return getStringValue(IC->getOperand(4));
}
  
/// getProducer - Return the compile unit's generator name.
///
const std::string CompileUnitWrapper::getProducer() const {
  return getStringValue(IC->getOperand(5));
}

//===----------------------------------------------------------------------===//

GlobalWrapper::GlobalWrapper(GlobalVariable *G)
: DebugInfoWrapper(G)
{
  // FIXME - should probably ease up on the number of operands (version.)
  assert(IC->getNumOperands() == 8 &&
         "Global does not have correct number of operands");
}

/// getTag - Return the global's tag number.  Currently should be 
/// DW_TAG_variable or DW_TAG_subprogram.
unsigned GlobalWrapper::getTag() const {
  ConstantUInt *CI = dyn_cast<ConstantUInt>(IC->getOperand(0));
  assert(CI && "Global tag not an unsigned integer");
  return CI->getValue();
}

/// getContext - Return the "lldb.compile_unit" context global.
///
GlobalVariable *GlobalWrapper::getContext() const {
  return dyn_cast<GlobalVariable>(IC->getOperand(1));
}

/// getName - Return the name of the global.
///
const std::string GlobalWrapper::getName() const {
  return getStringValue(IC->getOperand(2));
}

/// getType - Return the type of the global.
///
const GlobalVariable *GlobalWrapper::getType() const {
  return dyn_cast<GlobalVariable>(IC->getOperand(4));
}

/// isStatic - Return true if the global is static.
///
bool GlobalWrapper::isStatic() const {
  ConstantBool *CB = dyn_cast<ConstantBool>(IC->getOperand(5));
  assert(CB && "Global static flag is not boolean");
  return CB->getValue();
}

/// isDefinition - Return true if the global is a definition.
///
bool GlobalWrapper::isDefinition() const {
  ConstantBool *CB = dyn_cast<ConstantBool>(IC->getOperand(6));
  assert(CB && "Global definition flag is not boolean");
  return CB->getValue();
}

/// getGlobalVariable - Return the global variable (tag == DW_TAG_variable.)
///
GlobalVariable *GlobalWrapper::getGlobalVariable() const {
  ConstantExpr *CE = dyn_cast<ConstantExpr>(IC->getOperand(7));
  assert(CE && CE->getOpcode() == Instruction::Cast &&
         "Global location is not a cast of GlobalVariable");
  GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0));
  assert(GV && "Global location is not a cast of GlobalVariable");
  return GV;
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

