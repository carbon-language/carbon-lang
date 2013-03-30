//===- NVVMReflect.cpp - NVVM Emulate conditional compilation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass replaces occurences of __nvvm_reflect("string") with an
// integer based on -nvvm-reflect-list string=<int> option given to this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define NVVM_REFLECT_FUNCTION "__nvvm_reflect"

using namespace llvm;

namespace llvm { void initializeNVVMReflectPass(PassRegistry &); }

namespace {
class LLVM_LIBRARY_VISIBILITY NVVMReflect : public ModulePass {
private:
  //std::map<std::string, int> VarMap;
  StringMap<int> VarMap;
  typedef std::map<std::string, int>::iterator VarMapIter;
  Function *reflectFunction;

public:
  static char ID;
  NVVMReflect() : ModulePass(ID) {
    VarMap.clear();
    reflectFunction = 0;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const { AU.setPreservesAll(); }
  virtual bool runOnModule(Module &);

  void setVarMap();
};
}

static cl::opt<bool>
NVVMReflectEnabled("nvvm-reflect-enable", cl::init(true),
                   cl::desc("NVVM reflection, enabled by default"));

char NVVMReflect::ID = 0;
INITIALIZE_PASS(NVVMReflect, "nvvm-reflect",
                "Replace occurences of __nvvm_reflect() calls with 0/1", false,
                false)

static cl::list<std::string>
ReflectList("nvvm-reflect-list", cl::value_desc("name=0/1"),
            cl::desc("A list of string=num assignments, where num=0 or 1"),
            cl::ValueRequired);

/// This function does the same operation as perl's split.
/// For example, calling this with ("a=1,b=2,c=0", ",") will
/// return ["a=1", "b=2", "c=0"] in the return std::vector.
static std::vector<std::string>
Tokenize(const std::string &str, const std::string &delim) {
  std::vector<std::string> tokens;

  size_t p0 = 0, p1 = std::string::npos;
  while (p0 != std::string::npos) {
    p1 = str.find_first_of(delim, p0);
    if (p1 != p0) {
      std::string token = str.substr(p0, p1 - p0);
      tokens.push_back(token);
    }
    p0 = str.find_first_not_of(delim, p1);
  }

  return tokens;
}

/// The command line can look as follows :
/// -R a=1,b=2 -R c=3,d=0 -R e=2
/// The strings "a=1,b=2", "c=3,d=0", "e=2" are available in the
/// ReflectList vector. First, each of ReflectList[i] is 'split'
/// using "," as the delimiter. Then each of this part is split
/// using "=" as the delimiter.
void NVVMReflect::setVarMap() {
  for (unsigned i = 0, e = ReflectList.size(); i != e; ++i) {
    //    DEBUG(dbgs() << "Option : "  << ReflectList[i] << std::endl);
    std::vector<std::string> nameValList = Tokenize(ReflectList[i], ",");
    for (unsigned j = 0, ej = nameValList.size(); j != ej; ++j) {
      std::vector<std::string> nameValPair = Tokenize(nameValList[j], "=");
      assert(nameValPair.size() == 2 && "name=val expected");
      std::stringstream valstream(nameValPair[1]);
      int val;
      valstream >> val;
      assert((!(valstream.fail())) && "integer value expected");
      VarMap[nameValPair[0]] = val;
    }
  }
}

bool NVVMReflect::runOnModule(Module &M) {
  if (!NVVMReflectEnabled)
    return false;

  setVarMap();

  reflectFunction = M.getFunction(NVVM_REFLECT_FUNCTION);

  // If reflect function is not used, then there will be
  // no entry in the module.
  if (reflectFunction == 0) {
    return false;
  }

  // Validate _reflect function
  assert(reflectFunction->isDeclaration() &&
         "_reflect function should not have a body");
  assert(reflectFunction->getReturnType()->isIntegerTy() &&
         "_reflect's return type should be integer");

  std::vector<Instruction *> toRemove;

  // Go through the uses of reflectFunction in this Function.
  // Each of them should a CallInst with a ConstantArray argument.
  // First validate that. If the c-string corresponding to the
  // ConstantArray can be found successfully, see if it can be
  // found in VarMap. If so, replace the uses of CallInst with the
  // value found in VarMap. If not, replace the use  with value 0.
  for (Value::use_iterator iter = reflectFunction->use_begin(),
                           iterEnd = reflectFunction->use_end();
       iter != iterEnd; ++iter) {
    assert(isa<CallInst>(*iter) && "Only a call instruction can use _reflect");
    CallInst *reflect = cast<CallInst>(*iter);

    assert((reflect->getNumOperands() == 2) &&
           "Only one operand expect for _reflect function");
    // In cuda, we will have an extra constant-to-generic conversion of
    // the string.
    const Value *conv = reflect->getArgOperand(0);
    assert(isa<CallInst>(conv) && "Expected a const-to-gen conversion");
    const CallInst *convcall = cast<CallInst>(conv);
    const Value *str = convcall->getArgOperand(0);
    assert(isa<ConstantExpr>(str) &&
           "Format of _reflect function not recognized");
    const ConstantExpr *gep = cast<ConstantExpr>(str);

    const Value *sym = gep->getOperand(0);
    assert(isa<Constant>(sym) && "Format of _reflect function not recognized");

    const Constant *symstr = cast<Constant>(sym);

    assert(isa<ConstantDataSequential>(symstr->getOperand(0)) &&
           "Format of _reflect function not recognized");

    assert(cast<ConstantDataSequential>(symstr->getOperand(0))->isCString() &&
           "Format of _reflect function not recognized");

    std::string reflectArg =
        cast<ConstantDataSequential>(symstr->getOperand(0))->getAsString();

    reflectArg = reflectArg.substr(0, reflectArg.size() - 1);
    //    DEBUG(dbgs() << "Arg of _reflect : " << reflectArg << std::endl);

    int reflectVal = 0; // The default value is 0
    if (VarMap.find(reflectArg) != VarMap.end()) {
      reflectVal = VarMap[reflectArg];
    }
    reflect->replaceAllUsesWith(
        ConstantInt::get(reflect->getType(), reflectVal));
    toRemove.push_back(reflect);
  }
  if (toRemove.size() == 0)
    return false;

  for (unsigned i = 0, e = toRemove.size(); i != e; ++i)
    toRemove[i]->eraseFromParent();
  return true;
}
