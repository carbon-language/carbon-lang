//===-- UserInput.cpp - Interpreter Input Loop support --------------------===//
// 
//  This file implements the interpreter Input I/O loop.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"

// callMainFunction - This is a nasty gross hack that will dissapear when
// callFunction can parse command line options and stuff for us.
//
bool Interpreter::callMainFunction(const std::string &Name,
                                   const std::vector<std::string> &InputArgv) {
  Function *M = getModule().getNamedFunction(Name);
  if (M == 0) {
    std::cerr << "Could not find function '" << Name << "' in module!\n";
    return 1;
  }
  const FunctionType *MT = M->getFunctionType();

  std::vector<GenericValue> Args;
  if (MT->getParamTypes().size() >= 2) {
  PointerType *SPP = PointerType::get(PointerType::get(Type::SByteTy));
  if (MT->getParamTypes()[1] != SPP) {
    CW << "Second argument of '" << Name << "' should have type: '"
       << SPP << "'!\n";
    return true;
  }
  Args.push_back(PTOGV(CreateArgv(InputArgv)));
  }

  if (MT->getParamTypes().size() >= 1) {
  if (!MT->getParamTypes()[0]->isInteger()) {
    std::cout << "First argument of '" << Name << "' should be an integer!\n";
    return true;
  } else {
    GenericValue GV; GV.UIntVal = InputArgv.size();
    Args.insert(Args.begin(), GV);
  }
  }

  callFunction(M, Args);  // Start executing it...

  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  return false;
}
