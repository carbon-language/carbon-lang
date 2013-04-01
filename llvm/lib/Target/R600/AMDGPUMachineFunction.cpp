#include "AMDGPUMachineFunction.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"

namespace llvm {

const char *AMDGPUMachineFunction::ShaderTypeAttribute = "ShaderType";

AMDGPUMachineFunction::AMDGPUMachineFunction(const MachineFunction &MF) :
    MachineFunctionInfo() {
  AttributeSet Set = MF.getFunction()->getAttributes();
  Attribute A = Set.getAttribute(AttributeSet::FunctionIndex,
                                 ShaderTypeAttribute);

  if (A.isStringAttribute()) {
    StringRef Str = A.getValueAsString();
    if (Str.getAsInteger(0, ShaderType))
      llvm_unreachable("Can't parse shader type!");
  }
}

}
