//===-- llvm/CodeGen/SSARegMap.h --------------------------------*- C++ -*-===//
// 
// Map register numbers to register classes that are correctly sized (typed) to
// hold the information. Assists register allocation. Contained by
// MachineFunction, should be deleted by register allocator when it is no
// longer needed.
//   
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SSAREGMAP_H
#define LLVM_CODEGEN_SSAREGMAP_H

#include "llvm/Target/MRegisterInfo.h"

class TargetRegisterClass;

class SSARegMap {
  std::vector<const TargetRegisterClass*> RegClassMap;

  unsigned rescale(unsigned Reg) { 
    return Reg - MRegisterInfo::FirstVirtualRegister;
  }

 public:
  SSARegMap() {}

  const TargetRegisterClass* getRegClass(unsigned Reg) {
    unsigned actualReg = rescale(Reg);
    assert(actualReg < RegClassMap.size() && "Register out of bounds");
    return RegClassMap[actualReg];
  }

  void addRegMap(unsigned Reg, const TargetRegisterClass* RegClass) {
    assert(rescale(Reg) == RegClassMap.size() && 
           "Register mapping not added in sequential order!");
    RegClassMap.push_back(RegClass);
  }
};

#endif
