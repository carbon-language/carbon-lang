//===-- MachineCodeEmitter.cpp - Implement the MachineCodeEmitter itf -----===//
//
// This file implements the MachineCodeEmitter interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Function.h"
#include <iostream>

namespace {
  struct DebugMachineCodeEmitter : public MachineCodeEmitter {
    void startFunction(MachineFunction &F) {
      std::cout << "\n**** Writing machine code for function: "
                << F.getFunction()->getName() << "\n";
    }
    void finishFunction(MachineFunction &F) {
      std::cout << "\n";
    }
    void startBasicBlock(MachineBasicBlock &BB) {
      std::cout << "\n--- Basic Block: " << BB.getBasicBlock()->getName()<<"\n";
    }
    
    void emitByte(unsigned char B) {
      std::cout << "0x" << std::hex << (unsigned int)B << std::dec << " ";
    }
    void emitPCRelativeDisp(Value *V) {
      std::cout << "<disp %" << V->getName() << ": 0xXX 0xXX 0xXX 0xXX> ";
    }
    void emitGlobalAddress(GlobalValue *V, bool isPCRelative) {
      std::cout << "<addr %" << V->getName() << ": 0xXX 0xXX 0xXX 0xXX> ";
    }
    void emitGlobalAddress(const std::string &Name, bool isPCRelative) {
      std::cout << "<addr %" << Name << ": 0xXX 0xXX 0xXX 0xXX> ";
    }

    void emitFunctionConstantValueAddress(unsigned ConstantNum, int Offset) {
      std::cout << "<addr const#" << ConstantNum;
      if (Offset) std::cout << " + " << Offset;
      std::cout << "> ";
    }
  };
}


/// createDebugMachineCodeEmitter - Return a dynamically allocated machine
/// code emitter, which just prints the opcodes and fields out the cout.  This
/// can be used for debugging users of the MachineCodeEmitter interface.
///
MachineCodeEmitter *MachineCodeEmitter::createDebugMachineCodeEmitter() {
  return new DebugMachineCodeEmitter();
}
