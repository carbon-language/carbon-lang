//===-- MachineCodeEmitter.cpp - Implement the MachineCodeEmitter itf -----===//
//
// This file implements the MachineCodeEmitter interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Function.h"
#include <iostream>
#include <fstream>

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

    void startFunctionStub(const Function &F, unsigned StubSize) {
      std::cout << "\n--- Function stub for function: " << F.getName() << "\n";
    }
    void *finishFunctionStub(const Function &F) {
      std::cout << "\n";
      return 0;
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

namespace {
  class FilePrinterMachineCodeEmitter : public MachineCodeEmitter {
    std::ofstream f, actual;
    std::ostream &o;
    MachineCodeEmitter *MCE;
    unsigned counter;
    bool mustClose;
    unsigned values[4];
    
  public:
    FilePrinterMachineCodeEmitter() :
      f("lli.out"), o(f), counter(0), mustClose(true)
    {
      if (! f.good()) {
        std::cerr << "Cannot open 'lli.out' for writing\n";
        abort();
      }
      openActual();
    }

    FilePrinterMachineCodeEmitter(MachineCodeEmitter &M, std::ostream &os) :
      o(os), MCE(&M), counter(0)
    {
      FilePrinterMachineCodeEmitter();
      mustClose = false;
      openActual();
    }

    ~FilePrinterMachineCodeEmitter() { 
      o << "\n";
      actual.close();
      if (mustClose) f.close();
    }

    void openActual() {
      actual.open("lli.actual.obj");
      if (! actual.good()) {
        std::cerr << "Cannot open 'lli.actual.obj' for writing\n";
        abort();
      }
    }

    void startFunction(MachineFunction &F) {
      // resolve any outstanding calls
      if (MCE) MCE->startFunction(F);
    }
    void finishFunction(MachineFunction &F) {
      if (MCE) MCE->finishFunction(F);
    }

    void startBasicBlock(MachineBasicBlock &BB) {
      // if any instructions were waiting for the address of this block,
      // let them fix their addresses now
      if (MCE) MCE->startBasicBlock(BB);
    }

    void startFunctionStub(const Function &F, unsigned StubSize) {
      //
      if (MCE) MCE->startFunctionStub(F, StubSize);
    }

    void *finishFunctionStub(const Function &F) {
      if (MCE) return MCE->finishFunctionStub(F);
      else return 0;
    }
    
    void emitByte(unsigned char B) {
      if (MCE) MCE->emitByte(B);

      values[counter] = (unsigned int) B;
      if (++counter % 4 == 0 && counter != 0) {
        o << std::hex;
        for (unsigned i=0; i<4; ++i) {
          if (values[i] < 16) o << "0";
          o << values[i] << " ";
          actual << values[i];
        }
        actual.flush();

        o << std::dec << "\t";
        for (unsigned i=0; i<4; ++i) {
          for (int j=7; j>=0; --j) {
            o << ((values[i] >> j) & 1);
          }
          o << " ";
        }

        o << "\n";

        unsigned instr = 0;
        for (unsigned i=0; i<4; ++i)
          instr |= values[i] << (i*8);

        o << "--- * --- * --- * --- * ---\n";
        counter %= 4;
      }
    }
    void emitPCRelativeDisp(Value *V) {
      // put block in mapping BB -> { instr, address }. when BB is beginning to
      // output, find instr, set disp, overwrite instr at addr using the
      // unsigned value gotten from emitter
    }

    void emitGlobalAddress(GlobalValue *V, bool isPCRelative) {
      if (MCE) MCE->emitGlobalAddress(V, isPCRelative);
    }
    void emitGlobalAddress(const std::string &Name, bool isPCRelative) {
      if (MCE) MCE->emitGlobalAddress(Name, isPCRelative);
    }

    void emitFunctionConstantValueAddress(unsigned ConstantNum, int Offset) {
      if (MCE) MCE->emitFunctionConstantValueAddress(ConstantNum, Offset);
    }
  };
}

MachineCodeEmitter *MachineCodeEmitter::createFilePrinterMachineCodeEmitter(MachineCodeEmitter &MCE) {
  return new FilePrinterMachineCodeEmitter(MCE, std::cerr);
}
