//===-- MachineCodeEmitter.cpp - Implement the MachineCodeEmitter itf -----===//
//
// This file implements the MachineCodeEmitter interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Function.h"
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
    void emitWord(unsigned W) {
      std::cout << "0x" << std::hex << W << std::dec << " ";
    }

    uint64_t getGlobalValueAddress(GlobalValue *V) { return 0; }
    uint64_t getGlobalValueAddress(const std::string &Name) { return 0; }
    uint64_t getConstantPoolEntryAddress(unsigned Num) { return 0; }
    uint64_t getCurrentPCValue() { return 0; }

    // forceCompilationOf - Force the compilation of the specified function, and
    // return its address, because we REALLY need the address now.
    //
    // FIXME: This is JIT specific!
    //
    virtual uint64_t forceCompilationOf(Function *F) {
      return 0;
    }
  };
}


/// createDebugMachineCodeEmitter - Return a dynamically allocated machine
/// code emitter, which just prints the opcodes and fields out the cout.  This
/// can be used for debugging users of the MachineCodeEmitter interface.
///
MachineCodeEmitter *MachineCodeEmitter::createDebugEmitter() {
  return new DebugMachineCodeEmitter();
}

namespace {
  class FilePrinterEmitter : public MachineCodeEmitter {
    std::ofstream f, actual;
    std::ostream &o;
    MachineCodeEmitter &MCE;
    unsigned counter;
    bool mustClose;
    unsigned values[4];
    
  public:
    FilePrinterEmitter(MachineCodeEmitter &M, std::ostream &os)
      : f("lli.out"), o(os), MCE(M), counter(0), mustClose(false) {
      if (!f.good()) {
        std::cerr << "Cannot open 'lli.out' for writing\n";
        abort();
      }
      openActual();
    }
    
    ~FilePrinterEmitter() { 
      o << "\n";
      actual.close();
      if (mustClose) f.close();
    }

    void openActual() {
      actual.open("lli.actual.obj");
      if (!actual.good()) {
        std::cerr << "Cannot open 'lli.actual.obj' for writing\n";
        abort();
      }
    }

    void startFunction(MachineFunction &F) {
      // resolve any outstanding calls
      MCE.startFunction(F);
    }
    void finishFunction(MachineFunction &F) {
      MCE.finishFunction(F);
    }

    void startFunctionStub(const Function &F, unsigned StubSize) {
      MCE.startFunctionStub(F, StubSize);
    }

    void *finishFunctionStub(const Function &F) {
      return MCE.finishFunctionStub(F);
    }
    
    void emitByte(unsigned char B) {
      MCE.emitByte(B);
      actual << B; actual.flush();

      values[counter] = (unsigned int) B;
      if (++counter % 4 == 0 && counter != 0) {
        o << std::hex;
        for (unsigned i=0; i<4; ++i) {
          if (values[i] < 16) o << "0";
          o << values[i] << " ";
        }

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

    void emitWord(unsigned W) {
      MCE.emitWord(W);
    }
    uint64_t getGlobalValueAddress(GlobalValue *V) {
      return MCE.getGlobalValueAddress(V);
    }
    uint64_t getGlobalValueAddress(const std::string &Name) {
      return MCE.getGlobalValueAddress(Name);
    }
    uint64_t getConstantPoolEntryAddress(unsigned Num) {
      return MCE.getConstantPoolEntryAddress(Num);
    }
    uint64_t getCurrentPCValue() {
      return MCE.getCurrentPCValue();
    }
    // forceCompilationOf - Force the compilation of the specified function, and
    // return its address, because we REALLY need the address now.
    //
    // FIXME: This is JIT specific!
    //
    virtual uint64_t forceCompilationOf(Function *F) {
      return MCE.forceCompilationOf(F);
    }
  };
}

MachineCodeEmitter *
MachineCodeEmitter::createFilePrinterEmitter(MachineCodeEmitter &MCE) {
  return new FilePrinterEmitter(MCE, std::cerr);
}
