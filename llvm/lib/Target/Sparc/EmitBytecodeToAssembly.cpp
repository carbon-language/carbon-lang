//===-- EmitBytecodeToAssembly.cpp - Emit bytecode to Sparc .s File --------==//
//
// This file implements the pass that writes LLVM bytecode as data to a sparc
// assembly file.  The bytecode gets assembled into a special bytecode section
// of the executable for use at runtime later.
//
//===----------------------------------------------------------------------===//

#include "SparcInternals.h"
#include "llvm/Pass.h"
#include "llvm/Bytecode/Writer.h"
#include <iostream>

namespace {

  // sparcasmbuf - stream buf for encoding output bytes as .byte directives for
  // the sparc assembler.
  //
  class sparcasmbuf : public streambuf {
    std::ostream &BaseStr;
  public:
    typedef char           char_type;
    typedef int            int_type;
    typedef std::streampos pos_type;
    typedef std::streamoff off_type;
    
    sparcasmbuf(std::ostream &On) : BaseStr(On) {}

    virtual int_type overflow(int_type C) {
      if (C != EOF)
        BaseStr << "\t.byte " << C << "\n"; // Output C;
      return C;
    }
  };


  // osparcasmstream - Define an ostream implementation that uses a sparcasmbuf
  // as the underlying streambuf to write the data to.  This streambuf formats
  // the output as .byte directives for sparc output.
  //
  class osparcasmstream : public ostream {
    sparcasmbuf sb;
  public:
    typedef char           char_type;
    typedef int            int_type;
    typedef std::streampos pos_type;
    typedef std::streamoff off_type;

    explicit osparcasmstream(ostream &On) : ostream(&sb), sb(On) { }

    sparcasmbuf *rdbuf() const {
      return const_cast<sparcasmbuf*>(&sb);
    }
  };

  // SparcBytecodeWriter - Write bytecode out to a stream that is sparc'ified
  class SparcBytecodeWriter : public Pass {
    std::ostream &Out;
  public:
    SparcBytecodeWriter(std::ostream &out) : Out(out) {}

    const char *getPassName() const { return "Emit Bytecode to Sparc Assembly";}

    virtual bool run(Module &M) {
      // Write bytecode out to the sparc assembly stream
      Out << "\n\n!LLVM BYTECODE OUTPUT\n\t.section \".rodata\"\n\t.align 8\n";
      Out << "\t.global LLVMBytecode\n\t.type LLVMBytecode,#object\n";
      Out << "LLVMBytecode:\n";
      osparcasmstream OS(Out);
      WriteBytecodeToFile(&M, OS);

      Out << ".end_LLVMBytecode:\n";
      Out << "\t.size LLVMBytecode, .end_LLVMBytecode-LLVMBytecode\n\n";
      return false;
    }
  };
}  // end anonymous namespace

Pass *UltraSparc::getEmitBytecodeToAsmPass(std::ostream &Out) {
  return new SparcBytecodeWriter(Out);
}
