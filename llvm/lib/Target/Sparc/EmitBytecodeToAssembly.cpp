//===-- EmitBytecodeToAssembly.cpp - Emit bytecode to Sparc .s File --------==//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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

namespace llvm {

using std::ostream;

namespace {

  // sparcasmbuf - stream buf for encoding output bytes as .byte directives for
  // the sparc assembler.
  //
  class sparcasmbuf : public std::streambuf {
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
  class osparcasmstream : public std::ostream {
    sparcasmbuf sb;
  public:
    typedef char           char_type;
    typedef int            int_type;
    typedef std::streampos pos_type;
    typedef std::streamoff off_type;

    explicit osparcasmstream(std::ostream &On) : std::ostream(&sb), sb(On) { }

    sparcasmbuf *rdbuf() const {
      return const_cast<sparcasmbuf*>(&sb);
    }
  };

  static void writePrologue (std::ostream &Out, const std::string &comment,
			     const std::string &symName) {
    // Prologue:
    // Output a comment describing the object.
    Out << "!" << comment << "\n";   
    // Switch the current section to .rodata in the assembly output:
    Out << "\t.section \".rodata\"\n\t.align 8\n";  
    // Output a global symbol naming the object:
    Out << "\t.global " << symName << "\n";    
    Out << "\t.type " << symName << ",#object\n"; 
    Out << symName << ":\n"; 
  }

  static void writeEpilogue (std::ostream &Out, const std::string &symName) {
    // Epilogue:
    // Output a local symbol marking the end of the object:
    Out << ".end_" << symName << ":\n";    
    // Output size directive giving the size of the object:
    Out << "\t.size " << symName << ", .end_" << symName << "-" << symName
	<< "\n";
  }

  // SparcBytecodeWriter - Write bytecode out to a stream that is sparc'ified
  class SparcBytecodeWriter : public Pass {
    std::ostream &Out;
  public:
    SparcBytecodeWriter(std::ostream &out) : Out(out) {}

    const char *getPassName() const { return "Emit Bytecode to Sparc Assembly";}
    
    virtual bool run(Module &M) {
      // Write an object containing the bytecode to the SPARC assembly stream
      writePrologue (Out, "LLVM BYTECODE OUTPUT", "LLVMBytecode");
      osparcasmstream OS(Out);
      WriteBytecodeToFile(&M, OS);
      writeEpilogue (Out, "LLVMBytecode");

      // Write an object containing its length as an integer to the
      // SPARC assembly stream
      writePrologue (Out, "LLVM BYTECODE LENGTH", "llvm_length");
      Out <<"\t.word\t.end_LLVMBytecode-LLVMBytecode\n"; 
      writeEpilogue (Out, "llvm_length");

      return false;
    }
  };
}  // end anonymous namespace

Pass *createBytecodeAsmPrinterPass(std::ostream &Out) {
  return new SparcBytecodeWriter(Out);
}

} // End llvm namespace
