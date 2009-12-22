//===- HexDisassembler.cpp - Disassembler for hex strings -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the disassembler of strings of bytes written in
// hexadecimal, from standard input or from a file.
//
//===----------------------------------------------------------------------===//

#include "HexDisassembler.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

using namespace llvm;

class VectorMemoryObject : public MemoryObject {
private:
  const std::vector<unsigned char> &Bytes;
public:
  VectorMemoryObject(const std::vector<unsigned char> &bytes) : 
    Bytes(bytes) {
  }
  
  uint64_t getBase() const {
    return 0;
  }
  
  uint64_t getExtent() const {
    return Bytes.size();
  }

  int readByte(uint64_t addr, uint8_t *byte) const {
    if (addr > getExtent())
      return -1;
    else
      *byte = Bytes[addr];
    
    return 0;
  }
};

void printInst(const llvm::MCDisassembler &disassembler,
               llvm::MCInstPrinter &instPrinter,
               const std::vector<unsigned char> &bytes) {
  // Wrap the vector in a MemoryObject.
  
  VectorMemoryObject memoryObject(bytes);
  
  // Disassemble it.
  
  MCInst inst;
  uint64_t size;
  
  std::string verboseOStr;
  llvm::raw_string_ostream verboseOS(verboseOStr); 
  
  if (disassembler.getInstruction(inst, 
                                  size, 
                                  memoryObject, 
                                  0, 
                                  verboseOS)) {
    instPrinter.printInst(&inst);
    outs() << "\n";
  }
  else {
    errs() << "error: invalid instruction" << "\n";
    errs() << "Diagnostic log:" << "\n";
    errs() << verboseOStr.c_str() << "\n";
  }
}

int HexDisassembler::disassemble(const Target &T, const std::string &Triple,
                                 MemoryBuffer &Buffer) {
  // Set up disassembler.
  llvm::OwningPtr<const llvm::MCAsmInfo> AsmInfo(T.createAsmInfo(Triple));
  
  if (!AsmInfo) {
    errs() << "error: no assembly info for target " << Triple << "\n";
    return -1;
  }
  
  llvm::OwningPtr<const llvm::MCDisassembler> DisAsm(T.createMCDisassembler());
  if (!DisAsm) {
    errs() << "error: no disassembler for target " << Triple << "\n";
    return -1;
  }
  
  llvm::MCInstPrinter *InstPrinter = T.createMCInstPrinter(0, *AsmInfo, outs());
  
  if (!InstPrinter) {
    errs() << "error: no instruction printer for target " << Triple
      << "\n";
    return -1;
  }
  
  // Convert the input to a vector for disassembly.
  std::vector<unsigned char> ByteArray;
  
  StringRef Str = Buffer.getBuffer();
  
  while (!Str.empty()) {
    // Strip horizontal whitespace.
    if (size_t Pos = Str.find_first_not_of(" \t\r")) {
      Str = Str.substr(Pos);
      continue;
    }
    
    // If this is the end of a line or start of a comment, process the
    // instruction we have so far.
    if (Str[0] == '\n' || Str[0] == '#') {
      // If we have bytes to process, do so.
      if (!ByteArray.empty()) {
        printInst(*DisAsm, *InstPrinter, ByteArray);
        ByteArray.clear();
      }
      
      // Strip to the end of line if we already processed any bytes on this
      // line.  This strips the comment and/or the \n.
      if (Str[0] == '\n')
        Str = Str.substr(1);
      else {
        Str = Str.substr(Str.find_first_of('\n'));
        if (!Str.empty())
          Str = Str.substr(1);
      }
      continue;
    }
    
    // Get the current token.
    size_t Next = Str.find_first_of(" \t\n\r#");
    StringRef Value = Str.substr(0, Next);
    
    // Convert to a byte and add to the byte vector.
    unsigned ByteVal;
    if (Value.getAsInteger(0, ByteVal) || ByteVal > 255) {
      errs() << "warning: invalid input token '" << Value << "' of length " 
             << Next << "\n";
    } else {
      ByteArray.push_back((unsigned char)ByteVal);
    }
    Str = Str.substr(Next);
  }
  
  if (!ByteArray.empty())
    printInst(*DisAsm, *InstPrinter, ByteArray);
    
  return 0;
}
