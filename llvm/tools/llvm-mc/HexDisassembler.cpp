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

int HexDisassembler::disassemble(const Target &target,
                                 const std::string &tripleString,
                                 MemoryBuffer &buffer) {
  // Set up disassembler
  
  llvm::OwningPtr<const llvm::MCAsmInfo> asmInfo
    (target.createAsmInfo(tripleString));
  
  if (!asmInfo) {
    errs() << "error: no assembly info for target " << tripleString << "\n";
    return -1;
  }
  
  llvm::OwningPtr<const llvm::MCDisassembler> disassembler
    (target.createMCDisassembler());
  
  if (!disassembler) {
    errs() << "error: no disassembler for target " << tripleString << "\n";
    return -1;
  }
  
  llvm::MCInstPrinter *instPrinter = target.createMCInstPrinter(0,
                                                                *asmInfo,
                                                                outs());
  
  if (!instPrinter) {
    errs() << "error: no instruction printer for target " << tripleString
      << "\n";
    return -1;
  }
  
  // Convert the input to a vector for disassembly.
  
  std::vector<unsigned char> bytes;
  
  StringRef str = buffer.getBuffer();
  
  while (!str.empty()) {
    if(str.find_first_of("\n") < str.find_first_not_of(" \t\n\r")) {
      printInst(*disassembler, *instPrinter, bytes);
      
      bytes.clear();
    }
    
    // Skip leading space.
    str = str.substr(str.find_first_not_of(" \t\n\r"));
    
    // Get the current token.
    size_t next = str.find_first_of(" \t\n\r");
    
    if(next == (size_t)StringRef::npos)
      break;
    
    StringRef value = str.slice(0, next);
    
    // Convert to a byte and add to the byte vector.
    unsigned byte;
    if (value.getAsInteger(0, byte) || byte > 255) {
      errs() << "warning: invalid input token '" << value << "' of length " 
        << next << "\n";
    }
    else {
      bytes.push_back((unsigned char)byte);
    }
    str = str.substr(next);
  }
  
  if (!bytes.empty())
    printInst(*disassembler, *instPrinter, bytes);
    
  return 0;
}
