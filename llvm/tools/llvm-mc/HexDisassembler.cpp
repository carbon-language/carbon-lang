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
#include "llvm/Support/SourceMgr.h"
using namespace llvm;

typedef std::vector<std::pair<unsigned char, const char*> > ByteArrayTy;

namespace {
class VectorMemoryObject : public MemoryObject {
private:
  const ByteArrayTy &Bytes;
public:
  VectorMemoryObject(const ByteArrayTy &bytes) : Bytes(bytes) {}
  
  uint64_t getBase() const { return 0; }
  uint64_t getExtent() const { return Bytes.size(); }

  int readByte(uint64_t Addr, uint8_t *Byte) const {
    if (Addr > getExtent())
      return -1;
    *Byte = Bytes[Addr].first;
    return 0;
  }
};
}

static bool PrintInst(const llvm::MCDisassembler &DisAsm,
                      llvm::MCInstPrinter &Printer, const ByteArrayTy &Bytes,
                      SourceMgr &SM) {
  // Wrap the vector in a MemoryObject.
  VectorMemoryObject memoryObject(Bytes);
  
  // Disassemble it to a string and get the size of the instruction.
  MCInst Inst;
  uint64_t Size;
  
  std::string verboseOStr;
  llvm::raw_string_ostream verboseOS(verboseOStr); 
  
  if (!DisAsm.getInstruction(Inst, Size, memoryObject, 0, verboseOS)) {
    SM.PrintMessage(SMLoc::getFromPointer(Bytes[0].second),
                    "invalid instruction encoding", "error");
    errs() << "Diagnostic log:" << '\n';
    errs() << verboseOS.str() << '\n';
    return true;
  }
  
  Printer.printInst(&Inst);
  outs() << "\n";
  
  // If the disassembled instruction was smaller than the number of bytes we
  // read, reject the excess bytes.
  if (Bytes.size() != Size) {
    SM.PrintMessage(SMLoc::getFromPointer(Bytes[Size].second),
                    "excess data detected in input", "error");
    return true;
  }
  
  return false;
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
    errs() << "error: no instruction printer for target " << Triple << '\n';
    return -1;
  }
  
  bool ErrorOccurred = false;
  
  SourceMgr SM;
  SM.AddNewSourceBuffer(&Buffer, SMLoc());
  
  // Convert the input to a vector for disassembly.
  ByteArrayTy ByteArray;
  
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
        ErrorOccurred |= PrintInst(*DisAsm, *InstPrinter, ByteArray, SM);
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
      // If we have an error, print it and skip to the end of line.
      SM.PrintMessage(SMLoc::getFromPointer(Value.data()),
                                 "invalid input token", "error");
      ErrorOccurred = true;
      Str = Str.substr(Str.find('\n'));
      ByteArray.clear();
      continue;
    }
    
    ByteArray.push_back(std::make_pair((unsigned char)ByteVal, Value.data()));
    Str = Str.substr(Next);
  }
  
  if (!ByteArray.empty())
    ErrorOccurred |= PrintInst(*DisAsm, *InstPrinter, ByteArray, SM);
    
  return ErrorOccurred;
}
