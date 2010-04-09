//===- Disassembler.cpp - Disassembler for hex strings --------------------===//
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

#include "Disassembler.h"

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

static bool PrintInsts(const MCDisassembler &DisAsm,
                       MCInstPrinter &Printer, const ByteArrayTy &Bytes,
                       SourceMgr &SM) {
  // Wrap the vector in a MemoryObject.
  VectorMemoryObject memoryObject(Bytes);
  
  // Disassemble it to strings.
  uint64_t Size;
  uint64_t Index;
  
  for (Index = 0; Index < Bytes.size(); Index += Size) {
    MCInst Inst;
    
    if (DisAsm.getInstruction(Inst, Size, memoryObject, Index, 
                               /*REMOVE*/ nulls())) {
      Printer.printInst(&Inst, outs());
      outs() << "\n";
    }
    else {
      SM.PrintMessage(SMLoc::getFromPointer(Bytes[Index].second),
                      "invalid instruction encoding", "warning");
      if (Size == 0)
        Size = 1; // skip illegible bytes
    }
  }
  
  return false;
}

int Disassembler::disassemble(const Target &T, const std::string &Triple,
                              MemoryBuffer &Buffer) {
  // Set up disassembler.
  OwningPtr<const MCAsmInfo> AsmInfo(T.createAsmInfo(Triple));
  
  if (!AsmInfo) {
    errs() << "error: no assembly info for target " << Triple << "\n";
    return -1;
  }
  
  OwningPtr<const MCDisassembler> DisAsm(T.createMCDisassembler());
  if (!DisAsm) {
    errs() << "error: no disassembler for target " << Triple << "\n";
    return -1;
  }
  
  OwningPtr<MCInstPrinter> IP(T.createMCInstPrinter(0, *AsmInfo));
  if (!IP) {
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
    
    // If this is the end of a line or start of a comment, remove the rest of
    // the line.
    if (Str[0] == '\n' || Str[0] == '#') {
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
    ErrorOccurred |= PrintInsts(*DisAsm, *IP, ByteArray, SM);
    
  return ErrorOccurred;
}
