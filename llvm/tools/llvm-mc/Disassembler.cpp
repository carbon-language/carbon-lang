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
#include "../../lib/MC/MCDisassembler/EDDisassembler.h"
#include "../../lib/MC/MCDisassembler/EDInst.h"
#include "../../lib/MC/MCDisassembler/EDOperand.h"
#include "../../lib/MC/MCDisassembler/EDToken.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
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
    } else {
      SM.PrintMessage(SMLoc::getFromPointer(Bytes[Index].second),
                      "invalid instruction encoding", "warning");
      if (Size == 0)
        Size = 1; // skip illegible bytes
    }
  }
  
  return false;
}

static bool ByteArrayFromString(ByteArrayTy &ByteArray, 
                                StringRef &Str, 
                                SourceMgr &SM) {
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
      if (Str[0] == '\n') {
        Str = Str.substr(1);
      } else {
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
      Str = Str.substr(Str.find('\n'));
      ByteArray.clear();
      continue;
    }
    
    ByteArray.push_back(std::make_pair((unsigned char)ByteVal, Value.data()));
    Str = Str.substr(Next);
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
  
  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  OwningPtr<MCInstPrinter> IP(T.createMCInstPrinter(AsmPrinterVariant,
                                                    *AsmInfo));
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
  
  ErrorOccurred |= ByteArrayFromString(ByteArray, Str, SM);
  
  if (!ByteArray.empty())
    ErrorOccurred |= PrintInsts(*DisAsm, *IP, ByteArray, SM);
    
  return ErrorOccurred;
}

static int byteArrayReader(uint8_t *B, uint64_t A, void *Arg) {
  ByteArrayTy &ByteArray = *((ByteArrayTy*)Arg);
  
  if (A >= ByteArray.size())
    return -1;
  
  *B = ByteArray[A].first;
  
  return 0;
}

static int verboseEvaluator(uint64_t *V, unsigned R, void *Arg) {
  EDDisassembler &disassembler = *((EDDisassembler *)Arg);
  
  if (const char *regName = disassembler.nameWithRegisterID(R))
    outs() << "[" << regName << "/" << R << "]";
  
  if (disassembler.registerIsStackPointer(R))
    outs() << "(sp)";
  if (disassembler.registerIsProgramCounter(R))
    outs() << "(pc)";
  
  *V = 0;
  return 0;
}

int Disassembler::disassembleEnhanced(const std::string &TS, 
                                      MemoryBuffer &Buffer) {
  ByteArrayTy ByteArray;
  StringRef Str = Buffer.getBuffer();
  SourceMgr SM;
  
  SM.AddNewSourceBuffer(&Buffer, SMLoc());
  
  if (ByteArrayFromString(ByteArray, Str, SM)) {
    return -1;
  }
  
  Triple T(TS);
  EDDisassembler::AssemblySyntax AS;
  
  switch (T.getArch()) {
  default:
    errs() << "error: no default assembly syntax for " << TS.c_str() << "\n";
    return -1;
  case Triple::arm:
  case Triple::thumb:
    AS = EDDisassembler::kEDAssemblySyntaxARMUAL;
    break;
  case Triple::x86:
  case Triple::x86_64:
    AS = EDDisassembler::kEDAssemblySyntaxX86ATT;
    break;
  }
  
  EDDisassembler::initialize();
  EDDisassembler *disassembler =
    EDDisassembler::getDisassembler(TS.c_str(), AS);
  
  if (disassembler == 0) {
    errs() << "error: couldn't get disassembler for " << TS << '\n';
    return -1;
  }
  
  EDInst *inst =
    disassembler->createInst(byteArrayReader, 0, &ByteArray);
                             
  if (inst == 0) {
    errs() << "error: Didn't get an instruction\n";
    return -1;
  }
  
  unsigned numTokens = inst->numTokens();
  if ((int)numTokens < 0) {
    errs() << "error: couldn't count the instruction's tokens\n";
    return -1;
  }
  
  for (unsigned tokenIndex = 0; tokenIndex != numTokens; ++tokenIndex) {
    EDToken *token;
    
    if (inst->getToken(token, tokenIndex)) {
      errs() << "error: Couldn't get token\n";
      return -1;
    }
    
    const char *buf;
    if (token->getString(buf)) {
      errs() << "error: Couldn't get string for token\n";
      return -1;
    }
    
    outs() << '[';
    int operandIndex = token->operandID();
    
    if (operandIndex >= 0)
      outs() << operandIndex << "-";
    
    switch (token->type()) {
    default: outs() << "?"; break;
    case EDToken::kTokenWhitespace: outs() << "w"; break;
    case EDToken::kTokenPunctuation: outs() << "p"; break;
    case EDToken::kTokenOpcode: outs() << "o"; break;
    case EDToken::kTokenLiteral: outs() << "l"; break;
    case EDToken::kTokenRegister: outs() << "r"; break;
    }
    
    outs() << ":" << buf;
  
    if (token->type() == EDToken::kTokenLiteral) {
      outs() << "=";
      if (token->literalSign())
        outs() << "-";
      uint64_t absoluteValue;
      if (token->literalAbsoluteValue(absoluteValue)) {
        errs() << "error: Couldn't get the value of a literal token\n";
        return -1;
      }
      outs() << absoluteValue;
    } else if (token->type() == EDToken::kTokenRegister) {
      outs() << "=";
      unsigned regID;
      if (token->registerID(regID)) {
        errs() << "error: Couldn't get the ID of a register token\n";
        return -1;
      }
      outs() << "r" << regID;
    }
    
    outs() << "]";
  }
  
  outs() << " ";
    
  if (inst->isBranch())
    outs() << "<br> ";
  if (inst->isMove())
    outs() << "<mov> ";
  
  unsigned numOperands = inst->numOperands();
  
  if ((int)numOperands < 0) {
    errs() << "error: Couldn't count operands\n";
    return -1;
  }
  
  for (unsigned operandIndex = 0; operandIndex != numOperands; ++operandIndex) {
    outs() << operandIndex << ":";
    
    EDOperand *operand;
    if (inst->getOperand(operand, operandIndex)) {
      errs() << "error: couldn't get operand\n";
      return -1;
    }
    
    uint64_t evaluatedResult;
    evaluatedResult = operand->evaluate(evaluatedResult, verboseEvaluator,
                                        disassembler);
    outs() << "=" << evaluatedResult << " ";
  }
  
  outs() << '\n';
  
  return 0;
}

