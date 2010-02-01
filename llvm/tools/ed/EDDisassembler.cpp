//===-EDDisassembler.cpp - LLVM Enhanced Disassembler ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Enhanced Disassembly library's  disassembler class.
// The disassembler is responsible for vending individual instructions according
// to a given architecture and disassembly syntax.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/AsmParser.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetAsmLexer.h"
#include "llvm/Target/TargetAsmParser.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSelect.h"

#include "EDDisassembler.h"
#include "EDInst.h"

#include "../../lib/Target/X86/X86GenEDInfo.inc"

using namespace llvm;

bool EDDisassembler::sInitialized = false;
EDDisassembler::DisassemblerMap_t EDDisassembler::sDisassemblers;

struct InfoMap {
  Triple::ArchType Arch;
  const char *String;
  const InstInfo *Info;
};

static struct InfoMap infomap[] = {
  { Triple::x86,          "i386-unknown-unknown",   instInfoX86 },
  { Triple::x86_64,       "x86_64-unknown-unknown", instInfoX86 },
  { Triple::InvalidArch,  NULL,                     NULL        }
};

/// infoFromArch - Returns the InfoMap corresponding to a given architecture,
///   or NULL if there is an error
///
/// @arg arch - The Triple::ArchType for the desired architecture
static const InfoMap *infoFromArch(Triple::ArchType arch) {
  unsigned int infoIndex;
  
  for (infoIndex = 0; infomap[infoIndex].String != NULL; ++infoIndex) {
    if(arch == infomap[infoIndex].Arch)
      return &infomap[infoIndex];
  }
  
  return NULL;
}

/// getLLVMSyntaxVariant - gets the constant to use to get an assembly printer
///   for the desired assembly syntax, suitable for passing to 
///   Target::createMCInstPrinter()
///
/// @arg arch   - The target architecture
/// @arg syntax - The assembly syntax in sd form
static int getLLVMSyntaxVariant(Triple::ArchType arch,
                                EDAssemblySyntax_t syntax) {
  switch (syntax) {
  default:
    return -1;
  // Mappings below from X86AsmPrinter.cpp
  case kEDAssemblySyntaxX86ATT:
    if (arch == Triple::x86 || arch == Triple::x86_64)
      return 0;
    else
      return -1;
  case kEDAssemblySyntaxX86Intel:
    if (arch == Triple::x86 || arch == Triple::x86_64)
      return 1;
    else
      return -1;
  }
}

#define BRINGUP_TARGET(tgt)           \
  LLVMInitialize##tgt##TargetInfo();  \
  LLVMInitialize##tgt##Target();      \
  LLVMInitialize##tgt##AsmPrinter();  \
  LLVMInitialize##tgt##AsmParser();   \
  LLVMInitialize##tgt##Disassembler();

void EDDisassembler::initialize() {
  if (sInitialized)
    return;
  
  sInitialized = true;
  
  BRINGUP_TARGET(X86)
}

#undef BRINGUP_TARGET

EDDisassembler *EDDisassembler::getDisassembler(Triple::ArchType arch,
                                                EDAssemblySyntax_t syntax) {
  CPUKey key;
  key.Arch = arch;
  key.Syntax = syntax;
  
  EDDisassembler::DisassemblerMap_t::iterator i = sDisassemblers.find(key);
  
  if (i != sDisassemblers.end()) {
    return i->second;
  }
  else {
    EDDisassembler* sdd = new EDDisassembler(key);
    if(!sdd->valid()) {
      delete sdd;
      return NULL;
    }
    
    sDisassemblers[key] = sdd;
    
    return sdd;
  }
  
  return NULL;
}

EDDisassembler *EDDisassembler::getDisassembler(StringRef str,
                                                EDAssemblySyntax_t syntax) {
  Triple triple(str);
  
  return getDisassembler(triple.getArch(), syntax);
}

namespace {
  class EDAsmParser : public MCAsmParser {
    AsmLexer Lexer;
    MCContext Context;
    OwningPtr<MCStreamer> Streamer;
  public:
    // Mandatory functions
    EDAsmParser(const MCAsmInfo &MAI) : Lexer(MAI) {
      Streamer.reset(createNullStreamer(Context));
    }
    virtual ~EDAsmParser() { }
    MCAsmLexer &getLexer() { return Lexer; }
    MCContext &getContext() { return Context; }
    MCStreamer &getStreamer() { return *Streamer; }
    void Warning(SMLoc L, const Twine &Msg) { }
    bool Error(SMLoc L, const Twine &Msg) { return true; }
    const AsmToken &Lex() { return Lexer.Lex(); }
    bool ParseExpression(const MCExpr *&Res, SMLoc &EndLoc) {
      AsmToken token = Lex();
      if(token.isNot(AsmToken::Integer))
        return true;
      Res = MCConstantExpr::Create(token.getIntVal(), Context);
      return false;
    }
    bool ParseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) {
      assert(0 && "I can't ParseParenExpression()s!");
    }
    bool ParseAbsoluteExpression(int64_t &Res) {
      assert(0 && "I can't ParseAbsoluteExpression()s!");
    }
    
    /// setBuffer - loads a buffer into the parser
    /// @arg buf  - The buffer to read tokens from
    void setBuffer(const MemoryBuffer &buf) { Lexer.setBuffer(&buf); }
    /// parseInstName - When the lexer is positioned befor an instruction
    ///   name (with possible intervening whitespace), reads past the name,
    ///   returning 0 on success and -1 on failure
    /// @arg name - A reference to a string that is filled in with the
    ///             instruction name
    /// @arg loc  - A reference to a location that is filled in with the
    ///             position of the instruction name
    int parseInstName(StringRef &name, SMLoc &loc) {
      AsmToken tok = Lexer.Lex();
      if(tok.isNot(AsmToken::Identifier)) {
        return -1;
      }
      name = tok.getString();
      loc = tok.getLoc();
      return 0;
    }
  };
}

EDDisassembler::EDDisassembler(CPUKey &key) : 
  Valid(false), ErrorString(), ErrorStream(ErrorString), Key(key) {
  const InfoMap *infoMap = infoFromArch(key.Arch);
  
  if (!infoMap)
    return;
  
  const char *triple = infoMap->String;
  
  int syntaxVariant = getLLVMSyntaxVariant(key.Arch, key.Syntax);
  
  if (syntaxVariant < 0)
    return;
  
  std::string tripleString(triple);
  std::string errorString;
  
  Tgt = TargetRegistry::lookupTarget(tripleString, 
                                     errorString);
  
  if (!Tgt)
    return;
  
  std::string featureString;
  
  OwningPtr<const TargetMachine>
    targetMachine(Tgt->createTargetMachine(tripleString,
                                           featureString));
  
  const TargetRegisterInfo *registerInfo = targetMachine->getRegisterInfo();
  
  if (!registerInfo)
    return;
  
  AsmInfo.reset(Tgt->createAsmInfo(tripleString));
  
  if (!AsmInfo)
    return;
  
  Disassembler.reset(Tgt->createMCDisassembler());
  
  if (!Disassembler)
    return;
  
  InstString.reset(new std::string);
  InstStream.reset(new raw_string_ostream(*InstString));
  
  InstPrinter.reset(Tgt->createMCInstPrinter(syntaxVariant,
                                                *AsmInfo,
                                                *InstStream));
  
  if (!InstPrinter)
    return;
    
  GenericAsmLexer.reset(new AsmLexer(*AsmInfo));
  SpecificAsmLexer.reset(Tgt->createAsmLexer(*AsmInfo));
  SpecificAsmLexer->InstallLexer(*GenericAsmLexer);
                          
  InstInfos = infoMap->Info;
    
  Valid = true;
}

EDDisassembler::~EDDisassembler() {
  if(!valid())
    return;
}

namespace {
  /// EDMemoryObject - a subclass of MemoryObject that allows use of a callback
  ///   as provided by the sd interface.  See MemoryObject.
  class EDMemoryObject : public llvm::MemoryObject {
  private:
    EDByteReaderCallback Callback;
    void *Arg;
  public:
    EDMemoryObject(EDByteReaderCallback callback,
                   void *arg) : Callback(callback), Arg(arg) { }
    ~EDMemoryObject() { }
    uint64_t getBase() const { return 0x0; }
    uint64_t getExtent() const { return (uint64_t)-1; }
    int readByte(uint64_t address, uint8_t *ptr) const {
      if(!Callback)
        return -1;
      
      if(Callback(ptr, address, Arg))
        return -1;
      
      return 0;
    }
  };
}

EDInst *EDDisassembler::createInst(EDByteReaderCallback byteReader, 
                                   uint64_t address, 
                                   void *arg) {
  EDMemoryObject memoryObject(byteReader, arg);
  
  MCInst* inst = new MCInst;
  uint64_t byteSize;
  
  if (!Disassembler->getInstruction(*inst,
                                    byteSize,
                                    memoryObject,
                                    address,
                                    ErrorStream)) {
    delete inst;
    return NULL;
  }
  else {
    const InstInfo *thisInstInfo = &InstInfos[inst->getOpcode()];
    
    EDInst* sdInst = new EDInst(inst, byteSize, *this, thisInstInfo);
    return sdInst;
  }
}

void EDDisassembler::initMaps(const TargetRegisterInfo &registerInfo) {
  unsigned numRegisters = registerInfo.getNumRegs();
  unsigned registerIndex;
  
  for (registerIndex = 0; registerIndex < numRegisters; ++registerIndex) {
    const char* registerName = registerInfo.get(registerIndex).Name;
    
    RegVec.push_back(registerName);
    RegRMap[registerName] = registerIndex;
  }
  
  if (Key.Arch == Triple::x86 ||
      Key.Arch == Triple::x86_64) {
    stackPointers.insert(registerIDWithName("SP"));
    stackPointers.insert(registerIDWithName("ESP"));
    stackPointers.insert(registerIDWithName("RSP"));
    
    programCounters.insert(registerIDWithName("IP"));
    programCounters.insert(registerIDWithName("EIP"));
    programCounters.insert(registerIDWithName("RIP"));
  }
}

const char *EDDisassembler::nameWithRegisterID(unsigned registerID) const {
  if (registerID >= RegVec.size())
    return NULL;
  else
    return RegVec[registerID].c_str();
}

unsigned EDDisassembler::registerIDWithName(const char *name) const {
  regrmap_t::const_iterator iter = RegRMap.find(std::string(name));
  if (iter == RegRMap.end())
    return 0;
  else
    return (*iter).second;
}

bool EDDisassembler::registerIsStackPointer(unsigned registerID) {
  return (stackPointers.find(registerID) != stackPointers.end());
}

bool EDDisassembler::registerIsProgramCounter(unsigned registerID) {
  return (programCounters.find(registerID) != programCounters.end());
}

int EDDisassembler::printInst(std::string& str,
                              MCInst& inst) {
  PrinterMutex.acquire();
  
  InstPrinter->printInst(&inst);
  InstStream->flush();
  str = *InstString;
  InstString->clear();
  
  PrinterMutex.release();
  
  return 0;
}

int EDDisassembler::parseInst(SmallVectorImpl<MCParsedAsmOperand*> &operands,
                              SmallVectorImpl<AsmToken> &tokens,
                              const std::string &str) {
  int ret = 0;
  
  const char *cStr = str.c_str();
  MemoryBuffer *buf = MemoryBuffer::getMemBuffer(cStr, cStr + strlen(cStr));
  
  StringRef instName;
  SMLoc instLoc;
  
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(buf, SMLoc()); // ownership of buf handed over
  MCContext context;
  OwningPtr<MCStreamer> streamer
    (createNullStreamer(context));
  AsmParser genericParser(sourceMgr, context, *streamer, *AsmInfo);
  OwningPtr<TargetAsmParser> specificParser
    (Tgt->createAsmParser(genericParser));
  
  AsmToken OpcodeToken = genericParser.Lex();
  
  if(OpcodeToken.is(AsmToken::Identifier)) {
    instName = OpcodeToken.getString();
    instLoc = OpcodeToken.getLoc();
    if (specificParser->ParseInstruction(instName, instLoc, operands))
      ret = -1;
  }
  else {
    ret = -1;
  }
  
  SmallVectorImpl<MCParsedAsmOperand*>::iterator oi;
  
  for(oi = operands.begin(); oi != operands.end(); ++oi) {
    printf("Operand start %p, end %p\n", 
           (*oi)->getStartLoc().getPointer(),
           (*oi)->getEndLoc().getPointer());
  }
  
  ParserMutex.acquire();
  
  if (!ret) {
    GenericAsmLexer->setBuffer(buf);
  
    while (SpecificAsmLexer->Lex(),
           SpecificAsmLexer->isNot(AsmToken::Eof) &&
           SpecificAsmLexer->isNot(AsmToken::EndOfStatement)) {
      if (SpecificAsmLexer->is(AsmToken::Error)) {
        ret = -1;
        break;
      }
      tokens.push_back(SpecificAsmLexer->getTok());
    }
  }

  ParserMutex.release();
  
  return ret;
}

int EDDisassembler::llvmSyntaxVariant() const {
  return LLVMSyntaxVariant;
}
