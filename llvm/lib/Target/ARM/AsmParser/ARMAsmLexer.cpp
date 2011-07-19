//===-- ARMAsmLexer.cpp - Tokenize ARM assembly to AsmTokens --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMTargetMachine.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"

#include "llvm/Target/TargetAsmLexer.h"
#include "llvm/Target/TargetMachine.h"  // FIXME
#include "llvm/Target/TargetRegistry.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <string>
#include <map>

using namespace llvm;

namespace {

class ARMBaseAsmLexer : public TargetAsmLexer {
  const MCAsmInfo &AsmInfo;

  const AsmToken &lexDefinite() {
    return getLexer()->Lex();
  }

  AsmToken LexTokenUAL();
protected:
  typedef std::map <std::string, unsigned> rmap_ty;

  rmap_ty RegisterMap;

  void InitRegisterMap(const TargetRegisterInfo *info) {
    unsigned numRegs = info->getNumRegs();

    for (unsigned i = 0; i < numRegs; ++i) {
      const char *regName = info->getName(i);
      if (regName)
        RegisterMap[regName] = i;
    }
  }

  unsigned MatchRegisterName(StringRef Name) {
    rmap_ty::iterator iter = RegisterMap.find(Name.str());
    if (iter != RegisterMap.end())
      return iter->second;
    else
      return 0;
  }

  AsmToken LexToken() {
    if (!Lexer) {
      SetError(SMLoc(), "No MCAsmLexer installed");
      return AsmToken(AsmToken::Error, "", 0);
    }

    switch (AsmInfo.getAssemblerDialect()) {
    default:
      SetError(SMLoc(), "Unhandled dialect");
      return AsmToken(AsmToken::Error, "", 0);
    case 0:
      return LexTokenUAL();
    }
  }
public:
  ARMBaseAsmLexer(const Target &T, const MCAsmInfo &MAI)
    : TargetAsmLexer(T), AsmInfo(MAI) {
  }
};

class ARMAsmLexer : public ARMBaseAsmLexer {
public:
  ARMAsmLexer(const Target &T, const MCAsmInfo &MAI)
    : ARMBaseAsmLexer(T, MAI) {
    std::string tripleString("arm-unknown-unknown");
    std::string featureString;
    std::string CPU;
    OwningPtr<const TargetMachine>
      targetMachine(T.createTargetMachine(tripleString, CPU, featureString));
    InitRegisterMap(targetMachine->getRegisterInfo());
  }
};

class ThumbAsmLexer : public ARMBaseAsmLexer {
public:
  ThumbAsmLexer(const Target &T, const MCAsmInfo &MAI)
    : ARMBaseAsmLexer(T, MAI) {
    std::string tripleString("thumb-unknown-unknown");
    std::string featureString;
    std::string CPU;
    OwningPtr<const TargetMachine>
      targetMachine(T.createTargetMachine(tripleString, CPU, featureString));
    InitRegisterMap(targetMachine->getRegisterInfo());
  }
};

} // end anonymous namespace

AsmToken ARMBaseAsmLexer::LexTokenUAL() {
  const AsmToken &lexedToken = lexDefinite();

  switch (lexedToken.getKind()) {
  default: break;
  case AsmToken::Error:
    SetError(Lexer->getErrLoc(), Lexer->getErr());
    break;
  case AsmToken::Identifier: {
    std::string upperCase = lexedToken.getString().str();
    std::string lowerCase = LowercaseString(upperCase);
    StringRef lowerRef(lowerCase);

    unsigned regID = MatchRegisterName(lowerRef);
    // Check for register aliases.
    //   r13 -> sp
    //   r14 -> lr
    //   r15 -> pc
    //   ip  -> r12
    //   FIXME: Some assemblers support lots of others. Do we want them all?
    if (!regID) {
      regID = StringSwitch<unsigned>(lowerCase)
        .Case("r13", ARM::SP)
        .Case("r14", ARM::LR)
        .Case("r15", ARM::PC)
        .Case("ip", ARM::R12)
        .Default(0);
    }

    if (regID)
      return AsmToken(AsmToken::Register,
                      lexedToken.getString(),
                      static_cast<int64_t>(regID));
  }
  }

  return AsmToken(lexedToken);
}

extern "C" void LLVMInitializeARMAsmLexer() {
  RegisterAsmLexer<ARMAsmLexer> X(TheARMTarget);
  RegisterAsmLexer<ThumbAsmLexer> Y(TheThumbTarget);
}
