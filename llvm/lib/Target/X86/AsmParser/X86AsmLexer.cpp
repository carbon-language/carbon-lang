//===-- X86AsmLexer.cpp - Tokenize X86 assembly to AsmTokens --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Target/TargetAsmLexer.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "X86.h"

using namespace llvm;

namespace {
  
class X86AsmLexer : public TargetAsmLexer {
  const MCAsmInfo &AsmInfo;
  
  bool tentativeIsValid;
  AsmToken tentativeToken;
  
  const AsmToken &lexTentative() {
    tentativeToken = getLexer()->Lex();
    tentativeIsValid = true;
    return tentativeToken;
  }
  
  const AsmToken &lexDefinite() {
    if(tentativeIsValid) {
      tentativeIsValid = false;
      return tentativeToken;
    }
    else {
      return getLexer()->Lex();
    }
  }
  
  AsmToken LexTokenATT();
  AsmToken LexTokenIntel();
protected:
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
      return LexTokenATT();
    case 1:
      return LexTokenIntel();
    }
  }
public:
  X86AsmLexer(const Target &T, const MCAsmInfo &MAI)
    : TargetAsmLexer(T), AsmInfo(MAI), tentativeIsValid(false) {
  }
};

}

static unsigned MatchRegisterName(StringRef Name);

AsmToken X86AsmLexer::LexTokenATT() {
  const AsmToken lexedToken = lexDefinite();
  
  switch (lexedToken.getKind()) {
  default:
    return AsmToken(lexedToken);
  case AsmToken::Error:
    SetError(Lexer->getErrLoc(), Lexer->getErr());
    return AsmToken(lexedToken);
  case AsmToken::Percent:
  {
    const AsmToken &nextToken = lexTentative();
    if (nextToken.getKind() == AsmToken::Identifier) {
      unsigned regID = MatchRegisterName(nextToken.getString());
      
      if (regID) {
        lexDefinite();
        
        StringRef regStr(lexedToken.getString().data(),
                         lexedToken.getString().size() + 
                         nextToken.getString().size());
        
        return AsmToken(AsmToken::Register, 
                        regStr, 
                        static_cast<int64_t>(regID));
      }
      else {
        return AsmToken(lexedToken);
      }
    }
    else {
      return AsmToken(lexedToken);
    }
  }    
  }
}

AsmToken X86AsmLexer::LexTokenIntel() {
  const AsmToken &lexedToken = lexDefinite();
  
  switch(lexedToken.getKind()) {
  default:
    return AsmToken(lexedToken);
  case AsmToken::Error:
    SetError(Lexer->getErrLoc(), Lexer->getErr());
    return AsmToken(lexedToken);
  case AsmToken::Identifier:
  {
    std::string upperCase = lexedToken.getString().str();
    std::string lowerCase = LowercaseString(upperCase);
    StringRef lowerRef(lowerCase);
    
    unsigned regID = MatchRegisterName(lowerRef);
    
    if (regID) {
      return AsmToken(AsmToken::Register,
                      lexedToken.getString(),
                      static_cast<int64_t>(regID));
    }
    else {
      return AsmToken(lexedToken);
    }
  }
  }
}

extern "C" void LLVMInitializeX86AsmLexer() {
  RegisterAsmLexer<X86AsmLexer> X(TheX86_32Target);
  RegisterAsmLexer<X86AsmLexer> Y(TheX86_64Target);
}

#define REGISTERS_ONLY
#include "X86GenAsmMatcher.inc"
#undef REGISTERS_ONLY
