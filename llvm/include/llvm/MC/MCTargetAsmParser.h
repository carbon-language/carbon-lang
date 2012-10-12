//===-- llvm/MC/MCTargetAsmParser.h - Target Assembly Parser ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_TARGETPARSER_H
#define LLVM_MC_TARGETPARSER_H

#include "llvm/MC/MCParser/MCAsmParserExtension.h"

namespace llvm {
class MCStreamer;
class StringRef;
class SMLoc;
class AsmToken;
class MCParsedAsmOperand;
class MCInst;
template <typename T> class SmallVectorImpl;

/// MCTargetAsmParser - Generic interface to target specific assembly parsers.
class MCTargetAsmParser : public MCAsmParserExtension {
public:
  enum MatchResultTy {
    Match_InvalidOperand,
    Match_MissingFeature,
    Match_MnemonicFail,
    Match_Success,
    FIRST_TARGET_MATCH_RESULT_TY
  };

private:
  MCTargetAsmParser(const MCTargetAsmParser &) LLVM_DELETED_FUNCTION;
  void operator=(const MCTargetAsmParser &) LLVM_DELETED_FUNCTION;
protected: // Can only create subclasses.
  MCTargetAsmParser();

  /// AvailableFeatures - The current set of available features.
  unsigned AvailableFeatures;

public:
  virtual ~MCTargetAsmParser();

  unsigned getAvailableFeatures() const { return AvailableFeatures; }
  void setAvailableFeatures(unsigned Value) { AvailableFeatures = Value; }

  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                             SMLoc &EndLoc) = 0;

  /// ParseInstruction - Parse one assembly instruction.
  ///
  /// The parser is positioned following the instruction name. The target
  /// specific instruction parser should parse the entire instruction and
  /// construct the appropriate MCInst, or emit an error. On success, the entire
  /// line should be parsed up to and including the end-of-statement token. On
  /// failure, the parser is not required to read to the end of the line.
  //
  /// \param Name - The instruction name.
  /// \param NameLoc - The source location of the name.
  /// \param Operands [out] - The list of parsed operands, this returns
  ///        ownership of them to the caller.
  /// \return True on failure.
  virtual bool ParseInstruction(StringRef Name, SMLoc NameLoc,
                            SmallVectorImpl<MCParsedAsmOperand*> &Operands) = 0;

  /// ParseDirective - Parse a target specific assembler directive
  ///
  /// The parser is positioned following the directive name.  The target
  /// specific directive parser should parse the entire directive doing or
  /// recording any target specific work, or return true and do nothing if the
  /// directive is not target specific. If the directive is specific for
  /// the target, the entire line is parsed up to and including the
  /// end-of-statement token and false is returned.
  ///
  /// \param DirectiveID - the identifier token of the directive.
  virtual bool ParseDirective(AsmToken DirectiveID) = 0;

  /// mnemonicIsValid - This returns true if this is a valid mnemonic and false
  /// otherwise.
  virtual bool mnemonicIsValid(StringRef Mnemonic) = 0;

  /// MatchInstruction - Recognize a series of operands of a parsed instruction
  /// as an actual MCInst.  This returns false on success and returns true on
  /// failure to match.
  ///
  /// On failure, the target parser is responsible for emitting a diagnostic
  /// explaining the match failure.
  virtual bool
  MatchInstruction(SMLoc IDLoc, 
                   SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                   MCStreamer &Out, unsigned &Opcode, unsigned &OrigErrorInfo,
                   bool MatchingInlineAsm = false) {
    OrigErrorInfo = ~0x0;
    return true;
  }

  /// MatchAndEmitInstruction - Recognize a series of operands of a parsed
  /// instruction as an actual MCInst and emit it to the specified MCStreamer.
  /// This returns false on success and returns true on failure to match.
  ///
  /// On failure, the target parser is responsible for emitting a diagnostic
  /// explaining the match failure.
  virtual bool
  MatchAndEmitInstruction(SMLoc IDLoc,
                          SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                          MCStreamer &Out) = 0;

  /// checkTargetMatchPredicate - Validate the instruction match against
  /// any complex target predicates not expressible via match classes.
  virtual unsigned checkTargetMatchPredicate(MCInst &Inst) {
    return Match_Success;
  }

  virtual void convertToMapAndConstraints(unsigned Kind,
                       const SmallVectorImpl<MCParsedAsmOperand*> &Operands) = 0;
};

} // End llvm namespace

#endif
