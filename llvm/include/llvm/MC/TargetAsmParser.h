//===-- llvm/Target/TargetAsmParser.h - Target Assembly Parser --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETPARSER_H
#define LLVM_TARGET_TARGETPARSER_H

#include "llvm/MC/MCParser/MCAsmParserExtension.h"

namespace llvm {
class MCStreamer;
class StringRef;
class SMLoc;
class AsmToken;
class MCParsedAsmOperand;
template <typename T> class SmallVectorImpl;

/// TargetAsmParser - Generic interface to target specific assembly parsers.
class TargetAsmParser : public MCAsmParserExtension {
  TargetAsmParser(const TargetAsmParser &);   // DO NOT IMPLEMENT
  void operator=(const TargetAsmParser &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  TargetAsmParser();
 
  /// AvailableFeatures - The current set of available features.
  unsigned AvailableFeatures;

public:
  virtual ~TargetAsmParser();

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
  
};

} // End llvm namespace

#endif
