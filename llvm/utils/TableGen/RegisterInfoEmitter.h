//===- RegisterInfoEmitter.h - Generate a Register File Desc. ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of a target
// register file for a code generator.  It uses instances of the Register,
// RegisterAliases, and RegisterClass classes to gather this information.
//
//===----------------------------------------------------------------------===//

#ifndef REGISTER_INFO_EMITTER_H
#define REGISTER_INFO_EMITTER_H

#include "llvm/TableGen/TableGenBackend.h"
#include <vector>

namespace llvm {

class CodeGenRegBank;
struct CodeGenRegister;
class CodeGenTarget;

class RegisterInfoEmitter : public TableGenBackend {
  RecordKeeper &Records;
public:
  RegisterInfoEmitter(RecordKeeper &R) : Records(R) {}

  // runEnums - Print out enum values for all of the registers.
  void runEnums(raw_ostream &o, CodeGenTarget &Target, CodeGenRegBank &Bank);

  // runMCDesc - Print out MC register descriptions.
  void runMCDesc(raw_ostream &o, CodeGenTarget &Target, CodeGenRegBank &Bank);

  // runTargetHeader - Emit a header fragment for the register info emitter.
  void runTargetHeader(raw_ostream &o, CodeGenTarget &Target,
                       CodeGenRegBank &Bank);

  // runTargetDesc - Output the target register and register file descriptions.
  void runTargetDesc(raw_ostream &o, CodeGenTarget &Target,
                     CodeGenRegBank &Bank);

  // run - Output the register file description.
  void run(raw_ostream &o);

private:
  void EmitRegMapping(raw_ostream &o,
                      const std::vector<CodeGenRegister*> &Regs, bool isCtor);
  void EmitRegClasses(raw_ostream &OS, CodeGenTarget &Target);
};

} // End llvm namespace

#endif
