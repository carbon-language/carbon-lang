//===- RegisterInfoEmitter.cpp - Generate a Register File Desc. -*- C++ -*-===//
//
// This tablegen backend is responsible for emitting a description of a target
// register file for a code generator.  It uses instances of the Register,
// RegisterAliases, and RegisterClass classes to gather this information.
//
//===----------------------------------------------------------------------===//

#include "RegisterInfoEmitter.h"
#include "Record.h"

static void EmitSourceHeader(const std::string &Desc, std::ostream &o) {
  o << "//===- TableGen'erated file -------------------------------------*-"
       " C++ -*-===//\n//\n// " << Desc << "\n//\n// Automatically generate"
       "d file, do not edit!\n//\n//===------------------------------------"
       "----------------------------------===//\n\n";
}

void RegisterInfoEmitter::runHeader(std::ostream &OS) {
  std::vector<Record*> RegisterInfos =
    Records.getAllDerivedDefinitions("RegisterInfo");

  if (RegisterInfos.size() != 1)
    throw std::string("ERROR: Multiple subclasses of RegisterInfo defined!");

  EmitSourceHeader("Register Information Header Fragment", OS);

  std::string ClassName = RegisterInfos[0]->getValueAsString("ClassName");

  OS << "#include \"llvm/CodeGen/MRegisterInfo.h\"\n\n";

  OS << "struct " << ClassName << ": public MRegisterInfo {\n"
     << "  " << ClassName << "();\n"
     << "  const unsigned* getCalleeSaveRegs() const;\n"
     << "};\n\n";
}

void RegisterInfoEmitter::run(std::ostream &o) {

}
