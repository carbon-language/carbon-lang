//===- RegisterInfoEmitter.cpp - Generate a Register File Desc. -*- C++ -*-===//
//
// This tablegen backend is responsible for emitting a description of a target
// register file for a code generator.  It uses instances of the Register,
// RegisterAliases, and RegisterClass classes to gather this information.
//
//===----------------------------------------------------------------------===//

#include "RegisterInfoEmitter.h"
#include "Record.h"
#include "Support/StringExtras.h"
#include <set>

static void EmitSourceHeader(const std::string &Desc, std::ostream &o) {
  o << "//===- TableGen'erated file -------------------------------------*-"
       " C++ -*-===//\n//\n// " << Desc << "\n//\n// Automatically generate"
       "d file, do not edit!\n//\n//===------------------------------------"
       "----------------------------------===//\n\n";
}

// runEnums - Print out enum values for all of the registers.
void RegisterInfoEmitter::runEnums(std::ostream &OS) {
  std::vector<Record*> Registers = Records.getAllDerivedDefinitions("Register");

  if (Registers.size() == 0)
    throw std::string("No 'Register' subclasses defined!");

  std::string Namespace = Registers[0]->getValueAsString("Namespace");

  EmitSourceHeader("Target Register Enum Values", OS);

  if (!Namespace.empty())
    OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n    NoRegister,\n";

  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    OS << "    " << Registers[i]->getName() << ",\n";
  
  OS << "  };\n";
  if (!Namespace.empty())
    OS << "}\n";
}

void RegisterInfoEmitter::runHeader(std::ostream &OS) {
  std::vector<Record*> RegisterInfos =
    Records.getAllDerivedDefinitions("RegisterInfo");

  if (RegisterInfos.size() != 1)
    throw std::string("ERROR: Multiple subclasses of RegisterInfo defined!");

  EmitSourceHeader("Register Information Header Fragment", OS);

  std::string ClassName = RegisterInfos[0]->getValueAsString("ClassName");

  OS << "#include \"llvm/Target/MRegisterInfo.h\"\n\n";

  OS << "struct " << ClassName << " : public MRegisterInfo {\n"
     << "  " << ClassName << "();\n"
     << "  const unsigned* getCalleeSaveRegs() const;\n"
     << "};\n\n";
}

static std::string getQualifiedRecordName(Record *R) {
  std::string Namespace = R->getValueAsString("Namespace");
  if (Namespace.empty()) return R->getName();
  return Namespace + "::" + R->getName();
}

// RegisterInfoEmitter::run - Main register file description emitter.
//
void RegisterInfoEmitter::run(std::ostream &OS) {
  EmitSourceHeader("Register Information Source Fragment", OS);

  // Start out by emitting each of the register classes... to do this, we build
  // a set of registers which belong to a register class, this is to ensure that
  // each register is only in a single register class.
  //
  std::vector<Record*> RegisterClasses =
    Records.getAllDerivedDefinitions("RegisterClass");

  std::vector<Record*> Registers = Records.getAllDerivedDefinitions("Register");
  Record *RegisterClass = Records.getClass("Register");

  std::set<Record*> RegistersFound;

  // Loop over all of the register classes... emitting each one.
  OS << "namespace {     // Register classes...\n";
  std::vector<std::string> RegisterClassNames;
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    Record *RC = RegisterClasses[rc];
    std::string Name = RC->getName();
    if (Name[9] == '.') {
      static unsigned AnonCounter = 0;
      Name = "AnonRegClass_"+utostr(AnonCounter++);
    }

    // Emit the register list now...
    OS << "  // " << Name << " Register Class...\n  const unsigned " << Name
       << "[] = {\n    ";
    ListInit *RegList = RC->getValueAsListInit("MemberList");
    for (unsigned i = 0, e = RegList->getSize(); i != e; ++i) {
      Record *Reg = RegList->getElement(i);
      if (!Reg->isSubClassOf(RegisterClass))
        throw "Register Class member '" + Reg->getName() +
              " does not derive from the Register class!";
      if (RegistersFound.count(Reg))
        throw "Register '" + Reg->getName() +
              "' included in multiple register classes!";
      OS << getQualifiedRecordName(Reg) << ", ";
    }
    OS << "\n  };\n\n";

    OS << "  struct " << Name << "Class : public TargetRegisterClass {\n"
       << "    " << Name << "Class() : TargetRegisterClass("
       << RC->getValueAsInt("Size")/8 << ", " << RC->getValueAsInt("Alignment")
       << ", " << Name << ", " << Name << " + " << RegList->getSize()
       << ") {}\n";
    
    if (CodeInit *CI = dynamic_cast<CodeInit*>(RC->getValueInit("Methods")))
      OS << CI->getValue();
    else
      throw "Expected 'code' fragment for 'Methods' value in register class '"+
            RC->getName() + "'!";

    OS << "  } " << Name << "Instance;\n\n";

  }

  OS << "}\n";         // End of anonymous namespace...
}
