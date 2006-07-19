//===- RegisterInfoEmitter.cpp - Generate a Register File Desc. -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of a target
// register file for a code generator.  It uses instances of the Register,
// RegisterAliases, and RegisterClass classes to gather this information.
//
//===----------------------------------------------------------------------===//

#include "RegisterInfoEmitter.h"
#include "CodeGenTarget.h"
#include "CodeGenRegisters.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
using namespace llvm;

// runEnums - Print out enum values for all of the registers.
void RegisterInfoEmitter::runEnums(std::ostream &OS) {
  CodeGenTarget Target;
  const std::vector<CodeGenRegister> &Registers = Target.getRegisters();

  std::string Namespace = Registers[0].TheDef->getValueAsString("Namespace");

  EmitSourceFileHeader("Target Register Enum Values", OS);
  OS << "namespace llvm {\n\n";

  if (!Namespace.empty())
    OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n    NoRegister,\n";

  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    OS << "    " << Registers[i].getName() << (i != (e-1) ? ", \t// " : "  \t// ") << i+1 << "\n";

  OS << "  };\n";
  if (!Namespace.empty())
    OS << "}\n";
  OS << "} // End llvm namespace \n";
}

void RegisterInfoEmitter::runHeader(std::ostream &OS) {
  EmitSourceFileHeader("Register Information Header Fragment", OS);
  CodeGenTarget Target;
  const std::string &TargetName = Target.getName();
  std::string ClassName = TargetName + "GenRegisterInfo";

  OS << "#include \"llvm/Target/MRegisterInfo.h\"\n";
  OS << "#include <string>\n\n";

  OS << "namespace llvm {\n\n";

  OS << "struct " << ClassName << " : public MRegisterInfo {\n"
     << "  " << ClassName
     << "(int CallFrameSetupOpcode = -1, int CallFrameDestroyOpcode = -1);\n"
     << "  int getDwarfRegNum(unsigned RegNum) const;\n"
     << "};\n\n";

  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();

  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0].Namespace
       << " { // Register classes\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const std::string &Name = RegisterClasses[i].getName();

      // Output the register class definition.
      OS << "  struct " << Name << "Class : public TargetRegisterClass {\n"
         << "    " << Name << "Class();\n"
         << RegisterClasses[i].MethodProtos << "  };\n";

      // Output the extern for the instance.
      OS << "  extern " << Name << "Class\t" << Name << "RegClass;\n";
      // Output the extern for the pointer to the instance (should remove).
      OS << "  static TargetRegisterClass * const "<< Name <<"RegisterClass = &"
         << Name << "RegClass;\n";
    }
    OS << "} // end of namespace " << TargetName << "\n\n";
  }
  OS << "} // End llvm namespace \n";
}

bool isSubRegisterClass(const CodeGenRegisterClass &RC,
                        std::set<Record*> &RegSet) {
  for (unsigned i = 0, e = RC.Elements.size(); i != e; ++i) {
    Record *Reg = RC.Elements[i];
    if (!RegSet.count(Reg))
      return false;
  }
  return true;
}

// RegisterInfoEmitter::run - Main register file description emitter.
//
void RegisterInfoEmitter::run(std::ostream &OS) {
  CodeGenTarget Target;
  EmitSourceFileHeader("Register Information Source Fragment", OS);

  OS << "namespace llvm {\n\n";

  // Start out by emitting each of the register classes... to do this, we build
  // a set of registers which belong to a register class, this is to ensure that
  // each register is only in a single register class.
  //
  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();

  // Loop over all of the register classes... emitting each one.
  OS << "namespace {     // Register classes...\n";

  // RegClassesBelongedTo - Keep track of which register classes each reg
  // belongs to.
  std::multimap<Record*, const CodeGenRegisterClass*> RegClassesBelongedTo;

  // Emit the register enum value arrays for each RegisterClass
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RegisterClasses[rc];

    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.TheDef->getName();
  
    // Emit the register list now.
    OS << "  // " << Name << " Register Class...\n"
       << "  static const unsigned " << Name
       << "[] = {\n    ";
    for (unsigned i = 0, e = RC.Elements.size(); i != e; ++i) {
      Record *Reg = RC.Elements[i];
      OS << getQualifiedName(Reg) << ", ";

      // Keep track of which regclasses this register is in.
      RegClassesBelongedTo.insert(std::make_pair(Reg, &RC));
    }
    OS << "\n  };\n\n";
  }

  // Emit the ValueType arrays for each RegisterClass
  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RegisterClasses[rc];
    
    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.TheDef->getName() + "VTs";
    
    // Emit the register list now.
    OS << "  // " << Name 
       << " Register Class Value Types...\n"
       << "  static const MVT::ValueType " << Name
       << "[] = {\n    ";
    for (unsigned i = 0, e = RC.VTs.size(); i != e; ++i)
      OS << RC.VTs[i] << ", ";
    OS << "MVT::Other\n  };\n\n";
  }
  OS << "}  // end anonymous namespace\n\n";
  
  // Now that all of the structs have been emitted, emit the instances.
  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0].Namespace
       << " {   // Register class instances\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i)
      OS << "  " << RegisterClasses[i].getName()  << "Class\t"
         << RegisterClasses[i].getName() << "RegClass;\n";

    std::map<unsigned, std::set<unsigned> > SuperClassMap;
    OS << "\n";
    // Emit the sub-classes array for each RegisterClass
    for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
      const CodeGenRegisterClass &RC = RegisterClasses[rc];

      // Give the register class a legal C name if it's anonymous.
      std::string Name = RC.TheDef->getName();

      std::set<Record*> RegSet;
      for (unsigned i = 0, e = RC.Elements.size(); i != e; ++i) {
        Record *Reg = RC.Elements[i];
        RegSet.insert(Reg);
      }

      OS << "  // " << Name 
         << " Register Class sub-classes...\n"
         << "  static const TargetRegisterClass* const "
         << Name << "Subclasses [] = {\n    ";

      bool Empty = true;
      for (unsigned rc2 = 0, e2 = RegisterClasses.size(); rc2 != e2; ++rc2) {
        const CodeGenRegisterClass &RC2 = RegisterClasses[rc2];
        if (rc == rc2 || RC2.Elements.size() > RC.Elements.size() ||
            RC.SpillSize != RC2.SpillSize || !isSubRegisterClass(RC2, RegSet))
          continue;
      
        if (!Empty) OS << ", ";
        OS << "&" << getQualifiedName(RC2.TheDef) << "RegClass";
        Empty = false;

        std::map<unsigned, std::set<unsigned> >::iterator SCMI =
          SuperClassMap.find(rc2);
        if (SCMI == SuperClassMap.end()) {
          SuperClassMap.insert(std::make_pair(rc2, std::set<unsigned>()));
          SCMI = SuperClassMap.find(rc2);
        }
        SCMI->second.insert(rc);
      }

      OS << (!Empty ? ", " : "") << "NULL";
      OS << "\n  };\n\n";
    }

    for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
      const CodeGenRegisterClass &RC = RegisterClasses[rc];

      // Give the register class a legal C name if it's anonymous.
      std::string Name = RC.TheDef->getName();

      OS << "  // " << Name 
         << " Register Class super-classes...\n"
         << "  static const TargetRegisterClass* const "
         << Name << "Superclasses [] = {\n    ";

      bool Empty = true;
      std::map<unsigned, std::set<unsigned> >::iterator I =
        SuperClassMap.find(rc);
      if (I != SuperClassMap.end()) {
        for (std::set<unsigned>::iterator II = I->second.begin(),
               EE = I->second.end(); II != EE; ++II) {
          const CodeGenRegisterClass &RC2 = RegisterClasses[*II];
          if (!Empty) OS << ", ";
          OS << "&" << getQualifiedName(RC2.TheDef) << "RegClass";
          Empty = false;        
        }
      }

      OS << (!Empty ? ", " : "") << "NULL";
      OS << "\n  };\n\n";
    }


    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const CodeGenRegisterClass &RC = RegisterClasses[i];
      OS << RC.MethodBodies << "\n";
      OS << RC.getName() << "Class::" << RC.getName() 
         << "Class()  : TargetRegisterClass("
         << RC.getName() + "VTs" << ", "
         << RC.getName() + "Subclasses" << ", "
         << RC.getName() + "Superclasses" << ", "
         << RC.SpillSize/8 << ", "
         << RC.SpillAlignment/8 << ", " << RC.getName() << ", "
         << RC.getName() << " + " << RC.Elements.size() << ") {}\n";
    }
  
    OS << "}\n";
  }

  OS << "\nnamespace {\n";
  OS << "  const TargetRegisterClass* const RegisterClasses[] = {\n";
  for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i)
    OS << "    &" << getQualifiedName(RegisterClasses[i].TheDef)
       << "RegClass,\n";
  OS << "  };\n";

  // Emit register class aliases...
  std::map<Record*, std::set<Record*> > RegisterAliases;
  const std::vector<CodeGenRegister> &Regs = Target.getRegisters();

  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record *R = Regs[i].TheDef;
    std::vector<Record*> LI = Regs[i].TheDef->getValueAsListOfDefs("Aliases");
    // Add information that R aliases all of the elements in the list... and
    // that everything in the list aliases R.
    for (unsigned j = 0, e = LI.size(); j != e; ++j) {
      Record *Reg = LI[j];
      if (RegisterAliases[R].count(Reg))
        std::cerr << "Warning: register alias between " << getQualifiedName(R)
                  << " and " << getQualifiedName(Reg)
                  << " specified multiple times!\n";
      RegisterAliases[R].insert(Reg);

      if (RegisterAliases[Reg].count(R))
        std::cerr << "Warning: register alias between " << getQualifiedName(R)
                  << " and " << getQualifiedName(Reg)
                  << " specified multiple times!\n";
      RegisterAliases[Reg].insert(R);
    }
  }

  if (!RegisterAliases.empty())
    OS << "\n\n  // Register Alias Sets...\n";

  // Emit the empty alias list
  OS << "  const unsigned Empty_AliasSet[] = { 0 };\n";
  // Loop over all of the registers which have aliases, emitting the alias list
  // to memory.
  for (std::map<Record*, std::set<Record*> >::iterator
         I = RegisterAliases.begin(), E = RegisterAliases.end(); I != E; ++I) {
    OS << "  const unsigned " << I->first->getName() << "_AliasSet[] = { ";
    for (std::set<Record*>::iterator ASI = I->second.begin(),
           E = I->second.end(); ASI != E; ++ASI)
      OS << getQualifiedName(*ASI) << ", ";
    OS << "0 };\n";
  }

  OS<<"\n  const TargetRegisterDesc RegisterDescriptors[] = { // Descriptors\n";
  OS << "    { \"NOREG\",\t0 },\n";


  // Now that register alias sets have been emitted, emit the register
  // descriptors now.
  const std::vector<CodeGenRegister> &Registers = Target.getRegisters();
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    const CodeGenRegister &Reg = Registers[i];
    OS << "    { \"";
    if (!Reg.TheDef->getValueAsString("Name").empty())
      OS << Reg.TheDef->getValueAsString("Name");
    else
      OS << Reg.getName();
    OS << "\",\t";
    if (RegisterAliases.count(Reg.TheDef))
      OS << Reg.getName() << "_AliasSet },\n";
    else
      OS << "Empty_AliasSet },\n";
  }
  OS << "  };\n";      // End of register descriptors...
  OS << "}\n\n";       // End of anonymous namespace...

  std::string ClassName = Target.getName() + "GenRegisterInfo";

  // Emit the constructor of the class...
  OS << ClassName << "::" << ClassName
     << "(int CallFrameSetupOpcode, int CallFrameDestroyOpcode)\n"
     << "  : MRegisterInfo(RegisterDescriptors, " << Registers.size()+1
     << ", RegisterClasses, RegisterClasses+" << RegisterClasses.size() <<",\n "
     << "                 CallFrameSetupOpcode, CallFrameDestroyOpcode) {}\n\n";

  // Emit information about the dwarf register numbers.
  OS << "int " << ClassName << "::getDwarfRegNum(unsigned RegNum) const {\n";
  OS << "  static const int DwarfRegNums[] = { -1, // NoRegister";
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    if (!(i % 16)) OS << "\n    ";
    const CodeGenRegister &Reg = Registers[i];
    int DwarfRegNum = Reg.TheDef->getValueAsInt("DwarfNumber");
    OS << DwarfRegNum;
    if ((i + 1) != e)  OS << ", ";
  }
  OS << "\n  };\n";
  OS << "  assert(RegNum < (sizeof(DwarfRegNums)/sizeof(int)) &&\n";
  OS << "         \"RegNum exceeds number of registers\");\n";
  OS << "  return DwarfRegNums[RegNum];\n";
  OS << "}\n\n";

  OS << "} // End llvm namespace \n";
}
