//=- ClangDiagnosticsEmitter.cpp - Generate Clang diagnostics tables -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang diagnostics tables.
//
//===----------------------------------------------------------------------===//

#include "ClangDiagnosticsEmitter.h"
#include "Record.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/VectorExtras.h"
#include <set>
#include <map>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Warning Tables (.inc file) generation.
//===----------------------------------------------------------------------===//

void ClangDiagsDefsEmitter::run(raw_ostream &OS) {
  // Write the #if guard
  if (!Component.empty()) {
    std::string ComponentName = UppercaseString(Component);
    OS << "#ifdef " << ComponentName << "START\n";
    OS << "__" << ComponentName << "START = DIAG_START_" << ComponentName
       << ",\n";
    OS << "#undef " << ComponentName << "START\n";
    OS << "#endif\n";
  }

  const std::vector<Record*> &Diags =
    Records.getAllDerivedDefinitions("Diagnostic");
  
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    const Record &R = *Diags[i];
    // Filter by component.
    if (!Component.empty() && Component != R.getValueAsString("Component"))
      continue;
    
    OS << "DIAG(" << R.getName() << ", ";
    OS << R.getValueAsDef("Class")->getName();
    OS << ", diag::" << R.getValueAsDef("DefaultMapping")->getName();
    
    // Description string.
    OS << ", \"";
    OS.write_escaped(R.getValueAsString("Text")) << '"';
    
    // Warning associated with the diagnostic.
    if (DefInit *DI = dynamic_cast<DefInit*>(R.getValueInit("Group"))) {
      OS << ", \"";
      OS.write_escaped(DI->getDef()->getValueAsString("GroupName")) << '"';
    } else {
      OS << ", 0";
    }

    // SFINAE bit
    if (R.getValueAsBit("SFINAE"))
      OS << ", true";
    else
      OS << ", false";
    OS << ")\n";
  }
}

//===----------------------------------------------------------------------===//
// Warning Group Tables generation
//===----------------------------------------------------------------------===//

struct GroupInfo {
  std::vector<const Record*> DiagsInGroup;
  std::vector<std::string> SubGroups;
  unsigned IDNo;
};

void ClangDiagGroupsEmitter::run(raw_ostream &OS) {
  // Invert the 1-[0/1] mapping of diags to group into a one to many mapping of
  // groups to diags in the group.
  std::map<std::string, GroupInfo> DiagsInGroup;
  
  std::vector<Record*> Diags =
    Records.getAllDerivedDefinitions("Diagnostic");
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    const Record *R = Diags[i];
    DefInit *DI = dynamic_cast<DefInit*>(R->getValueInit("Group"));
    if (DI == 0) continue;
    std::string GroupName = DI->getDef()->getValueAsString("GroupName");
    DiagsInGroup[GroupName].DiagsInGroup.push_back(R);
  }
  
  // Add all DiagGroup's to the DiagsInGroup list to make sure we pick up empty
  // groups (these are warnings that GCC supports that clang never produces).
  Diags = Records.getAllDerivedDefinitions("DiagGroup");
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    Record *Group = Diags[i];
    GroupInfo &GI = DiagsInGroup[Group->getValueAsString("GroupName")];
    
    std::vector<Record*> SubGroups = Group->getValueAsListOfDefs("SubGroups");
    for (unsigned j = 0, e = SubGroups.size(); j != e; ++j)
      GI.SubGroups.push_back(SubGroups[j]->getValueAsString("GroupName"));
  }
  
  // Assign unique ID numbers to the groups.
  unsigned IDNo = 0;
  for (std::map<std::string, GroupInfo>::iterator
       I = DiagsInGroup.begin(), E = DiagsInGroup.end(); I != E; ++I, ++IDNo)
    I->second.IDNo = IDNo;
  
  // Walk through the groups emitting an array for each diagnostic of the diags
  // that are mapped to.
  OS << "\n#ifdef GET_DIAG_ARRAYS\n";
  unsigned MaxLen = 0;
  for (std::map<std::string, GroupInfo>::iterator
       I = DiagsInGroup.begin(), E = DiagsInGroup.end(); I != E; ++I) {
    MaxLen = std::max(MaxLen, (unsigned)I->first.size());
    
    std::vector<const Record*> &V = I->second.DiagsInGroup;
    if (!V.empty()) {
      OS << "static const short DiagArray" << I->second.IDNo << "[] = { ";
      for (unsigned i = 0, e = V.size(); i != e; ++i)
        OS << "diag::" << V[i]->getName() << ", ";
      OS << "-1 };\n";
    }
    
    const std::vector<std::string> &SubGroups = I->second.SubGroups;
    if (!SubGroups.empty()) {
      OS << "static const char DiagSubGroup" << I->second.IDNo << "[] = { ";
      for (unsigned i = 0, e = SubGroups.size(); i != e; ++i) {
        std::map<std::string, GroupInfo>::iterator RI =
          DiagsInGroup.find(SubGroups[i]);
        assert(RI != DiagsInGroup.end() && "Referenced without existing?");
        OS << RI->second.IDNo << ", ";
      }
      OS << "-1 };\n";
    }
  }
  OS << "#endif // GET_DIAG_ARRAYS\n\n";
  
  // Emit the table now.
  OS << "\n#ifdef GET_DIAG_TABLE\n";
  for (std::map<std::string, GroupInfo>::iterator
       I = DiagsInGroup.begin(), E = DiagsInGroup.end(); I != E; ++I) {
    // Group option string.
    OS << "  { \"";
    OS.write_escaped(I->first) << "\","
                               << std::string(MaxLen-I->first.size()+1, ' ');
    
    // Diagnostics in the group.
    if (I->second.DiagsInGroup.empty())
      OS << "0, ";
    else
      OS << "DiagArray" << I->second.IDNo << ", ";
    
    // Subgroups.
    if (I->second.SubGroups.empty())
      OS << 0;
    else
      OS << "DiagSubGroup" << I->second.IDNo;
    OS << " },\n";
  }
  OS << "#endif // GET_DIAG_TABLE\n\n";
}
