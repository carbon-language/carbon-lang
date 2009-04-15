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
#include "llvm/Support/Streams.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/VectorExtras.h"
#include <set>
#include <map>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Warning Tables (.inc file) generation.
//===----------------------------------------------------------------------===//

void ClangDiagsDefsEmitter::run(std::ostream &OS) {
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
    OS << ", \"";
    std::string S = R.getValueAsString("Text");
    EscapeString(S);
    OS << S << "\")\n";
  }
}

//===----------------------------------------------------------------------===//
// Warning Group Tables generation
//===----------------------------------------------------------------------===//

void ClangDiagGroupsEmitter::run(std::ostream &OS) {
  // Invert the 1-[0/1] mapping of diags to group into a one to many mapping of
  // groups to diags in the group.
  std::map<std::string, std::vector<const Record*> > DiagsInGroup;
  
  const std::vector<Record*> &Diags =
    Records.getAllDerivedDefinitions("Diagnostic");
  
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    const Record *R = Diags[i];
    DefInit *DI = dynamic_cast<DefInit*>(R->getValueInit("Group"));
    if (DI == 0) continue;
    DiagsInGroup[DI->getDef()->getValueAsString("GroupName")].push_back(R);
  }
  
  // Walk through the groups emitting an array for each diagnostic of the diags
  // that are mapped to.
  OS << "\n#ifdef GET_DIAG_ARRAYS\n";
  unsigned IDNo = 0;
  unsigned MaxLen = 0;
  for (std::map<std::string, std::vector<const Record*> >::iterator
       I = DiagsInGroup.begin(), E = DiagsInGroup.end(); I != E; ++I) {
    MaxLen = std::max(MaxLen, (unsigned)I->first.size());
    
    OS << "static const short DiagArray" << IDNo++
       << "[] = { ";
    std::vector<const Record*> &V = I->second;
    for (unsigned i = 0, e = V.size(); i != e; ++i)
      OS << "diag::" << V[i]->getName() << ", ";
    OS << "-1 };\n";
  }
  OS << "#endif // GET_DIAG_ARRAYS\n\n";
  
  // Emit the table now.
  OS << "\n#ifdef GET_DIAG_TABLE\n";
  IDNo = 0;
  for (std::map<std::string, std::vector<const Record*> >::iterator
       I = DiagsInGroup.begin(), E = DiagsInGroup.end(); I != E; ++I) {
    std::string S = I->first;
    EscapeString(S);
    OS << "  { \"" << S << "\","
       << std::string(MaxLen-I->first.size()+1, ' ')
       << "DiagArray" << IDNo++ << " },\n";
  }
  OS << "#endif // GET_DIAG_TABLE\n\n";
}
