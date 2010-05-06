//=== ClangASTNodesEmitter.cpp - Generate Clang AST node tables -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang AST node tables
//
//===----------------------------------------------------------------------===//

#include "ClangASTNodesEmitter.h"
#include "Record.h"
#include <map>
#include <cctype>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Statement Node Tables (.inc file) generation.
//===----------------------------------------------------------------------===//

// Create a macro-ized version of a name
static std::string macroName(std::string S) {
  for (unsigned i = 0; i < S.size(); ++i)
    S[i] = std::toupper(S[i]);

  return S;
}

// A map from a node to each of its derived nodes.
typedef std::multimap<Record*, Record*> ChildMap;
typedef ChildMap::const_iterator ChildIterator;

// Returns the first and last non-abstract subrecords
// Called recursively to ensure that nodes remain contiguous
static std::pair<Record *, Record *> EmitStmtNode(const ChildMap &Tree,
                                                  raw_ostream &OS,
                                                  Record *Base) {
  std::string BaseName = macroName(Base->getName());

  ChildIterator i = Tree.lower_bound(Base), e = Tree.upper_bound(Base);

  Record *First = 0, *Last = 0;
  // This might be the pseudo-node for Stmt; don't assume it has an Abstract
  // bit
  if (Base->getValue("Abstract") && !Base->getValueAsBit("Abstract"))
    First = Last = Base;

  for (; i != e; ++i) {
    Record *R = i->second;
    bool Abstract = R->getValueAsBit("Abstract");
    std::string NodeName = macroName(R->getName());

    OS << "#ifndef " << NodeName << "\n";
    OS << "#  define " << NodeName << "(Type, Base) "
        << BaseName << "(Type, Base)\n";
    OS << "#endif\n";

    if (Abstract)
      OS << "ABSTRACT(" << NodeName << "(" << R->getName() << ", "
          << Base->getName() << "))\n";
    else
      OS << NodeName << "(" << R->getName() << ", "
          << Base->getName() << ")\n";

    if (Tree.find(R) != Tree.end()) {
      const std::pair<Record *, Record *> &Result = EmitStmtNode(Tree, OS, R);
      if (!First && Result.first)
        First = Result.first;
      if (Result.second)
        Last = Result.second;
    } else {
      if (!Abstract) {
        Last = R;

        if (!First)
          First = R;
      }
    }

    OS << "#undef " << NodeName << "\n\n";
  }

  assert(!First == !Last && "Got a first or last node, but not the other");

  if (First) {
    OS << "#ifndef FIRST_" << BaseName << "\n";
    OS << "#  define FIRST_" << BaseName << "(CLASS)\n";
    OS << "#endif\n";
    OS << "#ifndef LAST_" << BaseName << "\n";
    OS << "#  define LAST_" << BaseName << "(CLASS)\n";
    OS << "#endif\n\n";

    OS << "FIRST_" << BaseName << "(" << First->getName() << ")\n";
    OS << "LAST_" << BaseName << "(" << Last->getName() << ")\n\n";
  }

  OS << "#undef FIRST_" << BaseName << "\n";
  OS << "#undef LAST_" << BaseName << "\n\n";

  return std::make_pair(First, Last);
}

void ClangStmtNodesEmitter::run(raw_ostream &OS) {
  // Write the preamble
  OS << "#ifndef ABSTRACT\n";
  OS << "#  define ABSTRACT(Stmt) Stmt\n";
  OS << "#endif\n\n";

  // Emit statements
  const std::vector<Record*> Stmts = Records.getAllDerivedDefinitions("Stmt");

  ChildMap Tree;

  // Create a pseudo-record to serve as the Stmt node, which isn't actually
  // output.
  Record Stmt ("Stmt", SMLoc());

  for (unsigned i = 0, e = Stmts.size(); i != e; ++i) {
    Record *R = Stmts[i];

    if (R->getValue("Base"))
      Tree.insert(std::make_pair(R->getValueAsDef("Base"), R));
    else
      Tree.insert(std::make_pair(&Stmt, R));
  }

  EmitStmtNode(Tree, OS, &Stmt);

  OS << "#undef STMT\n";
  OS << "#undef ABSTRACT\n";
}
