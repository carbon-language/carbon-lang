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
                                                  Record *Base,
						  bool Root = true) {
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
      OS << "ABSTRACT_STMT(" << NodeName << "(" << R->getName() << ", "
          << Base->getName() << "))\n";
    else
      OS << NodeName << "(" << R->getName() << ", "
          << Base->getName() << ")\n";

    if (Tree.find(R) != Tree.end()) {
      const std::pair<Record *, Record *> &Result
        = EmitStmtNode(Tree, OS, R, false);
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

  if (First) {
    assert (Last && "Got a first node but not a last node for a range!");
    if (Root)
      OS << "LAST_STMT_RANGE(";
    else
      OS << "STMT_RANGE(";
 
    OS << Base->getName() << ", " << First->getName() << ", "
       << Last->getName() << ")\n\n";
  }

  return std::make_pair(First, Last);
}

void ClangStmtNodesEmitter::run(raw_ostream &OS) {
  // Write the preamble
  OS << "#ifndef ABSTRACT_STMT\n";
  OS << "#  define ABSTRACT_STMT(Stmt) Stmt\n";
  OS << "#endif\n";

  OS << "#ifndef STMT_RANGE\n";
  OS << "#  define STMT_RANGE(Base, First, Last)\n";
  OS << "#endif\n\n";

  OS << "#ifndef LAST_STMT_RANGE\n";
  OS << "#  define LAST_STMT_RANGE(Base, First, Last) "
          "STMT_RANGE(Base, First, Last)\n";
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
  OS << "#undef STMT_RANGE\n";
  OS << "#undef LAST_STMT_RANGE\n";
  OS << "#undef ABSTRACT_STMT\n";
}
