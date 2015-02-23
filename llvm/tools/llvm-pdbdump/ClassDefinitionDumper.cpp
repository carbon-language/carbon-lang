//===- ClassDefinitionDumper.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClassDefinitionDumper.h"
#include "FunctionDumper.h"
#include "llvm-pdbdump.h"
#include "TypedefDumper.h"
#include "VariableDumper.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBaseClass.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"
#include "llvm/Support/Format.h"

using namespace llvm;

ClassDefinitionDumper::ClassDefinitionDumper() : PDBSymDumper(true) {}

void ClassDefinitionDumper::start(const PDBSymbolTypeUDT &Class,
                                  raw_ostream &OS, int Indent) {
  OS << "class " << Class.getName() << " {";
  auto Children = Class.findAllChildren();
  if (Children->getChildCount() == 0) {
    OS << "}";
    return;
  }

  // Try to dump symbols organized by member access level.  Public members
  // first, then protected, then private.  This might be slow, so it's worth
  // reconsidering the value of this if performance of large PDBs is a problem.
  // NOTE: Access level of nested types is not recorded in the PDB, so we have
  // a special case for them.
  SymbolGroupByAccess Groups;
  Groups.insert(std::make_pair(0, SymbolGroup()));
  Groups.insert(std::make_pair((int)PDB_MemberAccess::Private, SymbolGroup()));
  Groups.insert(
      std::make_pair((int)PDB_MemberAccess::Protected, SymbolGroup()));
  Groups.insert(std::make_pair((int)PDB_MemberAccess::Public, SymbolGroup()));

  while (auto Child = Children->getNext()) {
    PDB_MemberAccess Access = Child->getRawSymbol().getAccess();
    if (isa<PDBSymbolTypeBaseClass>(*Child))
      continue;

    auto &AccessGroup = Groups.find((int)Access)->second;

    if (auto Func = dyn_cast<PDBSymbolFunc>(Child.get())) {
      if (Func->isCompilerGenerated())
        continue;
      if (Func->getLength() == 0 && !Func->isPureVirtual())
        continue;
      Child.release();
      AccessGroup.Functions.push_back(std::unique_ptr<PDBSymbolFunc>(Func));
    } else if (auto Data = dyn_cast<PDBSymbolData>(Child.get())) {
      Child.release();
      AccessGroup.Data.push_back(std::unique_ptr<PDBSymbolData>(Data));
    } else {
      AccessGroup.Unknown.push_back(std::move(Child));
    }
  }

  int Count = 0;
  Count += dumpAccessGroup((PDB_MemberAccess)0, Groups[0], OS, Indent);
  Count += dumpAccessGroup(PDB_MemberAccess::Public,
                           Groups[(int)PDB_MemberAccess::Public], OS, Indent);
  Count +=
      dumpAccessGroup(PDB_MemberAccess::Protected,
                      Groups[(int)PDB_MemberAccess::Protected], OS, Indent);
  Count += dumpAccessGroup(PDB_MemberAccess::Private,
                           Groups[(int)PDB_MemberAccess::Private], OS, Indent);

  if (Count > 0)
    OS << newline(Indent);
  OS << "}";
}

int ClassDefinitionDumper::dumpAccessGroup(PDB_MemberAccess Access,
                                           const SymbolGroup &Group,
                                           raw_ostream &OS, int Indent) {
  if (Group.Functions.empty() && Group.Data.empty() && Group.Unknown.empty())
    return 0;

  int Count = 0;
  if (Access == PDB_MemberAccess::Private)
    OS << newline(Indent) << "private:";
  else if (Access == PDB_MemberAccess::Protected)
    OS << newline(Indent) << "protected:";
  else if (Access == PDB_MemberAccess::Public)
    OS << newline(Indent) << "public:";
  for (auto iter = Group.Functions.begin(), end = Group.Functions.end();
       iter != end; ++iter) {
    ++Count;
    (*iter)->dump(OS, Indent + 2, *this);
  }
  for (auto iter = Group.Data.begin(), end = Group.Data.end(); iter != end;
       ++iter) {
    ++Count;
    (*iter)->dump(OS, Indent + 2, *this);
  }
  for (auto iter = Group.Unknown.begin(), end = Group.Unknown.end();
       iter != end; ++iter) {
    ++Count;
    (*iter)->dump(OS, Indent + 2, *this);
  }
  return Count;
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeBaseClass &Symbol,
                                 raw_ostream &OS, int Indent) {}

void ClassDefinitionDumper::dump(const PDBSymbolData &Symbol, raw_ostream &OS,
                                 int Indent) {
  VariableDumper Dumper;
  Dumper.start(Symbol, OS, Indent);
}

void ClassDefinitionDumper::dump(const PDBSymbolFunc &Symbol, raw_ostream &OS,
                                 int Indent) {
  FunctionDumper Dumper;
  Dumper.start(Symbol, FunctionDumper::PointerType::None, OS, Indent);
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeVTable &Symbol,
                                 raw_ostream &OS, int Indent) {}

void ClassDefinitionDumper::dump(const PDBSymbolTypeEnum &Symbol,
                                 raw_ostream &OS, int Indent) {
  OS << newline(Indent) << "enum " << Symbol.getName();
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeTypedef &Symbol,
                                 raw_ostream &OS, int Indent) {
  OS << newline(Indent);
  TypedefDumper Dumper;
  Dumper.start(Symbol, OS, Indent);
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeUDT &Symbol,
                                 raw_ostream &OS, int Indent) {}
