//=- ClangSACheckersEmitter.cpp - Generate Clang SA checkers tables -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits Clang Static Analyzer checkers tables.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckersEmitter.h"
#include "Record.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
#include <string>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Static Analyzer Checkers Tables generation
//===----------------------------------------------------------------------===//

/// \brief True if it is specified hidden or a parent package is specified
/// as hidden, otherwise false.
static bool isHidden(const Record &R) {
  if (R.getValueAsBit("Hidden"))
    return true;
  // Not declared as hidden, check the parent package if it is hidden.
  if (DefInit *DI = dynamic_cast<DefInit*>(R.getValueInit("ParentPackage")))
    return isHidden(*DI->getDef());

  return false;
}

static bool isCheckerNamed(const Record *R) {
  return !R->getValueAsString("CheckerName").empty();
}

static std::string getPackageFullName(const Record *R);

static std::string getParentPackageFullName(const Record *R) {
  std::string name;
  if (DefInit *DI = dynamic_cast<DefInit*>(R->getValueInit("ParentPackage")))
    name = getPackageFullName(DI->getDef());
  return name;
}

static std::string getPackageFullName(const Record *R) {
  std::string name = getParentPackageFullName(R);
  if (!name.empty()) name += ".";
  return name + R->getValueAsString("PackageName");
}

static std::string getCheckerFullName(const Record *R) {
  std::string name = getParentPackageFullName(R);
  if (isCheckerNamed(R)) {
    if (!name.empty()) name += ".";
    name += R->getValueAsString("CheckerName");
  }
  return name;
}

static std::string getStringValue(const Record &R, StringRef field) {
  if (StringInit *
        SI = dynamic_cast<StringInit*>(R.getValueInit(field)))
    return SI->getValue();
  return std::string();
}

namespace {
struct GroupInfo {
  std::vector<const Record*> Checkers;
  llvm::DenseSet<const Record *> SubGroups;
  bool Hidden;
  unsigned Index;

  GroupInfo() : Hidden(false) { }
};
}

void ClangSACheckersEmitter::run(raw_ostream &OS) {
  std::vector<Record*> checkers = Records.getAllDerivedDefinitions("Checker");
  llvm::DenseMap<const Record *, unsigned> checkerRecIndexMap;
  for (unsigned i = 0, e = checkers.size(); i != e; ++i)
    checkerRecIndexMap[checkers[i]] = i;
  
  OS << "\n#ifdef GET_CHECKERS\n";
  for (unsigned i = 0, e = checkers.size(); i != e; ++i) {
    const Record &R = *checkers[i];

    OS << "CHECKER(" << "\"";
    std::string name;
    if (isCheckerNamed(&R))
      name = getCheckerFullName(&R);
    OS.write_escaped(name) << "\", ";
    OS << R.getName() << ", ";
    OS << getStringValue(R, "DescFile") << ", ";
    OS << "\"";
    OS.write_escaped(getStringValue(R, "HelpText")) << "\", ";
    // Hidden bit
    if (isHidden(R))
      OS << "true";
    else
      OS << "false";
    OS << ")\n";
  }
  OS << "#endif // GET_CHECKERS\n\n";

  // Invert the mapping of checkers to package/group into a one to many
  // mapping of packages/groups to checkers.
  std::map<std::string, GroupInfo> groupInfoByName;
  llvm::DenseMap<const Record *, GroupInfo *> recordGroupMap;

  std::vector<Record*> packages = Records.getAllDerivedDefinitions("Package");
  for (unsigned i = 0, e = packages.size(); i != e; ++i) {
    Record *R = packages[i];
    std::string fullName = getPackageFullName(R);
    if (!fullName.empty()) {
      GroupInfo &info = groupInfoByName[fullName];
      info.Hidden = isHidden(*R);
      recordGroupMap[R] = &info;
    }
  }

  std::vector<Record*>
      checkerGroups = Records.getAllDerivedDefinitions("CheckerGroup");
  for (unsigned i = 0, e = checkerGroups.size(); i != e; ++i) {
    Record *R = checkerGroups[i];
    std::string name = R->getValueAsString("GroupName");
    if (!name.empty()) {
      GroupInfo &info = groupInfoByName[name];
      recordGroupMap[R] = &info;
    }
  }

  for (unsigned i = 0, e = checkers.size(); i != e; ++i) {
    Record *R = checkers[i];
    Record *package = 0;
    if (DefInit *
          DI = dynamic_cast<DefInit*>(R->getValueInit("ParentPackage")))
      package = DI->getDef();
    if (!isCheckerNamed(R) && !package)
      throw "Checker '" + R->getName() + "' is neither named, nor in a package!";

    if (isCheckerNamed(R)) {
      // Create a pseudo-group to hold this checker.
      std::string fullName = getCheckerFullName(R);
      GroupInfo &info = groupInfoByName[fullName];
      info.Hidden = R->getValueAsBit("Hidden");
      recordGroupMap[R] = &info;
      info.Checkers.push_back(R);
    } else {
      recordGroupMap[package]->Checkers.push_back(R);
    }

    Record *currR = isCheckerNamed(R) ? R : package;
    // Insert the checker and its parent packages into the subgroups set of
    // the corresponding parent package.
    while (DefInit *DI
             = dynamic_cast<DefInit*>(currR->getValueInit("ParentPackage"))) {
      Record *parentPackage = DI->getDef();
      recordGroupMap[parentPackage]->SubGroups.insert(currR);
      currR = parentPackage;
    }
    // Insert the checker into the set of its group.
    if (DefInit *DI = dynamic_cast<DefInit*>(R->getValueInit("Group")))
      recordGroupMap[DI->getDef()]->Checkers.push_back(R);
  }

  unsigned index = 0;
  for (std::map<std::string, GroupInfo>::iterator
         I = groupInfoByName.begin(), E = groupInfoByName.end(); I != E; ++I)
    I->second.Index = index++;

  // Walk through the packages/groups/checkers emitting an array for each
  // set of checkers and an array for each set of subpackages.

  OS << "\n#ifdef GET_MEMBER_ARRAYS\n";
  unsigned maxLen = 0;
  for (std::map<std::string, GroupInfo>::iterator
         I = groupInfoByName.begin(), E = groupInfoByName.end(); I != E; ++I) {
    maxLen = std::max(maxLen, (unsigned)I->first.size());
    
    std::vector<const Record*> &V = I->second.Checkers;
    if (!V.empty()) {
      OS << "static const short CheckerArray" << I->second.Index << "[] = { ";
      for (unsigned i = 0, e = V.size(); i != e; ++i)
        OS << checkerRecIndexMap[V[i]] << ", ";
      OS << "-1 };\n";
    }
    
    llvm::DenseSet<const Record *> &subGroups = I->second.SubGroups;
    if (!subGroups.empty()) {
      OS << "static const short SubPackageArray" << I->second.Index << "[] = { ";
      for (llvm::DenseSet<const Record *>::iterator
             I = subGroups.begin(), E = subGroups.end(); I != E; ++I) {
        OS << recordGroupMap[*I]->Index << ", ";
      }
      OS << "-1 };\n";
    }
  }
  OS << "#endif // GET_MEMBER_ARRAYS\n\n";

  OS << "\n#ifdef GET_CHECKNAME_TABLE\n";
  for (std::map<std::string, GroupInfo>::iterator
         I = groupInfoByName.begin(), E = groupInfoByName.end(); I != E; ++I) {
    // Group option string.
    OS << "  { \"";
    OS.write_escaped(I->first) << "\","
                               << std::string(maxLen-I->first.size()+1, ' ');
    
    if (I->second.Checkers.empty())
      OS << "0, ";
    else
      OS << "CheckerArray" << I->second.Index << ", ";
    
    // Subgroups.
    if (I->second.SubGroups.empty())
      OS << "0, ";
    else
      OS << "SubPackageArray" << I->second.Index << ", ";

    OS << (I->second.Hidden ? "true" : "false");

    OS << " },\n";
  }
  OS << "#endif // GET_CHECKNAME_TABLE\n\n";
}
