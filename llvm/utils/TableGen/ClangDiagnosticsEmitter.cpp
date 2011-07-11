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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/VectorExtras.h"
#include <map>
#include <algorithm>
#include <functional>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Diagnostic category computation code.
//===----------------------------------------------------------------------===//

namespace {
class DiagGroupParentMap {
  RecordKeeper &Records;
  std::map<const Record*, std::vector<Record*> > Mapping;
public:
  DiagGroupParentMap(RecordKeeper &records) : Records(records) {
    std::vector<Record*> DiagGroups
      = Records.getAllDerivedDefinitions("DiagGroup");
    for (unsigned i = 0, e = DiagGroups.size(); i != e; ++i) {
      std::vector<Record*> SubGroups =
        DiagGroups[i]->getValueAsListOfDefs("SubGroups");
      for (unsigned j = 0, e = SubGroups.size(); j != e; ++j)
        Mapping[SubGroups[j]].push_back(DiagGroups[i]);
    }
  }
  
  const std::vector<Record*> &getParents(const Record *Group) {
    return Mapping[Group];
  }
};
} // end anonymous namespace.


static std::string
getCategoryFromDiagGroup(const Record *Group,
                         DiagGroupParentMap &DiagGroupParents) {
  // If the DiagGroup has a category, return it.
  std::string CatName = Group->getValueAsString("CategoryName");
  if (!CatName.empty()) return CatName;
  
  // The diag group may the subgroup of one or more other diagnostic groups,
  // check these for a category as well.
  const std::vector<Record*> &Parents = DiagGroupParents.getParents(Group);
  for (unsigned i = 0, e = Parents.size(); i != e; ++i) {
    CatName = getCategoryFromDiagGroup(Parents[i], DiagGroupParents);
    if (!CatName.empty()) return CatName;
  }
  return "";
}

/// getDiagnosticCategory - Return the category that the specified diagnostic
/// lives in.
static std::string getDiagnosticCategory(const Record *R,
                                         DiagGroupParentMap &DiagGroupParents) {
  // If the diagnostic is in a group, and that group has a category, use it.
  if (const DefInit *Group =
      dynamic_cast<const DefInit*>(R->getValueInit("Group"))) {
    // Check the diagnostic's diag group for a category.
    std::string CatName = getCategoryFromDiagGroup(Group->getDef(),
                                                   DiagGroupParents);
    if (!CatName.empty()) return CatName;
  }
  
  // If the diagnostic itself has a category, get it.
  return R->getValueAsString("CategoryName");
}

namespace {
  class DiagCategoryIDMap {
    RecordKeeper &Records;
    StringMap<unsigned> CategoryIDs;
    std::vector<std::string> CategoryStrings;
  public:
    DiagCategoryIDMap(RecordKeeper &records) : Records(records) {
      DiagGroupParentMap ParentInfo(Records);
      
      // The zero'th category is "".
      CategoryStrings.push_back("");
      CategoryIDs[""] = 0;
      
      std::vector<Record*> Diags =
      Records.getAllDerivedDefinitions("Diagnostic");
      for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
        std::string Category = getDiagnosticCategory(Diags[i], ParentInfo);
        if (Category.empty()) continue;  // Skip diags with no category.
        
        unsigned &ID = CategoryIDs[Category];
        if (ID != 0) continue;  // Already seen.
        
        ID = CategoryStrings.size();
        CategoryStrings.push_back(Category);
      }
    }
    
    unsigned getID(StringRef CategoryString) {
      return CategoryIDs[CategoryString];
    }
    
    typedef std::vector<std::string>::iterator iterator;
    iterator begin() { return CategoryStrings.begin(); }
    iterator end() { return CategoryStrings.end(); }
  };
} // end anonymous namespace.


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
    OS << "#endif\n\n";
  }

  const std::vector<Record*> &Diags =
    Records.getAllDerivedDefinitions("Diagnostic");
  
  DiagCategoryIDMap CategoryIDs(Records);
  DiagGroupParentMap DGParentMap(Records);

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
    if (const DefInit *DI =
        dynamic_cast<const DefInit*>(R.getValueInit("Group"))) {
      OS << ", \"";
      OS.write_escaped(DI->getDef()->getValueAsString("GroupName")) << '"';
    } else {
      OS << ", \"\"";
    }

    // SFINAE bit
    if (R.getValueAsBit("SFINAE"))
      OS << ", true";
    else
      OS << ", false";

    // Access control bit
    if (R.getValueAsBit("AccessControl"))
      OS << ", true";
    else
      OS << ", false";

    // Category number.
    OS << ", " << CategoryIDs.getID(getDiagnosticCategory(&R, DGParentMap));

    // Brief
    OS << ", \"";
    OS.write_escaped(R.getValueAsString("Brief")) << '"';

    // Explanation 
    OS << ", \"";
    OS.write_escaped(R.getValueAsString("Explanation")) << '"';
    OS << ")\n";
  }
}

//===----------------------------------------------------------------------===//
// Warning Group Tables generation
//===----------------------------------------------------------------------===//

static std::string getDiagCategoryEnum(llvm::StringRef name) {
  if (name.empty())
    return "DiagCat_None";
  llvm::SmallString<256> enumName = llvm::StringRef("DiagCat_");
  for (llvm::StringRef::iterator I = name.begin(), E = name.end(); I != E; ++I)
    enumName += isalnum(*I) ? *I : '_';
  return enumName.str();
}

namespace {
struct GroupInfo {
  std::vector<const Record*> DiagsInGroup;
  std::vector<std::string> SubGroups;
  unsigned IDNo;
};
} // end anonymous namespace.

void ClangDiagGroupsEmitter::run(raw_ostream &OS) {
  // Compute a mapping from a DiagGroup to all of its parents.
  DiagGroupParentMap DGParentMap(Records);
  
  // Invert the 1-[0/1] mapping of diags to group into a one to many mapping of
  // groups to diags in the group.
  std::map<std::string, GroupInfo> DiagsInGroup;
  
  std::vector<Record*> Diags =
    Records.getAllDerivedDefinitions("Diagnostic");
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    const Record *R = Diags[i];
    const DefInit *DI = dynamic_cast<const DefInit*>(R->getValueInit("Group"));
    if (DI == 0) continue;
    std::string GroupName = DI->getDef()->getValueAsString("GroupName");
    DiagsInGroup[GroupName].DiagsInGroup.push_back(R);
  }
  
  // Add all DiagGroup's to the DiagsInGroup list to make sure we pick up empty
  // groups (these are warnings that GCC supports that clang never produces).
  std::vector<Record*> DiagGroups
    = Records.getAllDerivedDefinitions("DiagGroup");
  for (unsigned i = 0, e = DiagGroups.size(); i != e; ++i) {
    Record *Group = DiagGroups[i];
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
      OS << "static const short DiagSubGroup" << I->second.IDNo << "[] = { ";
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
    OS << "  { ";
    OS << I->first.size() << ", ";
    OS << "\"";
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
  
  // Emit the category table next.
  DiagCategoryIDMap CategoriesByID(Records);
  OS << "\n#ifdef GET_CATEGORY_TABLE\n";
  for (DiagCategoryIDMap::iterator I = CategoriesByID.begin(),
       E = CategoriesByID.end(); I != E; ++I)
    OS << "CATEGORY(\"" << *I << "\", " << getDiagCategoryEnum(*I) << ")\n";
  OS << "#endif // GET_CATEGORY_TABLE\n\n";
}

//===----------------------------------------------------------------------===//
// Diagnostic name index generation
//===----------------------------------------------------------------------===//

namespace {
struct RecordIndexElement
{
  RecordIndexElement() {}
  explicit RecordIndexElement(Record const &R):
    Name(R.getName()) {}
  
  std::string Name;
};

struct RecordIndexElementSorter :
  public std::binary_function<RecordIndexElement, RecordIndexElement, bool> {
  
  bool operator()(RecordIndexElement const &Lhs,
                  RecordIndexElement const &Rhs) const {
    return Lhs.Name < Rhs.Name;
  }
  
};

} // end anonymous namespace.

void ClangDiagsIndexNameEmitter::run(raw_ostream &OS) {
  const std::vector<Record*> &Diags =
    Records.getAllDerivedDefinitions("Diagnostic");
  
  std::vector<RecordIndexElement> Index;
  Index.reserve(Diags.size());
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    const Record &R = *(Diags[i]);    
    Index.push_back(RecordIndexElement(R));
  }
  
  std::sort(Index.begin(), Index.end(), RecordIndexElementSorter());
  
  for (unsigned i = 0, e = Index.size(); i != e; ++i) {
    const RecordIndexElement &R = Index[i];
    
    OS << "DIAG_NAME_INDEX(" << R.Name << ")\n";
  }
}
