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
#include "llvm/ADT/VectorExtras.h"
#include "llvm/ADT/DenseSet.h"
#include <set>
#include <map>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Generic routines for all Clang TableGen backens.
//===----------------------------------------------------------------------===//

typedef std::vector<Record*> RecordVector;
typedef std::vector<Record*> SuperClassVector;
typedef std::vector<RecordVal> RecordValVector;

static const RecordVal* findRecordVal(const Record& R, const std::string &key) {  
  const RecordValVector &Vals = R.getValues();
  for (RecordValVector::const_iterator I=Vals.begin(), E=Vals.end(); I!=E; ++I)
    if ((*I).getName() == key)
      return &*I;
  
  return 0;
}

static const Record* getDiagKind(const Record* DiagClass, const Record &R) {  
  const SuperClassVector &SC = R.getSuperClasses();
  for (SuperClassVector::const_iterator I=SC.begin(), E=SC.end(); I!=E; ++I)
    if ((*I)->isSubClassOf(DiagClass) && 
        (*I)->getName() != "DiagnosticControlled")
      return *I;
  
  return 0;
}

static void EmitEscaped(std::ostream& OS, const std::string &s) {
  for (std::string::const_iterator I=s.begin(), E=s.end(); I!=E; ++I)
    switch (*I) {
      default: OS << *I; break;
      case '\"': OS << "\\" << *I; break;
      case '\\': OS << "\\\\"; break;
    }
}

static void EmitAllCaps(std::ostream& OS, const std::string &s) {
  for (std::string::const_iterator I=s.begin(), E=s.end(); I!=E; ++I)
    OS << char(toupper(*I));  
}

//===----------------------------------------------------------------------===//
// Warning Tables (.inc file) generation.
//===----------------------------------------------------------------------===//

static void ProcessDiag(std::ostream& OS, const Record* DiagClass,
                        const Record& R) {

  const Record* DiagKind = getDiagKind(DiagClass, R);
  if (!DiagKind)
    return;

  OS << "DIAG(" << R.getName() << ", ";
  EmitAllCaps(OS, DiagKind->getName());
  
  const RecordVal* Text = findRecordVal(R, "Text");
  assert(Text && "No 'Text' entry in Diagnostic.");
  const StringInit* TextVal = dynamic_cast<const StringInit*>(Text->getValue());
  assert(TextVal && "Value 'Text' must be a string.");
  OS << ", \"";
  EmitEscaped(OS, TextVal->getValue());
  OS << "\")\n";
}

void ClangDiagsDefsEmitter::run(std::ostream &OS) {
  const RecordVector &Diags = Records.getAllDerivedDefinitions("Diagnostic");
  
  const Record* DiagClass = Records.getClass("Diagnostic");
  assert(DiagClass && "No Diagnostic class defined.");  
  
  // Write the #if guard
  if (!Component.empty()) {
    OS << "#ifdef ";
    EmitAllCaps(OS, Component);
    OS << "START\n__";
    EmitAllCaps(OS, Component);
    OS << "START = DIAG_START_";
    EmitAllCaps(OS, Component);
    OS << ",\n#undef ";
    EmitAllCaps(OS, Component);
    OS << "START\n#endif\n";
  }
  
  for (RecordVector::const_iterator I=Diags.begin(), E=Diags.end(); I!=E; ++I) {
    if (!Component.empty()) {
      const RecordVal* V = findRecordVal(**I, "Component");
      if (!V)
        continue;

      const StringInit* SV = dynamic_cast<const StringInit*>(V->getValue());
      if (!SV || SV->getValue() != Component)
        continue;
    }
    
    ProcessDiag(OS, DiagClass, **I);
  }
}

//===----------------------------------------------------------------------===//
// Warning Group Tables generation.
//===----------------------------------------------------------------------===//

static const std::string &getOptName(const Record *R) {
  const RecordVal *V = findRecordVal(*R, "Name");
  assert(V && "Options must have a 'Name' value.");
  const StringInit* SV = dynamic_cast<const StringInit*>(V->getValue());
  assert(SV && "'Name' entry must be a string.");
  return SV->getValue();
}  

namespace {
struct VISIBILITY_HIDDEN CompareOptName {  
  bool operator()(const Record* A, const Record* B) {
    return getOptName(A) < getOptName(B);    
  }
};
}

typedef std::set<const Record*> DiagnosticSet;
typedef std::map<const Record*, DiagnosticSet, CompareOptName> OptionMap;
typedef llvm::DenseSet<const ListInit*> VisitedLists;

static void BuildGroup(DiagnosticSet& DS, VisitedLists &Visited, const Init* X);

static void BuildGroup(DiagnosticSet &DS, VisitedLists &Visited,
                       const ListInit* LV) {

  // Simple hack to prevent including a list multiple times.  This may be useful
  // if one declares an Option by including a bunch of other Options that
  // include other Options, etc.
  if (Visited.count(LV))
    return;
  
  Visited.insert(LV);
  
  // Iterate through the list and grab all DiagnosticControlled.
  for (ListInit::const_iterator I = LV->begin(), E = LV->end(); I!=E; ++I)
    BuildGroup(DS, Visited, *I);
}

static void BuildGroup(DiagnosticSet& DS, VisitedLists &Visited,
                       const Record *Def) {

  // If an Option includes another Option, inline the Diagnostics of the
  // included Option.
  if (Def->isSubClassOf("Option")) {
    if (const RecordVal* V = findRecordVal(*Def, "Members"))
      if (const ListInit* LV = dynamic_cast<const ListInit*>(V->getValue()))
        BuildGroup(DS, Visited, LV);

    return;
  }
  
  if (Def->isSubClassOf("DiagnosticControlled"))
    DS.insert(Def);
}

static void BuildGroup(DiagnosticSet& DS, VisitedLists &Visited,
                       const Init* X) {

  if (const DefInit *D = dynamic_cast<const DefInit*>(X))
    BuildGroup(DS, Visited, D->getDef());
  
  // We may have some other cases here in the future.
}


void ClangOptionsEmitter::run(std::ostream &OS) {
  // Build up a map from options to controlled diagnostics.
  OptionMap OM;  
       
  const RecordVector &Opts = Records.getAllDerivedDefinitions("Option");
  for (RecordVector::const_iterator I=Opts.begin(), E=Opts.end(); I!=E; ++I)
    if (const RecordVal* V = findRecordVal(**I, "Members"))
      if (const ListInit* LV = dynamic_cast<const ListInit*>(V->getValue())) {        
        VisitedLists Visited;
        BuildGroup(OM[*I], Visited, LV);
      }
  
  // Iterate through the OptionMap and emit the declarations.
  for (OptionMap::iterator I = OM.begin(), E = OM.end(); I!=E; ++I) {    
    // Output the option.
    OS << "static const diag::kind " << I->first->getName() << "[] = { ";
    
    DiagnosticSet &DS = I->second;
    bool first = true;
    for (DiagnosticSet::iterator I2 = DS.begin(), E2 = DS.end(); I2!=E2; ++I2) {
      if (first)
        first = false;
      else
        OS << ", ";
        
      OS << "diag::" << (*I2)->getName();
    }
    OS << " };\n";
  }
    
  // Now emit the OptionTable table.
  OS << "\nstatic const WarningOption OptionTable[] = {";
  bool first = true;
  for (OptionMap::iterator I = OM.begin(), E = OM.end(); I!=E; ++I) {
    if (first)
      first = false;
    else
      OS << ',';
    
    OS << "\n  {\"" << getOptName(I->first)
       << "\", DIAGS(" << I->first->getName() << ")}";
  }
  OS << "\n};\n";
}
