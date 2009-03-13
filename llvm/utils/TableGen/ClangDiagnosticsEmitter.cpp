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
#include "llvm/Support/Streams.h"
#include "llvm/ADT/VectorExtras.h"

using namespace llvm;
typedef std::vector<Record*> RecordVector;
typedef std::vector<Record*> SuperClassVector;
typedef std::vector<RecordVal> RecordValVector;

static const RecordVal* findRecordVal(const RecordValVector& Vals,
                                      const std::string &key) {
  for (RecordValVector::const_iterator I=Vals.begin(), E=Vals.end(); I!=E; ++I)
    if ((*I).getName() == key)
      return &*I;
  
  return 0;
}

static const Record* getDiagKind(const Record* DiagClass, const Record &R) {  
  const SuperClassVector &SC = R.getSuperClasses();
  for (SuperClassVector::const_iterator I=SC.begin(), E=SC.end(); I!=E; ++I)
    if ((*I)->isSubClassOf(DiagClass))
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

static void ProcessDiag(std::ostream& OS, const Record* DiagClass,
                        const Record& R) {

  const Record* DiagKind = getDiagKind(DiagClass, R);
  if (!DiagKind)
    return;

  OS << "DIAG(" << R.getName() << ", ";
  
  const std::string &s = DiagKind->getName();
  for (std::string::const_iterator I=s.begin(), E=s.end(); I!=E; ++I)
    OS << char(toupper(*I));
  
  const RecordVal* Text = findRecordVal(R.getValues(), "Text");
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
  
  for (RecordVector::const_iterator I=Diags.begin(), E=Diags.end(); I!=E; ++I) {
    // FIXME: Compare the component.
    ProcessDiag(OS, DiagClass, **I);
  }
}