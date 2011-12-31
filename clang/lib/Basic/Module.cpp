//===--- Module.h - Describe a module ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Module class, which describes a module in the source
// code.
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/Module.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
using namespace clang;

Module::~Module() {
  for (llvm::StringMap<Module *>::iterator I = SubModules.begin(), 
                                        IEnd = SubModules.end();
       I != IEnd; ++I) {
    delete I->getValue();
  }
  
}

/// \brief Determine whether a translation unit built using the current
/// language options has the given feature.
static bool hasFeature(StringRef Feature, const LangOptions &LangOpts) {
  return llvm::StringSwitch<bool>(Feature)
           .Case("blocks", LangOpts.Blocks)
           .Case("cplusplus", LangOpts.CPlusPlus)
           .Case("cplusplus11", LangOpts.CPlusPlus0x)
           .Case("objc", LangOpts.ObjC1)
           .Case("objc_arc", LangOpts.ObjCAutoRefCount)
           .Default(false);
}

bool 
Module::isAvailable(const LangOptions &LangOpts, StringRef &Feature) const {
  if (IsAvailable)
    return true;

  for (const Module *Current = this; Current; Current = Current->Parent) {
    for (unsigned I = 0, N = Current->Requires.size(); I != N; ++I) {
      if (!hasFeature(Current->Requires[I], LangOpts)) {
        Feature = Current->Requires[I];
        return false;
      }
    }
  }

  llvm_unreachable("could not find a reason why module is unavailable");
  return false;
}

bool Module::isSubModuleOf(Module *Other) const {
  const Module *This = this;
  do {
    if (This == Other)
      return true;
    
    This = This->Parent;
  } while (This);
  
  return false;
}

const Module *Module::getTopLevelModule() const {
  const Module *Result = this;
  while (Result->Parent)
    Result = Result->Parent;
  
  return Result;
}

std::string Module::getFullModuleName() const {
  llvm::SmallVector<StringRef, 2> Names;
  
  // Build up the set of module names (from innermost to outermost).
  for (const Module *M = this; M; M = M->Parent)
    Names.push_back(M->Name);
  
  std::string Result;
  for (llvm::SmallVector<StringRef, 2>::reverse_iterator I = Names.rbegin(),
                                                      IEnd = Names.rend(); 
       I != IEnd; ++I) {
    if (!Result.empty())
      Result += '.';
    
    Result += *I;
  }
  
  return Result;
}

const DirectoryEntry *Module::getUmbrellaDir() const {
  if (const FileEntry *Header = getUmbrellaHeader())
    return Header->getDir();
  
  return Umbrella.dyn_cast<const DirectoryEntry *>();
}

void Module::addRequirement(StringRef Feature, const LangOptions &LangOpts) {
  Requires.push_back(Feature);

  // If this feature is currently available, we're done.
  if (hasFeature(Feature, LangOpts))
    return;

  if (!IsAvailable)
    return;

  llvm::SmallVector<Module *, 2> Stack;
  Stack.push_back(this);
  while (!Stack.empty()) {
    Module *Current = Stack.back();
    Stack.pop_back();

    if (!Current->IsAvailable)
      continue;

    Current->IsAvailable = false;
    for (llvm::StringMap<Module *>::iterator Sub = Current->SubModules.begin(),
                                          SubEnd = Current->SubModules.end();
         Sub != SubEnd; ++Sub) {
      if (Sub->second->IsAvailable)
        Stack.push_back(Sub->second);
    }
  }
}

static void printModuleId(llvm::raw_ostream &OS, const ModuleId &Id) {
  for (unsigned I = 0, N = Id.size(); I != N; ++I) {
    if (I)
      OS << ".";
    OS << Id[I].first;
  }
}

void Module::print(llvm::raw_ostream &OS, unsigned Indent) const {
  OS.indent(Indent);
  if (IsFramework)
    OS << "framework ";
  if (IsExplicit)
    OS << "explicit ";
  OS << "module " << Name << " {\n";

  if (!Requires.empty()) {
    OS.indent(Indent + 2);
    OS << "requires ";
    for (unsigned I = 0, N = Requires.size(); I != N; ++I) {
      if (I)
        OS << ", ";
      OS << Requires[I];
    }
    OS << "\n";
  }
  
  if (const FileEntry *UmbrellaHeader = getUmbrellaHeader()) {
    OS.indent(Indent + 2);
    OS << "umbrella header \"";
    OS.write_escaped(UmbrellaHeader->getName());
    OS << "\"\n";
  } else if (const DirectoryEntry *UmbrellaDir = getUmbrellaDir()) {
    OS.indent(Indent + 2);
    OS << "umbrella \"";
    OS.write_escaped(UmbrellaDir->getName());
    OS << "\"\n";    
  }
  
  for (unsigned I = 0, N = Headers.size(); I != N; ++I) {
    OS.indent(Indent + 2);
    OS << "header \"";
    OS.write_escaped(Headers[I]->getName());
    OS << "\"\n";
  }
  
  for (llvm::StringMap<Module *>::const_iterator MI = SubModules.begin(), 
                                              MIEnd = SubModules.end();
       MI != MIEnd; ++MI)
    MI->getValue()->print(OS, Indent + 2);
  
  for (unsigned I = 0, N = Exports.size(); I != N; ++I) {
    OS.indent(Indent + 2);
    OS << "export ";
    if (Module *Restriction = Exports[I].getPointer()) {
      OS << Restriction->getFullModuleName();
      if (Exports[I].getInt())
        OS << ".*";
    } else {
      OS << "*";
    }
    OS << "\n";
  }

  for (unsigned I = 0, N = UnresolvedExports.size(); I != N; ++I) {
    OS.indent(Indent + 2);
    OS << "export ";
    printModuleId(OS, UnresolvedExports[I].Id);
    if (UnresolvedExports[I].Wildcard) {
      if (UnresolvedExports[I].Id.empty())
        OS << "*";
      else
        OS << ".*";
    }
    OS << "\n";
  }

  if (InferSubmodules) {
    OS.indent(Indent + 2);
    if (InferExplicitSubmodules)
      OS << "explicit ";
    OS << "module * {\n";
    if (InferExportWildcard) {
      OS.indent(Indent + 4);
      OS << "export *\n";
    }
    OS.indent(Indent + 2);
    OS << "}\n";
  }
  
  OS.indent(Indent);
  OS << "}\n";
}

void Module::dump() const {
  print(llvm::errs());
}


