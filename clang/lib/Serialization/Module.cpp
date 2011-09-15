//===--- Module.cpp - Module description ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Module class, which describes a module that has
//  been loaded from an AST file.
//
//===----------------------------------------------------------------------===//
#include "clang/Serialization/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "ASTReaderInternals.h"

using namespace clang;
using namespace serialization;
using namespace reader;

Module::Module(ModuleKind Kind)
  : Kind(Kind), DirectlyImported(false), SizeInBits(0), 
    LocalNumSLocEntries(0), SLocEntryBaseID(0),
    SLocEntryBaseOffset(0), SLocEntryOffsets(0),
    SLocFileOffsets(0), LocalNumIdentifiers(0), 
    IdentifierOffsets(0), BaseIdentifierID(0), IdentifierTableData(0),
    IdentifierLookupTable(0), BasePreprocessedEntityID(0),
    PreprocessedEntityOffsets(0), NumPreprocessedEntities(0),
    LocalNumHeaderFileInfos(0), 
    HeaderFileInfoTableData(0), HeaderFileInfoTable(0),
    HeaderFileFrameworkStrings(0),
    LocalNumSelectors(0), SelectorOffsets(0), BaseSelectorID(0),
    SelectorLookupTableData(0), SelectorLookupTable(0), LocalNumDecls(0),
    DeclOffsets(0), BaseDeclID(0),
    LocalNumCXXBaseSpecifiers(0), CXXBaseSpecifiersOffsets(0),
    LocalNumTypes(0), TypeOffsets(0), BaseTypeIndex(0), StatCache(0)
{}

Module::~Module() {
  for (DeclContextInfosMap::iterator I = DeclContextInfos.begin(),
       E = DeclContextInfos.end();
       I != E; ++I) {
    if (I->second.NameLookupTableData)
      delete static_cast<ASTDeclContextNameLookupTable*>(
                                                 I->second.NameLookupTableData);
  }
  
  delete static_cast<ASTIdentifierLookupTable *>(IdentifierLookupTable);
  delete static_cast<HeaderFileInfoLookupTable *>(HeaderFileInfoTable);
  delete static_cast<ASTSelectorLookupTable *>(SelectorLookupTable);
}

template<typename Key, typename Offset, unsigned InitialCapacity>
static void 
dumpLocalRemap(StringRef Name,
               const ContinuousRangeMap<Key, Offset, InitialCapacity> &Map) {
  if (Map.begin() == Map.end())
    return;
  
  typedef ContinuousRangeMap<Key, Offset, InitialCapacity> MapType;
  llvm::errs() << "  " << Name << ":\n";
  for (typename MapType::const_iterator I = Map.begin(), IEnd = Map.end(); 
       I != IEnd; ++I) {
    llvm::errs() << "    " << I->first << " -> " << I->second << "\n";
  }
}

void Module::dump() {
  llvm::errs() << "\nModule: " << FileName << "\n";
  if (!Imports.empty()) {
    llvm::errs() << "  Imports: ";
    for (unsigned I = 0, N = Imports.size(); I != N; ++I) {
      if (I)
        llvm::errs() << ", ";
      llvm::errs() << Imports[I]->FileName;
    }
    llvm::errs() << "\n";
  }
  
  // Remapping tables.
  llvm::errs() << "  Base source location offset: " << SLocEntryBaseOffset 
               << '\n';
  dumpLocalRemap("Source location offset local -> global map", SLocRemap);
  
  llvm::errs() << "  Base identifier ID: " << BaseIdentifierID << '\n'
               << "  Number of identifiers: " << LocalNumIdentifiers << '\n';
  dumpLocalRemap("Identifier ID local -> global map", IdentifierRemap);
  
  llvm::errs() << "  Base selector ID: " << BaseSelectorID << '\n'
               << "  Number of selectors: " << LocalNumSelectors << '\n';
  dumpLocalRemap("Selector ID local -> global map", SelectorRemap);
  
  llvm::errs() << "  Base preprocessed entity ID: " << BasePreprocessedEntityID
               << '\n'  
               << "  Number of preprocessed entities: " 
               << NumPreprocessedEntities << '\n';
  dumpLocalRemap("Preprocessed entity ID local -> global map", 
                 PreprocessedEntityRemap);
  
  llvm::errs() << "  Base type index: " << BaseTypeIndex << '\n'
               << "  Number of types: " << LocalNumTypes << '\n';
  dumpLocalRemap("Type index local -> global map", TypeRemap);
  
  llvm::errs() << "  Base decl ID: " << BaseDeclID << '\n'
               << "  Number of decls: " << LocalNumDecls << '\n';
  dumpLocalRemap("Decl ID local -> global map", DeclRemap);
}
