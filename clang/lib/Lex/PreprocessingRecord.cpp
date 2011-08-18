//===--- PreprocessingRecord.cpp - Record of Preprocessing ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the PreprocessingRecord class, which maintains a record
//  of what occurred during preprocessing, and its helpers.
//
//===----------------------------------------------------------------------===//
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Token.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Capacity.h"

using namespace clang;

ExternalPreprocessingRecordSource::~ExternalPreprocessingRecordSource() { }


InclusionDirective::InclusionDirective(PreprocessingRecord &PPRec,
                                       InclusionKind Kind, 
                                       StringRef FileName, 
                                       bool InQuotes, const FileEntry *File, 
                                       SourceRange Range)
  : PreprocessingDirective(InclusionDirectiveKind, Range), 
    InQuotes(InQuotes), Kind(Kind), File(File) 
{ 
  char *Memory 
    = (char*)PPRec.Allocate(FileName.size() + 1, llvm::alignOf<char>());
  memcpy(Memory, FileName.data(), FileName.size());
  Memory[FileName.size()] = 0;
  this->FileName = StringRef(Memory, FileName.size());
}

void PreprocessingRecord::MaybeLoadPreallocatedEntities() const {
  if (!ExternalSource || LoadedPreallocatedEntities)
    return;
  
  LoadedPreallocatedEntities = true;
  ExternalSource->ReadPreprocessedEntities();
}

PreprocessingRecord::PreprocessingRecord(bool IncludeNestedMacroExpansions)
  : IncludeNestedMacroExpansions(IncludeNestedMacroExpansions),
    ExternalSource(0), LoadedPreallocatedEntities(false)
{
}

PreprocessingRecord::iterator 
PreprocessingRecord::begin(bool OnlyLocalEntities) {
  if (OnlyLocalEntities)
    return iterator(this, 0);
  
  MaybeLoadPreallocatedEntities();
  return iterator(this, -(int)LoadedPreprocessedEntities.size());
}

PreprocessingRecord::iterator PreprocessingRecord::end(bool OnlyLocalEntities) {
  if (!OnlyLocalEntities)
    MaybeLoadPreallocatedEntities();
  
  return iterator(this, PreprocessedEntities.size());
}

void PreprocessingRecord::addPreprocessedEntity(PreprocessedEntity *Entity) {
  PreprocessedEntities.push_back(Entity);
}

void PreprocessingRecord::SetExternalSource(
                                    ExternalPreprocessingRecordSource &Source) {
  assert(!ExternalSource &&
         "Preprocessing record already has an external source");
  ExternalSource = &Source;
}

unsigned PreprocessingRecord::allocateLoadedEntities(unsigned NumEntities) {
  unsigned Result = LoadedPreprocessedEntities.size();
  LoadedPreprocessedEntities.resize(LoadedPreprocessedEntities.size() 
                                    + NumEntities);
  return Result;
}

void 
PreprocessingRecord::setLoadedPreallocatedEntity(unsigned Index, 
                                                 PreprocessedEntity *Entity) {
  assert(Index < LoadedPreprocessedEntities.size() &&
         "Out-of-bounds preallocated entity");
  LoadedPreprocessedEntities[Index] = Entity;
}

void PreprocessingRecord::RegisterMacroDefinition(MacroInfo *Macro, 
                                                  MacroDefinition *MD) {
  MacroDefinitions[Macro] = MD;
}

MacroDefinition *PreprocessingRecord::findMacroDefinition(const MacroInfo *MI) {
  llvm::DenseMap<const MacroInfo *, MacroDefinition *>::iterator Pos
    = MacroDefinitions.find(MI);
  if (Pos == MacroDefinitions.end())
    return 0;
  
  return Pos->second;
}

void PreprocessingRecord::MacroExpands(const Token &Id, const MacroInfo* MI,
                                       SourceRange Range) {
  if (!IncludeNestedMacroExpansions && Id.getLocation().isMacroID())
    return;

  if (MacroDefinition *Def = findMacroDefinition(MI))
    PreprocessedEntities.push_back(
                       new (*this) MacroExpansion(Id.getIdentifierInfo(),
                                                  Range, Def));
}

void PreprocessingRecord::MacroDefined(const Token &Id,
                                       const MacroInfo *MI) {
  SourceRange R(MI->getDefinitionLoc(), MI->getDefinitionEndLoc());
  MacroDefinition *Def
      = new (*this) MacroDefinition(Id.getIdentifierInfo(),
                                    MI->getDefinitionLoc(),
                                    R);
  MacroDefinitions[MI] = Def;
  PreprocessedEntities.push_back(Def);
}

void PreprocessingRecord::MacroUndefined(const Token &Id,
                                         const MacroInfo *MI) {
  llvm::DenseMap<const MacroInfo *, MacroDefinition *>::iterator Pos
    = MacroDefinitions.find(MI);
  if (Pos != MacroDefinitions.end())
    MacroDefinitions.erase(Pos);
}

void PreprocessingRecord::InclusionDirective(
    SourceLocation HashLoc,
    const clang::Token &IncludeTok,
    StringRef FileName,
    bool IsAngled,
    const FileEntry *File,
    clang::SourceLocation EndLoc,
    StringRef SearchPath,
    StringRef RelativePath) {
  InclusionDirective::InclusionKind Kind = InclusionDirective::Include;
  
  switch (IncludeTok.getIdentifierInfo()->getPPKeywordID()) {
  case tok::pp_include: 
    Kind = InclusionDirective::Include; 
    break;
    
  case tok::pp_import: 
    Kind = InclusionDirective::Import; 
    break;
    
  case tok::pp_include_next: 
    Kind = InclusionDirective::IncludeNext; 
    break;
    
  case tok::pp___include_macros: 
    Kind = InclusionDirective::IncludeMacros;
    break;
    
  default:
    llvm_unreachable("Unknown include directive kind");
    return;
  }
  
  clang::InclusionDirective *ID
    = new (*this) clang::InclusionDirective(*this, Kind, FileName, !IsAngled, 
                                            File, SourceRange(HashLoc, EndLoc));
  PreprocessedEntities.push_back(ID);
}

size_t PreprocessingRecord::getTotalMemory() const {
  return BumpAlloc.getTotalMemory()
    + llvm::capacity_in_bytes(MacroDefinitions)
    + llvm::capacity_in_bytes(PreprocessedEntities)
    + llvm::capacity_in_bytes(LoadedPreprocessedEntities);
}
