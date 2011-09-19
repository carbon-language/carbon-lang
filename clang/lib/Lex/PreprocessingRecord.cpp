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

PreprocessingRecord::PreprocessingRecord(SourceManager &SM,
                                         bool IncludeNestedMacroExpansions)
  : SourceMgr(SM), IncludeNestedMacroExpansions(IncludeNestedMacroExpansions),
    ExternalSource(0)
{
}

/// \brief Returns a pair of [Begin, End) iterators of preprocessed entities
/// that source range \arg R encompasses.
std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
PreprocessingRecord::getPreprocessedEntitiesInRange(SourceRange Range) {
  if (Range.isInvalid())
    return std::make_pair(iterator(this, 0), iterator(this, 0));
  assert(!SourceMgr.isBeforeInTranslationUnit(Range.getEnd(),Range.getBegin()));

  std::pair<unsigned, unsigned>
    Local = findLocalPreprocessedEntitiesInRange(Range);

  // Check if range spans local entities.
  if (!ExternalSource || SourceMgr.isLocalSourceLocation(Range.getBegin()))
    return std::make_pair(iterator(this, Local.first),
                          iterator(this, Local.second));

  std::pair<unsigned, unsigned>
    Loaded = ExternalSource->findPreprocessedEntitiesInRange(Range);

  // Check if range spans local entities.
  if (Loaded.first == Loaded.second)
    return std::make_pair(iterator(this, Local.first),
                          iterator(this, Local.second));

  unsigned TotalLoaded = LoadedPreprocessedEntities.size();

  // Check if range spans loaded entities.
  if (Local.first == Local.second)
    return std::make_pair(iterator(this, int(Loaded.first)-TotalLoaded),
                          iterator(this, int(Loaded.second)-TotalLoaded));

  // Range spands loaded and local entities.
  return std::make_pair(iterator(this, int(Loaded.first)-TotalLoaded),
                        iterator(this, Local.second));
}

std::pair<unsigned, unsigned>
PreprocessingRecord::findLocalPreprocessedEntitiesInRange(
                                                      SourceRange Range) const {
  if (Range.isInvalid())
    return std::make_pair(0,0);
  assert(!SourceMgr.isBeforeInTranslationUnit(Range.getEnd(),Range.getBegin()));

  unsigned Begin = findBeginLocalPreprocessedEntity(Range.getBegin());
  unsigned End = findEndLocalPreprocessedEntity(Range.getEnd());
  return std::make_pair(Begin, End);
}

namespace {

template <SourceLocation (SourceRange::*getRangeLoc)() const>
struct PPEntityComp {
  const SourceManager &SM;

  explicit PPEntityComp(const SourceManager &SM) : SM(SM) { }

  bool operator()(PreprocessedEntity *L, SourceLocation RHS) {
    SourceLocation LHS = getLoc(L);
    return SM.isBeforeInTranslationUnit(LHS, RHS);
  }

  bool operator()(SourceLocation LHS, PreprocessedEntity *R) {
    SourceLocation RHS = getLoc(R);
    return SM.isBeforeInTranslationUnit(LHS, RHS);
  }

  SourceLocation getLoc(PreprocessedEntity *PPE) const {
    SourceRange Range = PPE->getSourceRange();
    return (Range.*getRangeLoc)();
  }
};

}

unsigned PreprocessingRecord::findBeginLocalPreprocessedEntity(
                                                     SourceLocation Loc) const {
  if (SourceMgr.isLoadedSourceLocation(Loc))
    return 0;

  std::vector<PreprocessedEntity *>::const_iterator
    I = std::lower_bound(PreprocessedEntities.begin(),
                         PreprocessedEntities.end(),
                         Loc,
                         PPEntityComp<&SourceRange::getEnd>(SourceMgr));
  return I - PreprocessedEntities.begin();
}

unsigned PreprocessingRecord::findEndLocalPreprocessedEntity(
                                                     SourceLocation Loc) const {
  if (SourceMgr.isLoadedSourceLocation(Loc))
    return 0;

  std::vector<PreprocessedEntity *>::const_iterator
  I = std::upper_bound(PreprocessedEntities.begin(),
                       PreprocessedEntities.end(),
                       Loc,
                       PPEntityComp<&SourceRange::getBegin>(SourceMgr));
  return I - PreprocessedEntities.begin();
}

void PreprocessingRecord::addPreprocessedEntity(PreprocessedEntity *Entity) {
  SourceLocation Loc = Entity->getSourceRange().getBegin();
  assert((PreprocessedEntities.empty() ||
          !SourceMgr.isBeforeInTranslationUnit(Loc,
                     PreprocessedEntities.back()->getSourceRange().getEnd())) &&
         "Adding a preprocessed entity that is before the previous one in TU");
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

void PreprocessingRecord::RegisterMacroDefinition(MacroInfo *Macro,
                                                  PPEntityID PPID) {
  MacroDefinitions[Macro] = PPID;
}

/// \brief Retrieve the preprocessed entity at the given ID.
PreprocessedEntity *PreprocessingRecord::getPreprocessedEntity(PPEntityID PPID){
  if (PPID < 0) {
    assert(unsigned(-PPID-1) < LoadedPreprocessedEntities.size() &&
           "Out-of bounds loaded preprocessed entity");
    return getLoadedPreprocessedEntity(LoadedPreprocessedEntities.size()+PPID);
  }
  assert(unsigned(PPID) < PreprocessedEntities.size() &&
         "Out-of bounds local preprocessed entity");
  return PreprocessedEntities[PPID];
}

/// \brief Retrieve the loaded preprocessed entity at the given index.
PreprocessedEntity *
PreprocessingRecord::getLoadedPreprocessedEntity(unsigned Index) {
  assert(Index < LoadedPreprocessedEntities.size() && 
         "Out-of bounds loaded preprocessed entity");
  assert(ExternalSource && "No external source to load from");
  PreprocessedEntity *&Entity = LoadedPreprocessedEntities[Index];
  if (!Entity) {
    Entity = ExternalSource->ReadPreprocessedEntity(Index);
    if (!Entity) // Failed to load.
      Entity = new (*this)
         PreprocessedEntity(PreprocessedEntity::InvalidKind, SourceRange());
  }
  return Entity;
}

MacroDefinition *PreprocessingRecord::findMacroDefinition(const MacroInfo *MI) {
  llvm::DenseMap<const MacroInfo *, PPEntityID>::iterator Pos
    = MacroDefinitions.find(MI);
  if (Pos == MacroDefinitions.end())
    return 0;
  
  PreprocessedEntity *Entity = getPreprocessedEntity(Pos->second);
  if (Entity->isInvalid())
    return 0;
  return cast<MacroDefinition>(Entity);
}

void PreprocessingRecord::MacroExpands(const Token &Id, const MacroInfo* MI,
                                       SourceRange Range) {
  if (!IncludeNestedMacroExpansions && Id.getLocation().isMacroID())
    return;

  if (MI->isBuiltinMacro())
    addPreprocessedEntity(
                      new (*this) MacroExpansion(Id.getIdentifierInfo(),Range));
  else if (MacroDefinition *Def = findMacroDefinition(MI))
    addPreprocessedEntity(
                       new (*this) MacroExpansion(Def, Range));
}

void PreprocessingRecord::MacroDefined(const Token &Id,
                                       const MacroInfo *MI) {
  SourceRange R(MI->getDefinitionLoc(), MI->getDefinitionEndLoc());
  MacroDefinition *Def
      = new (*this) MacroDefinition(Id.getIdentifierInfo(),
                                    MI->getDefinitionLoc(),
                                    R);
  addPreprocessedEntity(Def);
  MacroDefinitions[MI] = getPPEntityID(PreprocessedEntities.size()-1,
                                       /*isLoaded=*/false);
}

void PreprocessingRecord::MacroUndefined(const Token &Id,
                                         const MacroInfo *MI) {
  llvm::DenseMap<const MacroInfo *, PPEntityID>::iterator Pos
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
  addPreprocessedEntity(ID);
}

size_t PreprocessingRecord::getTotalMemory() const {
  return BumpAlloc.getTotalMemory()
    + llvm::capacity_in_bytes(MacroDefinitions)
    + llvm::capacity_in_bytes(PreprocessedEntities)
    + llvm::capacity_in_bytes(LoadedPreprocessedEntities);
}
