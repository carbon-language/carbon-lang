//===--- PreprocessingRecord.h - Record of Preprocessing --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PreprocessingRecord class, which maintains a record
//  of what occurred during preprocessing.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LEX_PREPROCESSINGRECORD_H
#define LLVM_CLANG_LEX_PREPROCESSINGRECORD_H

#include "clang/Lex/PPCallbacks.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace clang {
  class IdentifierInfo;
  class PreprocessingRecord;
}

/// \brief Allocates memory within a Clang preprocessing record.
void* operator new(size_t bytes, clang::PreprocessingRecord& PR,
                   unsigned alignment = 8) throw();

/// \brief Frees memory allocated in a Clang preprocessing record.
void operator delete(void* ptr, clang::PreprocessingRecord& PR,
                     unsigned) throw();

namespace clang {
  class MacroDefinition;
  class FileEntry;

  /// \brief Base class that describes a preprocessed entity, which may be a
  /// preprocessor directive or macro expansion.
  class PreprocessedEntity {
  public:
    /// \brief The kind of preprocessed entity an object describes.
    enum EntityKind {
      /// \brief A macro expansion.
      MacroExpansionKind,
      
      /// \brief A preprocessing directive whose kind is not specified.
      ///
      /// This kind will be used for any preprocessing directive that does not
      /// have a more specific kind within the \c DirectiveKind enumeration.
      PreprocessingDirectiveKind,
      
      /// \brief A macro definition.
      MacroDefinitionKind,
      
      /// \brief An inclusion directive, such as \c #include, \c
      /// #import, or \c #include_next.
      InclusionDirectiveKind,

      FirstPreprocessingDirective = PreprocessingDirectiveKind,
      LastPreprocessingDirective = InclusionDirectiveKind
    };

  private:
    /// \brief The kind of preprocessed entity that this object describes.
    EntityKind Kind;
    
    /// \brief The source range that covers this preprocessed entity.
    SourceRange Range;
    
  protected:
    PreprocessedEntity(EntityKind Kind, SourceRange Range)
      : Kind(Kind), Range(Range) { }
    
  public:
    /// \brief Retrieve the kind of preprocessed entity stored in this object.
    EntityKind getKind() const { return Kind; }
    
    /// \brief Retrieve the source range that covers this entire preprocessed 
    /// entity.
    SourceRange getSourceRange() const { return Range; }
    
    // Implement isa/cast/dyncast/etc.
    static bool classof(const PreprocessedEntity *) { return true; }

    // Only allow allocation of preprocessed entities using the allocator 
    // in PreprocessingRecord or by doing a placement new.
    void* operator new(size_t bytes, PreprocessingRecord& PR,
                       unsigned alignment = 8) throw() {
      return ::operator new(bytes, PR, alignment);
    }
    
    void* operator new(size_t bytes, void* mem) throw() {
      return mem;
    }
    
    void operator delete(void* ptr, PreprocessingRecord& PR, 
                         unsigned alignment) throw() {
      return ::operator delete(ptr, PR, alignment);
    }
    
    void operator delete(void*, std::size_t) throw() { }
    void operator delete(void*, void*) throw() { }
    
  private:
    // Make vanilla 'new' and 'delete' illegal for preprocessed entities.
    void* operator new(size_t bytes) throw();
    void operator delete(void* data) throw();
  };
  
  /// \brief Records the location of a macro expansion.
  class MacroExpansion : public PreprocessedEntity {
    /// \brief The name of the macro being expanded.
    IdentifierInfo *Name;
    
    /// \brief The definition of this macro.
    MacroDefinition *Definition;
    
  public:
    MacroExpansion(IdentifierInfo *Name, SourceRange Range,
                   MacroDefinition *Definition)
      : PreprocessedEntity(MacroExpansionKind, Range), Name(Name),
        Definition(Definition) { }
    
    /// \brief The name of the macro being expanded.
    IdentifierInfo *getName() const { return Name; }
    
    /// \brief The definition of the macro being expanded.
    MacroDefinition *getDefinition() const { return Definition; }

    // Implement isa/cast/dyncast/etc.
    static bool classof(const PreprocessedEntity *PE) {
      return PE->getKind() == MacroExpansionKind;
    }
    static bool classof(const MacroExpansion *) { return true; }

  };
  
  /// \brief Records the presence of a preprocessor directive.
  class PreprocessingDirective : public PreprocessedEntity {
  public:
    PreprocessingDirective(EntityKind Kind, SourceRange Range) 
      : PreprocessedEntity(Kind, Range) { }
    
    // Implement isa/cast/dyncast/etc.
    static bool classof(const PreprocessedEntity *PD) { 
      return PD->getKind() >= FirstPreprocessingDirective &&
             PD->getKind() <= LastPreprocessingDirective;
    }
    static bool classof(const PreprocessingDirective *) { return true; }    
  };
  
  /// \brief Record the location of a macro definition.
  class MacroDefinition : public PreprocessingDirective {
    /// \brief The name of the macro being defined.
    const IdentifierInfo *Name;
    
    /// \brief The location of the macro name in the macro definition.
    SourceLocation Location;

  public:
    explicit MacroDefinition(const IdentifierInfo *Name, SourceLocation Location,
                             SourceRange Range)
      : PreprocessingDirective(MacroDefinitionKind, Range), Name(Name), 
        Location(Location) { }
    
    /// \brief Retrieve the name of the macro being defined.
    const IdentifierInfo *getName() const { return Name; }
    
    /// \brief Retrieve the location of the macro name in the definition.
    SourceLocation getLocation() const { return Location; }
    
    // Implement isa/cast/dyncast/etc.
    static bool classof(const PreprocessedEntity *PE) {
      return PE->getKind() == MacroDefinitionKind;
    }
    static bool classof(const MacroDefinition *) { return true; }
  };

  /// \brief Record the location of an inclusion directive, such as an
  /// \c #include or \c #import statement.
  class InclusionDirective : public PreprocessingDirective {
  public:
    /// \brief The kind of inclusion directives known to the
    /// preprocessor.
    enum InclusionKind {
      /// \brief An \c #include directive.
      Include,
      /// \brief An Objective-C \c #import directive.
      Import,
      /// \brief A GNU \c #include_next directive.
      IncludeNext,
      /// \brief A Clang \c #__include_macros directive.
      IncludeMacros
    };

  private:
    /// \brief The name of the file that was included, as written in
    /// the source.
    llvm::StringRef FileName;

    /// \brief Whether the file name was in quotation marks; otherwise, it was
    /// in angle brackets.
    unsigned InQuotes : 1;

    /// \brief The kind of inclusion directive we have.
    ///
    /// This is a value of type InclusionKind.
    unsigned Kind : 2;

    /// \brief The file that was included.
    const FileEntry *File;

  public:
    InclusionDirective(PreprocessingRecord &PPRec,
                       InclusionKind Kind, llvm::StringRef FileName, 
                       bool InQuotes, const FileEntry *File, SourceRange Range);
    
    /// \brief Determine what kind of inclusion directive this is.
    InclusionKind getKind() const { return static_cast<InclusionKind>(Kind); }
    
    /// \brief Retrieve the included file name as it was written in the source.
    llvm::StringRef getFileName() const { return FileName; }
    
    /// \brief Determine whether the included file name was written in quotes;
    /// otherwise, it was written in angle brackets.
    bool wasInQuotes() const { return InQuotes; }
    
    /// \brief Retrieve the file entry for the actual file that was included
    /// by this directive.
    const FileEntry *getFile() const { return File; }
        
    // Implement isa/cast/dyncast/etc.
    static bool classof(const PreprocessedEntity *PE) {
      return PE->getKind() == InclusionDirectiveKind;
    }
    static bool classof(const InclusionDirective *) { return true; }
  };
  
  /// \brief An abstract class that should be subclassed by any external source
  /// of preprocessing record entries.
  class ExternalPreprocessingRecordSource {
  public:
    virtual ~ExternalPreprocessingRecordSource();
    
    /// \brief Read any preallocated preprocessed entities from the external
    /// source.
    virtual void ReadPreprocessedEntities() = 0;
    
    /// \brief Read the preprocessed entity at the given offset.
    virtual PreprocessedEntity *
    ReadPreprocessedEntityAtOffset(uint64_t Offset) = 0;
  };
  
  /// \brief A record of the steps taken while preprocessing a source file,
  /// including the various preprocessing directives processed, macros 
  /// expanded, etc.
  class PreprocessingRecord : public PPCallbacks {
    /// \brief Whether we should include nested macro expansions in
    /// the preprocessing record.
    bool IncludeNestedMacroExpansions;
    
    /// \brief Allocator used to store preprocessing objects.
    llvm::BumpPtrAllocator BumpAlloc;

    /// \brief The set of preprocessed entities in this record, in order they
    /// were seen.
    std::vector<PreprocessedEntity *> PreprocessedEntities;
    
    /// \brief Mapping from MacroInfo structures to their definitions.
    llvm::DenseMap<const MacroInfo *, MacroDefinition *> MacroDefinitions;

    /// \brief External source of preprocessed entities.
    ExternalPreprocessingRecordSource *ExternalSource;
    
    /// \brief The number of preallocated entities (that are known to the
    /// external source).
    unsigned NumPreallocatedEntities;
    
    /// \brief Whether we have already loaded all of the preallocated entities.
    mutable bool LoadedPreallocatedEntities;
    
    void MaybeLoadPreallocatedEntities() const ;
    
  public:
    /// \brief Construct 
    explicit PreprocessingRecord(bool IncludeNestedMacroExpansions);
    
    /// \brief Allocate memory in the preprocessing record.
    void *Allocate(unsigned Size, unsigned Align = 8) {
      return BumpAlloc.Allocate(Size, Align);
    }
    
    /// \brief Deallocate memory in the preprocessing record.
    void Deallocate(void *Ptr) { }
    
    size_t getTotalMemory() const {
      return BumpAlloc.getTotalMemory();
    }
    
    // Iteration over the preprocessed entities.
    typedef std::vector<PreprocessedEntity *>::iterator iterator;
    typedef std::vector<PreprocessedEntity *>::const_iterator const_iterator;
    iterator begin(bool OnlyLocalEntities = false);
    iterator end(bool OnlyLocalEntities = false);
    const_iterator begin(bool OnlyLocalEntities = false) const;
    const_iterator end(bool OnlyLocalEntities = false) const;

    /// \brief Add a new preprocessed entity to this record.
    void addPreprocessedEntity(PreprocessedEntity *Entity);
    
    /// \brief Set the external source for preprocessed entities.
    void SetExternalSource(ExternalPreprocessingRecordSource &Source,
                           unsigned NumPreallocatedEntities);

    /// \brief Retrieve the external source for preprocessed entities.
    ExternalPreprocessingRecordSource *getExternalSource() const {
      return ExternalSource;
    }
    
    unsigned getNumPreallocatedEntities() const {
      return NumPreallocatedEntities;
    }
    
    /// \brief Set the preallocated entry at the given index to the given
    /// preprocessed entity.
    void SetPreallocatedEntity(unsigned Index, PreprocessedEntity *Entity);

    /// \brief Register a new macro definition.
    void RegisterMacroDefinition(MacroInfo *Macro, MacroDefinition *MD);
                           
    /// \brief Retrieve the preprocessed entity at the given index.
    PreprocessedEntity *getPreprocessedEntity(unsigned Index) {
      assert(Index < PreprocessedEntities.size() &&
             "Out-of-bounds preprocessed entity");
      return PreprocessedEntities[Index];
    }
    
    /// \brief Retrieve the macro definition that corresponds to the given
    /// \c MacroInfo.
    MacroDefinition *findMacroDefinition(const MacroInfo *MI);
    
    virtual void MacroExpands(const Token &Id, const MacroInfo* MI);
    virtual void MacroDefined(const Token &Id, const MacroInfo *MI);
    virtual void MacroUndefined(const Token &Id, const MacroInfo *MI);
    virtual void InclusionDirective(SourceLocation HashLoc,
                                    const Token &IncludeTok,
                                    llvm::StringRef FileName,
                                    bool IsAngled,
                                    const FileEntry *File,
                                    SourceLocation EndLoc,
                                    llvm::StringRef SearchPath,
                                    llvm::StringRef RelativePath);
  };
} // end namespace clang

inline void* operator new(size_t bytes, clang::PreprocessingRecord& PR,
                          unsigned alignment) throw() {
  return PR.Allocate(bytes, alignment);
}

inline void operator delete(void* ptr, clang::PreprocessingRecord& PR,
                            unsigned) throw() {
  PR.Deallocate(ptr);
}

#endif // LLVM_CLANG_LEX_PREPROCESSINGRECORD_H
