//===- MCContext.h - Machine Code Context -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCONTEXT_H
#define LLVM_MC_MCCONTEXT_H

#include "llvm/MC/SectionKind.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include <vector> // FIXME: Shouldn't be needed.

namespace llvm {
  class MCAsmInfo;
  class MCExpr;
  class MCSection;
  class MCSymbol;
  class MCLabel;
  class MCDwarfFile;
  class MCDwarfLoc;
  class MCLineSection;
  class StringRef;
  class Twine;
  class MCSectionMachO;

  /// MCContext - Context object for machine code objects.  This class owns all
  /// of the sections that it creates.
  ///
  class MCContext {
    MCContext(const MCContext&); // DO NOT IMPLEMENT
    MCContext &operator=(const MCContext&); // DO NOT IMPLEMENT

    /// The MCAsmInfo for this target.
    const MCAsmInfo &MAI;

    /// Symbols - Bindings of names to symbols.
    StringMap<MCSymbol*> Symbols;

    /// NextUniqueID - The next ID to dole out to an unnamed assembler temporary
    /// symbol.
    unsigned NextUniqueID;

    /// Instances of directional local labels.
    DenseMap<unsigned, MCLabel *> Instances;
    /// NextInstance() creates the next instance of the directional local label
    /// for the LocalLabelVal and adds it to the map if needed.
    unsigned NextInstance(int64_t LocalLabelVal);
    /// GetInstance() gets the current instance of the directional local label
    /// for the LocalLabelVal and adds it to the map if needed.
    unsigned GetInstance(int64_t LocalLabelVal);
    
    /// The file name of the log file from the enviromment variable
    /// AS_SECURE_LOG_FILE.  Which must be set before the .secure_log_unique
    /// directive is used or it is an error.
    char *SecureLogFile;
    /// The stream that gets written to for the .secure_log_unique directive.
    raw_ostream *SecureLog;
    /// Boolean toggled when .secure_log_unique / .secure_log_reset is seen to
    /// catch errors if .secure_log_unique appears twice without
    /// .secure_log_reset appearing between them.
    bool SecureLogUsed;

    /// The dwarf file and directory tables from the dwarf .file directive.
    std::vector<MCDwarfFile *> MCDwarfFiles;
    std::vector<StringRef> MCDwarfDirs;

    /// The current dwarf line information from the last dwarf .loc directive.
    MCDwarfLoc CurrentDwarfLoc;
    bool DwarfLocSeen;

    /// The dwarf line information from the .loc directives for the sections
    /// with assembled machine instructions have after seeing .loc directives.
    DenseMap<const MCSection *, MCLineSection *> MCLineSections;

    /// Allocator - Allocator object used for creating machine code objects.
    ///
    /// We use a bump pointer allocator to avoid the need to track all allocated
    /// objects.
    BumpPtrAllocator Allocator;
    
    void *MachOUniquingMap, *ELFUniquingMap, *COFFUniquingMap;
  public:
    explicit MCContext(const MCAsmInfo &MAI);
    ~MCContext();
    
    const MCAsmInfo &getAsmInfo() const { return MAI; }

    /// @name Symbol Managment
    /// @{
    
    /// CreateTempSymbol - Create and return a new assembler temporary symbol
    /// with a unique but unspecified name.
    MCSymbol *CreateTempSymbol();

    /// CreateDirectionalLocalSymbol - Create the defintion of a directional
    /// local symbol for numbered label (used for "1:" defintions).
    MCSymbol *CreateDirectionalLocalSymbol(int64_t LocalLabelVal);

    /// GetDirectionalLocalSymbol - Create and return a directional local
    /// symbol for numbered label (used for "1b" or 1f" references).
    MCSymbol *GetDirectionalLocalSymbol(int64_t LocalLabelVal, int bORf);

    /// GetOrCreateSymbol - Lookup the symbol inside with the specified
    /// @p Name.  If it exists, return it.  If not, create a forward
    /// reference and return it.
    ///
    /// @param Name - The symbol name, which must be unique across all symbols.
    MCSymbol *GetOrCreateSymbol(StringRef Name);
    MCSymbol *GetOrCreateSymbol(const Twine &Name);

    /// LookupSymbol - Get the symbol for \p Name, or null.
    MCSymbol *LookupSymbol(StringRef Name) const;

    /// @}
    
    /// @name Section Managment
    /// @{

    /// getMachOSection - Return the MCSection for the specified mach-o section.
    /// This requires the operands to be valid.
    const MCSectionMachO *getMachOSection(StringRef Segment,
                                          StringRef Section,
                                          unsigned TypeAndAttributes,
                                          unsigned Reserved2,
                                          SectionKind K);
    const MCSectionMachO *getMachOSection(StringRef Segment,
                                          StringRef Section,
                                          unsigned TypeAndAttributes,
                                          SectionKind K) {
      return getMachOSection(Segment, Section, TypeAndAttributes, 0, K);
    }
    
    const MCSection *getELFSection(StringRef Section, unsigned Type,
                                   unsigned Flags, SectionKind Kind,
                                   bool IsExplicit = false,
                                   unsigned EntrySize = 0);

    const MCSection *getCOFFSection(StringRef Section, unsigned Characteristics,
                                    int Selection, SectionKind Kind);

    const MCSection *getCOFFSection(StringRef Section, unsigned Characteristics,
                                    SectionKind Kind) {
      return getCOFFSection (Section, Characteristics, 0, Kind);
    }

    
    /// @}

    /// @name Dwarf Managment
    /// @{

    /// GetDwarfFile - creates an entry in the dwarf file and directory tables.
    unsigned GetDwarfFile(StringRef FileName, unsigned FileNumber);

    bool ValidateDwarfFileNumber(unsigned FileNumber);

    const std::vector<MCDwarfFile *> &getMCDwarfFiles() {
      return MCDwarfFiles;
    }
    const std::vector<StringRef> &getMCDwarfDirs() {
      return MCDwarfDirs;
    }
    DenseMap<const MCSection *, MCLineSection *> &getMCLineSections() {
      return MCLineSections;
    }

    /// setCurrentDwarfLoc - saves the information from the currently parsed
    /// dwarf .loc directive and sets DwarfLocSeen.  When the next instruction      /// is assembled an entry in the line number table with this information and
    /// the address of the instruction will be created.
    void setCurrentDwarfLoc(unsigned FileNum, unsigned Line, unsigned Column,
                            unsigned Flags, unsigned Isa) {
      CurrentDwarfLoc.setFileNum(FileNum);
      CurrentDwarfLoc.setLine(Line);
      CurrentDwarfLoc.setColumn(Column);
      CurrentDwarfLoc.setFlags(Flags);
      CurrentDwarfLoc.setIsa(Isa);
      DwarfLocSeen = true;
    }
    void clearDwarfLocSeen() { DwarfLocSeen = false; }

    bool getDwarfLocSeen() { return DwarfLocSeen; }
    const MCDwarfLoc &getCurrentDwarfLoc() { return CurrentDwarfLoc; }

    /// @}

    char *getSecureLogFile() { return SecureLogFile; }
    raw_ostream *getSecureLog() { return SecureLog; }
    bool getSecureLogUsed() { return SecureLogUsed; }
    void setSecureLog(raw_ostream *Value) {
      SecureLog = Value;
    }
    void setSecureLogUsed(bool Value) {
      SecureLogUsed = Value;
    }

    void *Allocate(unsigned Size, unsigned Align = 8) {
      return Allocator.Allocate(Size, Align);
    }
    void Deallocate(void *Ptr) {
    }
  };

} // end namespace llvm

// operator new and delete aren't allowed inside namespaces.
// The throw specifications are mandated by the standard.
/// @brief Placement new for using the MCContext's allocator.
///
/// This placement form of operator new uses the MCContext's allocator for
/// obtaining memory. It is a non-throwing new, which means that it returns
/// null on error. (If that is what the allocator does. The current does, so if
/// this ever changes, this operator will have to be changed, too.)
/// Usage looks like this (assuming there's an MCContext 'Context' in scope):
/// @code
/// // Default alignment (16)
/// IntegerLiteral *Ex = new (Context) IntegerLiteral(arguments);
/// // Specific alignment
/// IntegerLiteral *Ex2 = new (Context, 8) IntegerLiteral(arguments);
/// @endcode
/// Please note that you cannot use delete on the pointer; it must be
/// deallocated using an explicit destructor call followed by
/// @c Context.Deallocate(Ptr).
///
/// @param Bytes The number of bytes to allocate. Calculated by the compiler.
/// @param C The MCContext that provides the allocator.
/// @param Alignment The alignment of the allocated memory (if the underlying
///                  allocator supports it).
/// @return The allocated memory. Could be NULL.
inline void *operator new(size_t Bytes, llvm::MCContext &C,
                          size_t Alignment = 16) throw () {
  return C.Allocate(Bytes, Alignment);
}
/// @brief Placement delete companion to the new above.
///
/// This operator is just a companion to the new above. There is no way of
/// invoking it directly; see the new operator for more details. This operator
/// is called implicitly by the compiler if a placement new expression using
/// the MCContext throws in the object constructor.
inline void operator delete(void *Ptr, llvm::MCContext &C, size_t)
              throw () {
  C.Deallocate(Ptr);
}

/// This placement form of operator new[] uses the MCContext's allocator for
/// obtaining memory. It is a non-throwing new[], which means that it returns
/// null on error.
/// Usage looks like this (assuming there's an MCContext 'Context' in scope):
/// @code
/// // Default alignment (16)
/// char *data = new (Context) char[10];
/// // Specific alignment
/// char *data = new (Context, 8) char[10];
/// @endcode
/// Please note that you cannot use delete on the pointer; it must be
/// deallocated using an explicit destructor call followed by
/// @c Context.Deallocate(Ptr).
///
/// @param Bytes The number of bytes to allocate. Calculated by the compiler.
/// @param C The MCContext that provides the allocator.
/// @param Alignment The alignment of the allocated memory (if the underlying
///                  allocator supports it).
/// @return The allocated memory. Could be NULL.
inline void *operator new[](size_t Bytes, llvm::MCContext& C,
                            size_t Alignment = 16) throw () {
  return C.Allocate(Bytes, Alignment);
}

/// @brief Placement delete[] companion to the new[] above.
///
/// This operator is just a companion to the new[] above. There is no way of
/// invoking it directly; see the new[] operator for more details. This operator
/// is called implicitly by the compiler if a placement new[] expression using
/// the MCContext throws in the object constructor.
inline void operator delete[](void *Ptr, llvm::MCContext &C) throw () {
  C.Deallocate(Ptr);
}

#endif
