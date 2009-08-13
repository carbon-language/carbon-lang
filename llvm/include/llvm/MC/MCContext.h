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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"

namespace llvm {
  class MCValue;
  class MCSection;
  class MCSymbol;
  class StringRef;

  /// MCContext - Context object for machine code objects.  This class owns all
  /// of the sections that it creates.
  ///
  class MCContext {
    MCContext(const MCContext&); // DO NOT IMPLEMENT
    MCContext &operator=(const MCContext&); // DO NOT IMPLEMENT

    /// Sections - Bindings of names to allocated sections.
    StringMap<MCSection*> Sections;

    /// Symbols - Bindings of names to symbols.
    StringMap<MCSymbol*> Symbols;

    /// SymbolValues - Bindings of symbols to values.
    //
    // FIXME: Is there a good reason to not just put this in the MCSymbol?
    DenseMap<MCSymbol*, MCValue> SymbolValues;

    /// Allocator - Allocator object used for creating machine code objects.
    ///
    /// We use a bump pointer allocator to avoid the need to track all allocated
    /// objects.
    BumpPtrAllocator Allocator;
  public:
    MCContext();
    ~MCContext();

    /// GetSection - Look up a section with the given @param Name, returning
    /// null if it doesn't exist.
    MCSection *GetSection(const StringRef &Name) const;
    
    void SetSection(const StringRef &Name, MCSection *S) {
      MCSection *&Entry = Sections[Name];
      assert(Entry == 0 && "Multiple sections with the same name created");
      Entry = S;
    }
    
    /// CreateSymbol - Create a new symbol with the specified @param Name.
    ///
    /// @param Name - The symbol name, which must be unique across all symbols.
    MCSymbol *CreateSymbol(const StringRef &Name);

    /// GetOrCreateSymbol - Lookup the symbol inside with the specified
    /// @param Name.  If it exists, return it.  If not, create a forward
    /// reference and return it.
    ///
    /// @param Name - The symbol name, which must be unique across all symbols.
    MCSymbol *GetOrCreateSymbol(const StringRef &Name);
    
    /// CreateTemporarySymbol - Create a new temporary symbol with the specified
    /// @param Name.
    ///
    /// @param Name - The symbol name, for debugging purposes only, temporary
    /// symbols do not surive assembly. If non-empty the name must be unique
    /// across all symbols.
    MCSymbol *CreateTemporarySymbol(const StringRef &Name = "");

    /// LookupSymbol - Get the symbol for @param Name, or null.
    MCSymbol *LookupSymbol(const StringRef &Name) const;

    /// ClearSymbolValue - Erase a value binding for @param Symbol, if one
    /// exists.
    void ClearSymbolValue(MCSymbol *Symbol);

    /// SetSymbolValue - Set the value binding for @param Symbol to @param
    /// Value.
    void SetSymbolValue(MCSymbol *Symbol, const MCValue &Value);

    /// GetSymbolValue - Return the current value for @param Symbol, or null if
    /// none exists.
    const MCValue *GetSymbolValue(MCSymbol *Symbol) const;

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
