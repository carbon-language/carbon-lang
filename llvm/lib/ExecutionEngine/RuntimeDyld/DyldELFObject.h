//===-- DyldELFObject.h - Dynamically loaded ELF object  ----0---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Dynamically loaded ELF object class, a subclass of ELFObjectFile. Used
// to represent a loadable ELF image.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIMEDYLD_DYLDELFOBJECT_H
#define LLVM_RUNTIMEDYLD_DYLDELFOBJECT_H

#include "llvm/Object/ELF.h"


namespace llvm {

using support::endianness;
using namespace llvm::object;

template<support::endianness target_endianness, bool is64Bits>
class DyldELFObject : public ELFObjectFile<target_endianness, is64Bits> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)

  typedef Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  typedef Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  typedef Elf_Rel_Impl<target_endianness, is64Bits, false> Elf_Rel;
  typedef Elf_Rel_Impl<target_endianness, is64Bits, true> Elf_Rela;

  typedef typename ELFObjectFile<target_endianness, is64Bits>::
    Elf_Ehdr Elf_Ehdr;
  Elf_Ehdr *Header;

  // Update section headers according to the current location in memory
  virtual void rebaseObject(std::vector<uint8_t*> *MemoryMap);
  // Record memory addresses for cleanup
  virtual void saveAddress(std::vector<uint8_t*> *MemoryMap, uint8_t *addr);

protected:
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;

public:
  DyldELFObject(MemoryBuffer *Object, std::vector<uint8_t*> *MemoryMap,
                error_code &ec);

  // Methods for type inquiry through isa, cast, and dyn_cast
  static inline bool classof(const Binary *v) {
    return (isa<ELFObjectFile<target_endianness, is64Bits> >(v)
            && classof(cast<ELFObjectFile<target_endianness, is64Bits> >(v)));
  }
  static inline bool classof(
      const ELFObjectFile<target_endianness, is64Bits> *v) {
    return v->isDyldType();
  }
  static inline bool classof(const DyldELFObject *v) {
    return true;
  }
};

template<support::endianness target_endianness, bool is64Bits>
DyldELFObject<target_endianness, is64Bits>::DyldELFObject(MemoryBuffer *Object,
      std::vector<uint8_t*> *MemoryMap, error_code &ec)
  : ELFObjectFile<target_endianness, is64Bits>(Object, ec)
  , Header(0) {
  this->isDyldELFObject = true;
  Header = const_cast<Elf_Ehdr *>(
      reinterpret_cast<const Elf_Ehdr *>(this->base()));
  if (Header->e_shoff == 0)
    return;

  // Mark the image as a dynamic shared library
  Header->e_type = ELF::ET_DYN;

  rebaseObject(MemoryMap);
}

// Walk through the ELF headers, updating virtual addresses to reflect where
// the object is currently loaded in memory
template<support::endianness target_endianness, bool is64Bits>
void DyldELFObject<target_endianness, is64Bits>::rebaseObject(
    std::vector<uint8_t*> *MemoryMap) {
  typedef typename ELFDataTypeTypedefHelper<
          target_endianness, is64Bits>::value_type addr_type;

  uint8_t *base_p = const_cast<uint8_t *>(this->base());
  Elf_Shdr *sectionTable =
      reinterpret_cast<Elf_Shdr *>(base_p + Header->e_shoff);
  uint64_t numSections = this->getNumSections();

  // Allocate memory space for NOBITS sections (such as .bss), which only exist
  // in memory, but don't occupy space in the object file.
  // Update the address in the section headers to reflect this allocation.
  for (uint64_t index = 0; index < numSections; index++) {
    Elf_Shdr *sec = reinterpret_cast<Elf_Shdr *>(
        reinterpret_cast<char *>(sectionTable) + index * Header->e_shentsize);

    // Only update sections that are meant to be present in program memory
    if (sec->sh_flags & ELF::SHF_ALLOC) {
      uint8_t *addr = base_p + sec->sh_offset;
      if (sec->sh_type == ELF::SHT_NOBITS) {
        addr = static_cast<uint8_t *>(calloc(sec->sh_size, 1));
        saveAddress(MemoryMap, addr);
      }
      else {
        // FIXME: Currently memory with RWX permissions is allocated. In the
        // future, make sure that permissions are as necessary
        if (sec->sh_flags & ELF::SHF_WRITE) {
            // see FIXME above
        }
        if (sec->sh_flags & ELF::SHF_EXECINSTR) {
            // see FIXME above
        }
      }
      assert(sizeof(addr_type) == sizeof(intptr_t) &&
             "Cross-architecture ELF dy-load is not supported!");
      sec->sh_addr = static_cast<addr_type>(intptr_t(addr));
    }
  }

  // Now allocate actual space for COMMON symbols, which also don't occupy
  // space in the object file.
  // We want to allocate space for all COMMON symbols at once, so the flow is:
  // 1. Go over all symbols, find those that are in COMMON. For each such
  //    symbol, record its size and the value field in its symbol header in a
  //    special vector.
  // 2. Allocate memory for all COMMON symbols in one fell swoop.
  // 3. Using the recorded information from (1), update the address fields in
  //    the symbol headers of the COMMON symbols to reflect their allocated
  //    address.
  uint64_t TotalSize = 0;
  std::vector<std::pair<Elf_Addr *, uint64_t> > SymbAddrInfo;
  error_code ec = object_error::success;
  for (symbol_iterator si = this->begin_symbols(),
       se = this->end_symbols(); si != se; si.increment(ec)) {
    uint64_t Size = 0;
    ec = si->getSize(Size);
    Elf_Sym* symb = const_cast<Elf_Sym*>(
        this->getSymbol(si->getRawDataRefImpl()));
    if (ec == object_error::success &&
        this->getSymbolTableIndex(symb) == ELF::SHN_COMMON && Size > 0) {
      SymbAddrInfo.push_back(std::make_pair(&(symb->st_value), Size));
      TotalSize += Size;
    }
  }

  uint8_t* SectionPtr = (uint8_t *)calloc(TotalSize, 1);
  saveAddress(MemoryMap, SectionPtr);

  typedef typename std::vector<std::pair<Elf_Addr *, uint64_t> >::iterator
      AddrInfoIterator;
  AddrInfoIterator EndIter = SymbAddrInfo.end();
  for (AddrInfoIterator AddrIter = SymbAddrInfo.begin();
       AddrIter != EndIter; ++AddrIter) {
    assert(sizeof(addr_type) == sizeof(intptr_t) &&
           "Cross-architecture ELF dy-load is not supported!");
    *(AddrIter->first) = static_cast<addr_type>(intptr_t(SectionPtr));
    SectionPtr += AddrIter->second;
  }
}

// Record memory addresses for callers
template<support::endianness target_endianness, bool is64Bits>
void DyldELFObject<target_endianness, is64Bits>::saveAddress(
    std::vector<uint8_t*> *MemoryMap, uint8_t* addr) {
  if (MemoryMap)
    MemoryMap->push_back(addr);
  else
    errs() << "WARNING: Memory leak - cannot record memory for ELF dyld.";
}

template<support::endianness target_endianness, bool is64Bits>
error_code DyldELFObject<target_endianness, is64Bits>::getSymbolAddress(
    DataRefImpl Symb, uint64_t &Result) const {
  this->validateSymbol(Symb);
  const Elf_Sym *symb = this->getSymbol(Symb);
  if (this->getSymbolTableIndex(symb) == ELF::SHN_COMMON) {
    Result = symb->st_value;
    return object_error::success;
  }
  else {
    return ELFObjectFile<target_endianness, is64Bits>::getSymbolAddress(
        Symb, Result);
  }
}

}

#endif

//===-- DyldELFObject.h - Dynamically loaded ELF object  ----0---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Dynamically loaded ELF object class, a subclass of ELFObjectFile. Used
// to represent a loadable ELF image.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIMEDYLD_DYLDELFOBJECT_H
#define LLVM_RUNTIMEDYLD_DYLDELFOBJECT_H

#include "llvm/Object/ELF.h"


namespace llvm {

using support::endianness;
using namespace llvm::object;

template<support::endianness target_endianness, bool is64Bits>
class DyldELFObject : public ELFObjectFile<target_endianness, is64Bits> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)

  typedef Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  typedef Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  typedef Elf_Rel_Impl<target_endianness, is64Bits, false> Elf_Rel;
  typedef Elf_Rel_Impl<target_endianness, is64Bits, true> Elf_Rela;

  typedef typename ELFObjectFile<target_endianness, is64Bits>::
    Elf_Ehdr Elf_Ehdr;
  Elf_Ehdr *Header;

  // Update section headers according to the current location in memory
  virtual void rebaseObject(std::vector<uint8_t*> *MemoryMap);
  // Record memory addresses for cleanup
  virtual void saveAddress(std::vector<uint8_t*> *MemoryMap, uint8_t *addr);

protected:
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;

public:
  DyldELFObject(MemoryBuffer *Object, std::vector<uint8_t*> *MemoryMap,
                error_code &ec);

  // Methods for type inquiry through isa, cast, and dyn_cast
  static inline bool classof(const Binary *v) {
    return (isa<ELFObjectFile<target_endianness, is64Bits> >(v)
            && classof(cast<ELFObjectFile<target_endianness, is64Bits> >(v)));
  }
  static inline bool classof(
      const ELFObjectFile<target_endianness, is64Bits> *v) {
    return v->isDyldType();
  }
  static inline bool classof(const DyldELFObject *v) {
    return true;
  }
};

template<support::endianness target_endianness, bool is64Bits>
DyldELFObject<target_endianness, is64Bits>::DyldELFObject(MemoryBuffer *Object,
      std::vector<uint8_t*> *MemoryMap, error_code &ec)
  : ELFObjectFile<target_endianness, is64Bits>(Object, ec)
  , Header(0) {
  this->isDyldELFObject = true;
  Header = const_cast<Elf_Ehdr *>(
      reinterpret_cast<const Elf_Ehdr *>(this->base()));
  if (Header->e_shoff == 0)
    return;

  // Mark the image as a dynamic shared library
  Header->e_type = ELF::ET_DYN;

  rebaseObject(MemoryMap);
}

// Walk through the ELF headers, updating virtual addresses to reflect where
// the object is currently loaded in memory
template<support::endianness target_endianness, bool is64Bits>
void DyldELFObject<target_endianness, is64Bits>::rebaseObject(
    std::vector<uint8_t*> *MemoryMap) {
  typedef typename ELFDataTypeTypedefHelper<
          target_endianness, is64Bits>::value_type addr_type;

  uint8_t *base_p = const_cast<uint8_t *>(this->base());
  Elf_Shdr *sectionTable =
      reinterpret_cast<Elf_Shdr *>(base_p + Header->e_shoff);
  uint64_t numSections = this->getNumSections();

  // Allocate memory space for NOBITS sections (such as .bss), which only exist
  // in memory, but don't occupy space in the object file.
  // Update the address in the section headers to reflect this allocation.
  for (uint64_t index = 0; index < numSections; index++) {
    Elf_Shdr *sec = reinterpret_cast<Elf_Shdr *>(
        reinterpret_cast<char *>(sectionTable) + index * Header->e_shentsize);

    // Only update sections that are meant to be present in program memory
    if (sec->sh_flags & ELF::SHF_ALLOC) {
      uint8_t *addr = base_p + sec->sh_offset;
      if (sec->sh_type == ELF::SHT_NOBITS) {
        addr = static_cast<uint8_t *>(calloc(sec->sh_size, 1));
        saveAddress(MemoryMap, addr);
      }
      else {
        // FIXME: Currently memory with RWX permissions is allocated. In the
        // future, make sure that permissions are as necessary
        if (sec->sh_flags & ELF::SHF_WRITE) {
            // see FIXME above
        }
        if (sec->sh_flags & ELF::SHF_EXECINSTR) {
            // see FIXME above
        }
      }
      assert(sizeof(addr_type) == sizeof(intptr_t) &&
             "Cross-architecture ELF dy-load is not supported!");
      sec->sh_addr = static_cast<addr_type>(intptr_t(addr));
    }
  }

  // Now allocate actual space for COMMON symbols, which also don't occupy
  // space in the object file.
  // We want to allocate space for all COMMON symbols at once, so the flow is:
  // 1. Go over all symbols, find those that are in COMMON. For each such
  //    symbol, record its size and the value field in its symbol header in a
  //    special vector.
  // 2. Allocate memory for all COMMON symbols in one fell swoop.
  // 3. Using the recorded information from (1), update the address fields in
  //    the symbol headers of the COMMON symbols to reflect their allocated
  //    address.
  uint64_t TotalSize = 0;
  std::vector<std::pair<Elf_Addr *, uint64_t> > SymbAddrInfo;
  error_code ec = object_error::success;
  for (symbol_iterator si = this->begin_symbols(),
       se = this->end_symbols(); si != se; si.increment(ec)) {
    uint64_t Size = 0;
    ec = si->getSize(Size);
    Elf_Sym* symb = const_cast<Elf_Sym*>(
        this->getSymbol(si->getRawDataRefImpl()));
    if (ec == object_error::success &&
        this->getSymbolTableIndex(symb) == ELF::SHN_COMMON && Size > 0) {
      SymbAddrInfo.push_back(std::make_pair(&(symb->st_value), Size));
      TotalSize += Size;
    }
  }

  uint8_t* SectionPtr = (uint8_t *)calloc(TotalSize, 1);
  saveAddress(MemoryMap, SectionPtr);

  typedef typename std::vector<std::pair<Elf_Addr *, uint64_t> >::iterator
      AddrInfoIterator;
  AddrInfoIterator EndIter = SymbAddrInfo.end();
  for (AddrInfoIterator AddrIter = SymbAddrInfo.begin();
       AddrIter != EndIter; ++AddrIter) {
    assert(sizeof(addr_type) == sizeof(intptr_t) &&
           "Cross-architecture ELF dy-load is not supported!");
    *(AddrIter->first) = static_cast<addr_type>(intptr_t(SectionPtr));
    SectionPtr += AddrIter->second;
  }
}

// Record memory addresses for callers
template<support::endianness target_endianness, bool is64Bits>
void DyldELFObject<target_endianness, is64Bits>::saveAddress(
    std::vector<uint8_t*> *MemoryMap, uint8_t* addr) {
  if (MemoryMap)
    MemoryMap->push_back(addr);
  else
    errs() << "WARNING: Memory leak - cannot record memory for ELF dyld.";
}

template<support::endianness target_endianness, bool is64Bits>
error_code DyldELFObject<target_endianness, is64Bits>::getSymbolAddress(
    DataRefImpl Symb, uint64_t &Result) const {
  this->validateSymbol(Symb);
  const Elf_Sym *symb = this->getSymbol(Symb);
  if (this->getSymbolTableIndex(symb) == ELF::SHN_COMMON) {
    Result = symb->st_value;
    return object_error::success;
  }
  else {
    return ELFObjectFile<target_endianness, is64Bits>::getSymbolAddress(
        Symb, Result);
  }
}

}

#endif

