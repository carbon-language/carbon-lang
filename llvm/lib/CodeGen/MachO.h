//=== MachO.h - Mach-O structures and constants -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines MachO .
//
//===----------------------------------------------------------------------===//

#ifndef MACHO_H
#define MACHO_H

#include "llvm/CodeGen/BinaryObject.h"
#include <string>
#include <vector>

namespace llvm {

class GlobalValue;
class MCAsmInfo;

/// MachOSym - This struct contains information about each symbol that is
/// added to logical symbol table for the module.  This is eventually
/// turned into a real symbol table in the file.
struct MachOSym {
  const GlobalValue *GV;    // The global value this corresponds to.
  std::string GVName;       // The mangled name of the global value.
  uint32_t    n_strx;       // index into the string table
  uint8_t     n_type;       // type flag
  uint8_t     n_sect;       // section number or NO_SECT
  int16_t     n_desc;       // see <mach-o/stab.h>
  uint64_t    n_value;      // value for this symbol (or stab offset)
  
  // Constants for the n_sect field
  // see <mach-o/nlist.h>
  enum { NO_SECT = 0 };   // symbol is not in any section

  // Constants for the n_type field
  // see <mach-o/nlist.h>
  enum { N_UNDF  = 0x0,  // undefined, n_sect == NO_SECT
         N_ABS   = 0x2,  // absolute, n_sect == NO_SECT
         N_SECT  = 0xe,  // defined in section number n_sect
         N_PBUD  = 0xc,  // prebound undefined (defined in a dylib)
         N_INDR  = 0xa   // indirect
  };
  // The following bits are OR'd into the types above. For example, a type
  // of 0x0f would be an external N_SECT symbol (0x0e | 0x01).
  enum { N_EXT  = 0x01,   // external symbol bit
         N_PEXT = 0x10    // private external symbol bit
  };
  
  // Constants for the n_desc field
  // see <mach-o/loader.h>
  enum { REFERENCE_FLAG_UNDEFINED_NON_LAZY          = 0,
         REFERENCE_FLAG_UNDEFINED_LAZY              = 1,
         REFERENCE_FLAG_DEFINED                     = 2,
         REFERENCE_FLAG_PRIVATE_DEFINED             = 3,
         REFERENCE_FLAG_PRIVATE_UNDEFINED_NON_LAZY  = 4,
         REFERENCE_FLAG_PRIVATE_UNDEFINED_LAZY      = 5
  };
  enum { N_NO_DEAD_STRIP = 0x0020, // symbol is not to be dead stripped
         N_WEAK_REF      = 0x0040, // symbol is weak referenced
         N_WEAK_DEF      = 0x0080  // coalesced symbol is a weak definition
  };
  
  MachOSym(const GlobalValue *gv, std::string name, uint8_t sect,
           const MCAsmInfo *MAI);

  struct SymCmp {
    // FIXME: this does not appear to be sorting 'f' after 'F'
    bool operator()(const MachOSym &LHS, const MachOSym &RHS) {
      return LHS.GVName < RHS.GVName;
    }
  };


  /// PartitionByLocal - Simple boolean predicate that returns true if Sym is
  /// a local symbol rather than an external symbol.
  
  static inline bool PartitionByLocal(const MachOSym &Sym) {
    return (Sym.n_type & (MachOSym::N_EXT | MachOSym::N_PEXT)) == 0;
  }
  
  /// PartitionByDefined - Simple boolean predicate that returns true if Sym is
  /// defined in this module.
  
  static inline bool PartitionByDefined(const MachOSym &Sym) {
    // FIXME: Do N_ABS or N_INDR count as defined?
    return (Sym.n_type & MachOSym::N_SECT) == MachOSym::N_SECT;
  }

}; // end struct MachOSym

/// MachOHeader - This struct contains the header information about a
/// specific architecture type/subtype pair that is emitted to the file.

struct MachOHeader {
  uint32_t  magic;      // mach magic number identifier
  uint32_t  filetype;   // type of file
  uint32_t  ncmds;      // number of load commands
  uint32_t  sizeofcmds; // the size of all the load commands
  uint32_t  flags;      // flags
  uint32_t  reserved;   // 64-bit only
  
  /// HeaderData - The actual data for the header which we are building
  /// up for emission to the file.
  std::vector<unsigned char> HeaderData;

  // Constants for the filetype field
  // see <mach-o/loader.h> for additional info on the various types
  enum { MH_OBJECT     = 1, // relocatable object file
         MH_EXECUTE    = 2, // demand paged executable file
         MH_FVMLIB     = 3, // fixed VM shared library file
         MH_CORE       = 4, // core file
         MH_PRELOAD    = 5, // preloaded executable file
         MH_DYLIB      = 6, // dynamically bound shared library
         MH_DYLINKER   = 7, // dynamic link editor
         MH_BUNDLE     = 8, // dynamically bound bundle file
         MH_DYLIB_STUB = 9, // shared library stub for static linking only
         MH_DSYM       = 10 // companion file wiht only debug sections
  };
  
  // Constants for the flags field
  enum { MH_NOUNDEFS                = 1 << 0,
            // the object file has no undefined references
         MH_INCRLINK                = 1 << 1,
            // the object file is the output of an incremental link against
            // a base file and cannot be link edited again
         MH_DYLDLINK                = 1 << 2,
            // the object file is input for the dynamic linker and cannot be
            // statically link edited again.
         MH_BINDATLOAD              = 1 << 3,
            // the object file's undefined references are bound by the
            // dynamic linker when loaded.
         MH_PREBOUND                = 1 << 4,
            // the file has its dynamic undefined references prebound
         MH_SPLIT_SEGS              = 1 << 5,
            // the file has its read-only and read-write segments split
            // see <mach/shared_memory_server.h>
         MH_LAZY_INIT               = 1 << 6,
            // the shared library init routine is to be run lazily via
            // catching memory faults to its writable segments (obsolete)
         MH_TWOLEVEL                = 1 << 7,
            // the image is using two-level namespace bindings
         MH_FORCE_FLAT              = 1 << 8,
            // the executable is forcing all images to use flat namespace
            // bindings.
         MH_NOMULTIDEFS             = 1 << 8,
            // this umbrella guarantees no multiple definitions of symbols
            // in its sub-images so the two-level namespace hints can
            // always be used.
         MH_NOFIXPREBINDING         = 1 << 10,
            // do not have dyld notify the prebidning agent about this
            // executable.
         MH_PREBINDABLE             = 1 << 11,
            // the binary is not prebound but can have its prebinding
            // redone.  only used when MH_PREBOUND is not set.
         MH_ALLMODSBOUND            = 1 << 12,
            // indicates that this binary binds to all two-level namespace
            // modules of its dependent libraries.  Only used when
            // MH_PREBINDABLE and MH_TWOLEVEL are both set.
         MH_SUBSECTIONS_VIA_SYMBOLS = 1 << 13,
            // safe to divide up the sections into sub-sections via symbols
            // for dead code stripping.
         MH_CANONICAL               = 1 << 14,
            // the binary has been canonicalized via the unprebind operation
         MH_WEAK_DEFINES            = 1 << 15,
            // the final linked image contains external weak symbols
         MH_BINDS_TO_WEAK           = 1 << 16,
            // the final linked image uses weak symbols
         MH_ALLOW_STACK_EXECUTION   = 1 << 17
            // When this bit is set, all stacks in the task will be given
            // stack execution privilege.  Only used in MH_EXECUTE filetype
  };

  MachOHeader() : magic(0), filetype(0), ncmds(0), sizeofcmds(0), flags(0),
                  reserved(0) {}

  /// cmdSize - This routine returns the size of the MachOSection as written
  /// to disk, depending on whether the destination is a 64 bit Mach-O file.
  unsigned cmdSize(bool is64Bit) const {
    if (is64Bit)
      return 8 * sizeof(uint32_t);
    else
      return 7 * sizeof(uint32_t);
  }

  /// setMagic - This routine sets the appropriate value for the 'magic'
  /// field based on pointer size and endianness.
  void setMagic(bool isLittleEndian, bool is64Bit) {
    if (isLittleEndian)
      if (is64Bit) magic = 0xcffaedfe;
      else         magic = 0xcefaedfe;
    else
      if (is64Bit) magic = 0xfeedfacf;
      else         magic = 0xfeedface;
  }

}; // end struct MachOHeader

/// MachOSegment - This struct contains the necessary information to
/// emit the load commands for each section in the file.
struct MachOSegment {
  uint32_t    cmd;      // LC_SEGMENT or LC_SEGMENT_64
  uint32_t    cmdsize;  // Total size of this struct and section commands
  std::string segname;  // segment name
  uint64_t    vmaddr;   // address of this segment
  uint64_t    vmsize;   // size of this segment, may be larger than filesize
  uint64_t    fileoff;  // offset in file
  uint64_t    filesize; // amount to read from file
  uint32_t    maxprot;  // maximum VM protection
  uint32_t    initprot; // initial VM protection
  uint32_t    nsects;   // number of sections in this segment
  uint32_t    flags;    // flags
  
  // The following constants are getting pulled in by one of the
  // system headers, which creates a neat clash with the enum.
#if !defined(VM_PROT_NONE)
#define VM_PROT_NONE    0x00
#endif
#if !defined(VM_PROT_READ)
#define VM_PROT_READ    0x01
#endif
#if !defined(VM_PROT_WRITE)
#define VM_PROT_WRITE   0x02
#endif
#if !defined(VM_PROT_EXECUTE)
#define VM_PROT_EXECUTE 0x04
#endif
#if !defined(VM_PROT_ALL)
#define VM_PROT_ALL     0x07
#endif

  // Constants for the vm protection fields
  // see <mach-o/vm_prot.h>
  enum { SEG_VM_PROT_NONE     = VM_PROT_NONE, 
         SEG_VM_PROT_READ     = VM_PROT_READ, // read permission
         SEG_VM_PROT_WRITE    = VM_PROT_WRITE, // write permission
         SEG_VM_PROT_EXECUTE  = VM_PROT_EXECUTE,
         SEG_VM_PROT_ALL      = VM_PROT_ALL
  };

  // Constants for the cmd field
  // see <mach-o/loader.h>
  enum { LC_SEGMENT    = 0x01,  // segment of this file to be mapped
         LC_SEGMENT_64 = 0x19   // 64-bit segment of this file to be mapped
  };

  /// cmdSize - This routine returns the size of the MachOSection as written
  /// to disk, depending on whether the destination is a 64 bit Mach-O file.
  unsigned cmdSize(bool is64Bit) const {
    if (is64Bit)
      return 6 * sizeof(uint32_t) + 4 * sizeof(uint64_t) + 16;
    else
      return 10 * sizeof(uint32_t) + 16;  // addresses only 32 bits
  }

  MachOSegment(const std::string &seg, bool is64Bit)
    : cmd(is64Bit ? LC_SEGMENT_64 : LC_SEGMENT), cmdsize(0), segname(seg),
      vmaddr(0), vmsize(0), fileoff(0), filesize(0), maxprot(VM_PROT_ALL),
      initprot(VM_PROT_ALL), nsects(0), flags(0) { }
};

/// MachOSection - This struct contains information about each section in a 
/// particular segment that is emitted to the file.  This is eventually
/// turned into the SectionCommand in the load command for a particlar
/// segment.

struct MachOSection : public BinaryObject { 
  std::string  sectname; // name of this section, 
  std::string  segname;  // segment this section goes in
  uint64_t  addr;        // memory address of this section
  uint32_t  offset;      // file offset of this section
  uint32_t  align;       // section alignment (power of 2)
  uint32_t  reloff;      // file offset of relocation entries
  uint32_t  nreloc;      // number of relocation entries
  uint32_t  flags;       // flags (section type and attributes)
  uint32_t  reserved1;   // reserved (for offset or index)
  uint32_t  reserved2;   // reserved (for count or sizeof)
  uint32_t  reserved3;   // reserved (64 bit only)

  /// A unique number for this section, which will be used to match symbols
  /// to the correct section.
  uint32_t Index;

  /// RelocBuffer - A buffer to hold the mach-o relocations before we write
  /// them out at the appropriate location in the file.
  std::vector<unsigned char> RelocBuffer;

  // Constants for the section types (low 8 bits of flags field)
  // see <mach-o/loader.h>
  enum { S_REGULAR = 0,
            // regular section
         S_ZEROFILL = 1,
            // zero fill on demand section
         S_CSTRING_LITERALS = 2,
            // section with only literal C strings
         S_4BYTE_LITERALS = 3,
            // section with only 4 byte literals
         S_8BYTE_LITERALS = 4,
            // section with only 8 byte literals
         S_LITERAL_POINTERS = 5, 
            // section with only pointers to literals
         S_NON_LAZY_SYMBOL_POINTERS = 6,
            // section with only non-lazy symbol pointers
         S_LAZY_SYMBOL_POINTERS = 7,
            // section with only lazy symbol pointers
         S_SYMBOL_STUBS = 8,
            // section with only symbol stubs
            // byte size of stub in the reserved2 field
         S_MOD_INIT_FUNC_POINTERS = 9,
            // section with only function pointers for initialization
         S_MOD_TERM_FUNC_POINTERS = 10,
            // section with only function pointers for termination
         S_COALESCED = 11,
            // section contains symbols that are coalesced
         S_GB_ZEROFILL = 12,
            // zero fill on demand section (that can be larger than 4GB)
         S_INTERPOSING = 13,
            // section with only pairs of function pointers for interposing
         S_16BYTE_LITERALS = 14
            // section with only 16 byte literals
  };
  
  // Constants for the section flags (high 24 bits of flags field)
  // see <mach-o/loader.h>
  enum { S_ATTR_PURE_INSTRUCTIONS   = 1 << 31,
            // section contains only true machine instructions
         S_ATTR_NO_TOC              = 1 << 30,
            // section contains coalesced symbols that are not to be in a 
            // ranlib table of contents
         S_ATTR_STRIP_STATIC_SYMS   = 1 << 29,
            // ok to strip static symbols in this section in files with the
            // MY_DYLDLINK flag
         S_ATTR_NO_DEAD_STRIP       = 1 << 28,
            // no dead stripping
         S_ATTR_LIVE_SUPPORT        = 1 << 27,
            // blocks are live if they reference live blocks
         S_ATTR_SELF_MODIFYING_CODE = 1 << 26,
            // used with i386 code stubs written on by dyld
         S_ATTR_DEBUG               = 1 << 25,
            // a debug section
         S_ATTR_SOME_INSTRUCTIONS   = 1 << 10,
            // section contains some machine instructions
         S_ATTR_EXT_RELOC           = 1 << 9,
            // section has external relocation entries
         S_ATTR_LOC_RELOC           = 1 << 8
            // section has local relocation entries
  };

  /// cmdSize - This routine returns the size of the MachOSection as written
  /// to disk, depending on whether the destination is a 64 bit Mach-O file.
  unsigned cmdSize(bool is64Bit) const {
    if (is64Bit)
      return 7 * sizeof(uint32_t) + 2 * sizeof(uint64_t) + 32;
    else
      return 9 * sizeof(uint32_t) + 32;  // addresses only 32 bits
  }

  MachOSection(const std::string &seg, const std::string &sect)
    : BinaryObject(), sectname(sect), segname(seg), addr(0), offset(0),
      align(2), reloff(0), nreloc(0), flags(0), reserved1(0), reserved2(0),
      reserved3(0) { }

}; // end struct MachOSection

/// MachOSymTab - This struct contains information about the offsets and 
/// size of symbol table information.
/// segment.
struct MachODySymTab {
  uint32_t cmd;             // LC_DYSYMTAB
  uint32_t cmdsize;         // sizeof(MachODySymTab)
  uint32_t ilocalsym;       // index to local symbols
  uint32_t nlocalsym;       // number of local symbols
  uint32_t iextdefsym;      // index to externally defined symbols
  uint32_t nextdefsym;      // number of externally defined symbols
  uint32_t iundefsym;       // index to undefined symbols
  uint32_t nundefsym;       // number of undefined symbols
  uint32_t tocoff;          // file offset to table of contents
  uint32_t ntoc;            // number of entries in table of contents
  uint32_t modtaboff;       // file offset to module table
  uint32_t nmodtab;         // number of module table entries
  uint32_t extrefsymoff;    // offset to referenced symbol table
  uint32_t nextrefsyms;     // number of referenced symbol table entries
  uint32_t indirectsymoff;  // file offset to the indirect symbol table
  uint32_t nindirectsyms;   // number of indirect symbol table entries
  uint32_t extreloff;       // offset to external relocation entries
  uint32_t nextrel;         // number of external relocation entries
  uint32_t locreloff;       // offset to local relocation entries
  uint32_t nlocrel;         // number of local relocation entries

  // Constants for the cmd field
  // see <mach-o/loader.h>
  enum { LC_DYSYMTAB = 0x0B  // dynamic link-edit symbol table info
  };
  
  MachODySymTab() : cmd(LC_DYSYMTAB), cmdsize(20 * sizeof(uint32_t)),
    ilocalsym(0), nlocalsym(0), iextdefsym(0), nextdefsym(0),
    iundefsym(0), nundefsym(0), tocoff(0), ntoc(0), modtaboff(0),
    nmodtab(0), extrefsymoff(0), nextrefsyms(0), indirectsymoff(0),
    nindirectsyms(0), extreloff(0), nextrel(0), locreloff(0), nlocrel(0) {}

}; // end struct MachODySymTab

} // end namespace llvm

#endif

