//===- lib/ReaderWriter/MachO/MachOFormat.hpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//
// This file contains all the structs and constants needed to write a 
// mach-o final linked image.  The names of the structs and constants
// are the same as in the darwin native header <mach-o/loader.h> so
// they will be familiar to anyone who has used that header.
//

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Memory.h"

#ifndef LLD_READER_WRITER_MACHO_FORMAT_H_
#define LLD_READER_WRITER_MACHO_FORMAT_H_

namespace lld {
namespace mach_o {


enum { 
  MH_MAGIC    = 0xfeedface,
  MH_MAGIC_64 = 0xfeedfacf 
};

enum {
  CPU_TYPE_ARM  =   0x0000000C,
  CPU_TYPE_I386 =   0x00000007,
  CPU_TYPE_X86_64 = 0x01000007
};

enum {
  CPU_SUBTYPE_X86_ALL    = 0x00000003,
  CPU_SUBTYPE_X86_64_ALL = 0x00000003,
  CPU_SUBTYPE_ARM_V6     = 0x00000006,
  CPU_SUBTYPE_ARM_V7     = 0x00000009,
  CPU_SUBTYPE_ARM_V7S    = 0x0000000B
};

enum {
  MH_OBJECT     = 0x1,
  MH_EXECUTE    = 0x2,
  MH_PRELOAD    = 0x5,
  MH_DYLIB      = 0x6,
  MH_DYLINKER   = 0x7,
  MH_BUNDLE     = 0x8,
  MH_DYLIB_STUB = 0x9,
  MH_KEXT_BUNDLE= 0xB
};


//
// Every mach-o file starts with this header.  The header size is
// 28 bytes for 32-bit architecures and 32-bytes for 64-bit architectures.
//
class mach_header {
public:
  uint32_t    magic;  
  uint32_t    cputype;
  uint32_t    cpusubtype;  
  uint32_t    filetype;  
  uint32_t    ncmds;    
  uint32_t    sizeofcmds;  
  uint32_t    flags;  
  uint32_t    reserved;  
 
  uint64_t size() {
    return (magic == 0xfeedfacf) ? 32 : 28;
  }
  
  void copyTo(uint8_t *to, bool swap=false) {
    ::memcpy(to, (char*)&magic, this->size());
  }
 
  void recordLoadCommand(const class load_command *lc);    
};


//
// Every mach-o file has a list of load commands after the mach_header.
// Each load command starts with a type and length, so you can iterate
// through the load commands even if you don't understand the content
// of a particular type.
//
// The model for handling endianness and 32 vs 64 bitness is that the in-memory
// object is always 64-bit and the native endianess.  The endianess swapping
// and pointer sizing is done when writing (copyTo method) or when reading
// (constructor that takes a buffer).
//
// The load_command subclasses are designed so to mirror the traditional "C"
// structs, so you can get and set the same field names (e.g. seg->vmaddr = 0).
//
class load_command {
public:
  const uint32_t  cmd;        // type of load command
  const uint32_t  cmdsize;    // length of load command including this header
  
  load_command(uint32_t cmdNumber, uint32_t sz, bool is64, bool align=false)
    : cmd(cmdNumber), cmdsize(pointerAlign(sz, is64, align)) {
  }
  
  virtual ~load_command() {
  }
  
  virtual void copyTo(uint8_t *to, bool swap=false) = 0;
private:
  // Load commands must be pointer-size aligned. Most load commands are
  // a fixed size, so there is a runtime assert to check those.  For variable
  // length load commands, setting the align option to true will add padding
  // at the end of the load command to round up its size for proper alignment.
  uint32_t pointerAlign(uint32_t size, bool is64, bool align) {
    if ( align ) {
       if ( is64 )
        return (size + 7) & (-8);
      else
        return (size + 3) & (-4);
    }
    else {
      if ( is64 )
        assert((size % 8) == 0);
      else
        assert((size % 4) == 0);
      return size;
    }
  }

};

inline void mach_header::recordLoadCommand(const load_command *lc) {
  ++ncmds;
  sizeofcmds += lc->cmdsize;
}

// Supported load command types
enum {
  LC_SEGMENT        = 0x00000001,
  LC_SYMTAB         = 0x00000002,
  LC_UNIXTHREAD     = 0x00000005,
  LC_LOAD_DYLIB     = 0x0000000C,
  LC_LOAD_DYLINKER  = 0x0000000E,
  LC_SEGMENT_64     = 0x00000019,
  LC_MAIN           = 0x80000028,
  LC_DYLD_INFO_ONLY = 0x80000022
};

// Memory protection bit used in segment_command.initprot
enum {
  VM_PROT_NONE    = 0x0,
  VM_PROT_READ    = 0x1,
  VM_PROT_WRITE   = 0x2,
  VM_PROT_EXECUTE = 0x4,
};

// Bits for the section.flags field
enum {
  // Section "type" is the low byte
  SECTION_TYPE              = 0x000000FF,
  S_REGULAR                 = 0x00000000,
  S_ZEROFILL                = 0x00000001,
  S_CSTRING_LITERALS        = 0x00000002,
  S_NON_LAZY_SYMBOL_POINTERS= 0x00000006,
  S_LAZY_SYMBOL_POINTERS    = 0x00000007,
  S_SYMBOL_STUBS            = 0x00000008,
  
  // Other bits in section.flags
  S_ATTR_PURE_INSTRUCTIONS  = 0x80000000,
  S_ATTR_SOME_INSTRUCTIONS  = 0x00000400
};


// section record for 32-bit architectures
struct section {
  char      sectname[16];  
  char      segname[16];  
  uint32_t  addr;  
  uint32_t  size;    
  uint32_t  offset;  
  uint32_t  align;    
  uint32_t  reloff;  
  uint32_t  nreloc;    
  uint32_t  flags;    
  uint32_t  reserved1;
  uint32_t  reserved2;  
};

// section record for 64-bit architectures
struct section_64 {
  char      sectname[16];  
  char      segname[16];  
  uint64_t  addr;  
  uint64_t  size;    
  uint32_t  offset;  
  uint32_t  align;    
  uint32_t  reloff;  
  uint32_t  nreloc;    
  uint32_t  flags;    
  uint32_t  reserved1;
  uint32_t  reserved2;  
  uint32_t  reserved3;
};


//
// A segment load command has a fixed set of fields followed by an 'nsect'
// array of section records.  The in-memory object uses a pointer to
// a dynamically allocated array of sections.  
//
class segment_command : public load_command {
public:
  char      segname[16];  
  uint64_t  vmaddr;    
  uint64_t  vmsize;    
  uint64_t  fileoff;  
  uint64_t  filesize;  
  uint32_t  maxprot;  
  uint32_t  initprot;  
  uint32_t  nsects;    
  uint32_t  flags;  
  section_64 *sections;
  
  segment_command(unsigned sectCount, bool is64)
    : load_command((is64 ? LC_SEGMENT_64 : LC_SEGMENT), 
                   (is64 ? (72 + sectCount*80) : (56 + sectCount*68)),
                   is64),
     vmaddr(0), vmsize(0), fileoff(0), filesize(0), 
      maxprot(0), initprot(0), nsects(sectCount), flags(0) {
    sections = new section_64[sectCount];
    this->nsects = sectCount;
  }
  
  ~segment_command() {
    delete sections;
  }
  
  void copyTo(uint8_t *to, bool swap) {
    if ( swap ) {
      assert(0 && "non-native endianness not supported yet");
    }
    else {
      if( is64() ) {
        // in-memory matches on-disk, so copy segment fields followed by sections
        ::memcpy(to, (uint8_t*)&cmd, 72);
        if ( nsects != 0 )
          ::memcpy(&to[72], sections, sizeof(section_64)*nsects);
      }
      else {
        // on-disk is 32-bit struct, so copy each field
        ::memcpy(to, (uint8_t*)&cmd, 24);
        copy32(to, 24, vmaddr);
        copy32(to, 28, vmsize);
        copy32(to, 32, fileoff);
        copy32(to, 36, filesize);
        copy32(to, 40, maxprot);
        copy32(to, 44, initprot);
        copy32(to, 48, nsects);
        copy32(to, 52, flags);
        for(uint32_t i=0; i < nsects; ++i) {
          unsigned off = 56+i*68;
          ::memcpy(&to[off], sections[i].sectname, 32);
          copy32(to, off+32, sections[i].addr);
          copy32(to, off+36, sections[i].size);
          copy32(to, off+40, sections[i].offset);
          copy32(to, off+44, sections[i].align);
          copy32(to, off+48, sections[i].reloff);
          copy32(to, off+52, sections[i].nreloc);
          copy32(to, off+56, sections[i].flags);
          copy32(to, off+60, sections[i].reserved1);
          copy32(to, off+64, sections[i].reserved2);
        }
      }
    }
  }

private:
  void copy32(uint8_t *to, unsigned offset, uint64_t value) {
    uint32_t value32 = value; // FIXME: range check
    ::memcpy(&to[offset], &value32, sizeof(uint32_t));
  }

  bool is64() { 
    return (cmd == LC_SEGMENT_64); 
  }
};



//
// The dylinker_command contains the path to the dynamic loader to use
// with the program (e.g. "/usr/lib/dyld"). So, it is variable length.
// But load commands must be pointer size aligned.
//
//
class dylinker_command : public load_command {
public:
  uint32_t  name_offset;
private:
  StringRef _name;
public:
  dylinker_command(StringRef path, bool is64) 
    : load_command(LC_LOAD_DYLINKER,12 + path.size(), is64, true), 
       name_offset(12), _name(path) {
  }

  virtual void copyTo(uint8_t *to, bool swap=false) {
    if ( swap ) {
      assert(0 && "non-native endianness not supported yet");
    }
    else {
      // in-memory matches on-disk, so copy first fields followed by path
      ::memcpy(to, (uint8_t*)&cmd, 12);
      ::memcpy(&to[12], _name.data(), _name.size());
      ::memset(&to[12+_name.size()], 0, cmdsize-(12+_name.size()));
    }
  }

};



//
// The symtab_command just holds the offset to the array of nlist structs
// and the offsets to the string pool for all symbol names.
//
class symtab_command : public load_command {
public:
  uint32_t  symoff;  
  uint32_t  nsyms;  
  uint32_t  stroff;  
  uint32_t  strsize;  

  symtab_command(bool is64) 
    : load_command(LC_SYMTAB, 24, is64), 
      symoff(0), nsyms(0), stroff(0), strsize(0) {
  }
  
  virtual void copyTo(uint8_t *to, bool swap=false) {
    if ( swap ) {
      assert(0 && "non-native endianness not supported yet");
    }
    else {
      // in-memory matches on-disk, so copy fields 
      ::memcpy(to, (uint8_t*)&cmd, 24);
    }
  }
  
};


//
// The entry_point_command load command holds the offset to the function
// _main in a dynamic executable.  
//
class entry_point_command : public load_command {
public:
  uint64_t  entryoff;  
  uint64_t  stacksize; 

  entry_point_command(bool is64) 
    : load_command(LC_MAIN, 24, is64), entryoff(0), stacksize(0) {
  }
  
  virtual void copyTo(uint8_t *to, bool swap=false) {
    if ( swap ) {
      assert(0 && "non-native endianness not supported yet");
    }
    else {
      // in-memory matches on-disk, so copy fields 
      ::memcpy(to, (uint8_t*)&cmd, 24);
    }
  }
};


//
// The thread_command load command holds the set of initial register values
// for a dynamic executable.  In reality, only the PC and SP are used.
//
class thread_command : public load_command {
public:
	uint32_t	fields_flavor;
	uint32_t	fields_count;
private:
  uint32_t   _cpuType;
  uint8_t   *_registerArray;

public:
  thread_command(uint32_t cpuType, bool is64) 
    : load_command(LC_UNIXTHREAD, 16+registersBufferSize(cpuType), is64),
      fields_count(registersBufferSize(cpuType)/4), _cpuType(cpuType) {
    switch ( cpuType ) {
      case CPU_TYPE_I386:
        fields_flavor = 1;  // i386_THREAD_STATE
        break;
      case CPU_TYPE_X86_64:
        fields_flavor = 4;  // x86_THREAD_STATE64;
        break;
      case CPU_TYPE_ARM:
        fields_flavor = 1;  // ARM_THREAD_STATE
        break;
      default:
        assert(0 && "unsupported cpu type");
    }
    _registerArray = reinterpret_cast<uint8_t*>(
                                    ::calloc(registersBufferSize(cpuType), 1));
    assert(_registerArray);
  }
  
  virtual void copyTo(uint8_t *to, bool swap=false) {
    if ( swap ) {
      assert(0 && "non-native endianness not supported yet");
    }
    else {
      // in-memory matches on-disk, so copy fixed fields 
      ::memcpy(to, (uint8_t*)&cmd, 16);
      // that register array
      ::memcpy(&to[16], _registerArray, registersBufferSize(_cpuType));
    }
  }
  
  void setPC(uint64_t pc) {
    uint32_t *regs32 = reinterpret_cast<uint32_t*>(_registerArray);
    uint64_t *regs64 = reinterpret_cast<uint64_t*>(_registerArray);
    switch ( _cpuType ) {
      case CPU_TYPE_I386:
        regs32[10] = pc;
        break;
      case CPU_TYPE_X86_64:
        regs64[16] = pc;
        break;
      case CPU_TYPE_ARM:
        regs32[15] = pc;
        break;
      default:
        assert(0 && "unsupported cpu type");
    }
  }
  
  virtual ~thread_command() {
    ::free(_registerArray); 
  }

private:
  uint32_t registersBufferSize(uint32_t cpuType) {
    switch ( cpuType ) {
      case CPU_TYPE_I386:
        return 64;        // i386_THREAD_STATE_COUNT * 4
      case CPU_TYPE_X86_64:
        return 168;       // x86_THREAD_STATE64_COUNT * 4
      case CPU_TYPE_ARM:
        return 68;        // ARM_THREAD_STATE_COUNT * 4
    }
    assert(0 && "unsupported cpu type");
    return 0;
  }
    
    
  
};





//
// The dylib_command load command holds the name/path of a dynamic shared
// library which this mach-o image depends on.
//
struct dylib_command : public load_command {
  uint32_t  name_offset;
  uint32_t  timestamp;
  uint32_t  current_version;    
  uint32_t  compatibility_version;
private:  
  StringRef _loadPath;
public:
  
  dylib_command(StringRef path, bool is64) 
    : load_command(LC_LOAD_DYLIB, 24 + path.size(), is64, true), 
      name_offset(24), timestamp(0), 
      current_version(0x10000), compatibility_version(0x10000),
      _loadPath(path) {
  }
  
  virtual void copyTo(uint8_t *to, bool swap=false) {
    if ( swap ) {
      assert(0 && "non-native endianness not supported yet");
    }
    else {
      // in-memory matches on-disk, so copy first fields followed by path
      ::memcpy(to, (uint8_t*)&cmd, 24);
      ::memcpy(&to[24], _loadPath.data(), _loadPath.size());
      ::memset(&to[24+_loadPath.size()], 0, cmdsize-(24+_loadPath.size()));
    }
  }

};


//
// The dyld_info_command load command holds the offsets to various tables
// of information needed by dyld to prepare the image for execution.
//
struct dyld_info_command : public load_command {
  uint32_t   rebase_off;  
  uint32_t   rebase_size;  
  uint32_t   bind_off;  
  uint32_t   bind_size;  
  uint32_t   weak_bind_off;  
  uint32_t   weak_bind_size; 
  uint32_t   lazy_bind_off;
  uint32_t   lazy_bind_size; 
  uint32_t   export_off;  
  uint32_t   export_size;  

  dyld_info_command(bool is64) 
    : load_command(LC_DYLD_INFO_ONLY, 48, is64), 
        rebase_off(0), rebase_size(0),
        bind_off(0), bind_size(0), weak_bind_off(0), weak_bind_size(0), 
        lazy_bind_off(0), lazy_bind_size(0), export_off(0), export_size(0) {
   }
  
  virtual void copyTo(uint8_t *to, bool swap=false) {
    if ( swap ) {
      assert(0 && "non-native endianness not supported yet");
    }
    else {
      // in-memory matches on-disk, so copy fields 
      ::memcpy(to, (uint8_t*)&cmd, 48);
    }
  }
};


enum {
  BIND_TYPE_POINTER               = 1,
  BIND_TYPE_TEXT_ABSOLUTE32        = 2,
  BIND_TYPE_TEXT_PCREL32          = 3
};

enum {
  BIND_SPECIAL_DYLIB_SELF             = 0,
  BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE  = -1,
  BIND_SPECIAL_DYLIB_FLAT_LOOKUP      =  -2
};

enum {
  BIND_SYMBOL_FLAGS_WEAK_IMPORT           = 0x1,
  BIND_SYMBOL_FLAGS_NON_WEAK_DEFINITION    =  0x8
};

enum {
  BIND_OPCODE_MASK                              = 0xF0,
  BIND_IMMEDIATE_MASK                           = 0x0F,
  BIND_OPCODE_DONE                              = 0x00,
  BIND_OPCODE_SET_DYLIB_ORDINAL_IMM             = 0x10,
  BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB            = 0x20,
  BIND_OPCODE_SET_DYLIB_SPECIAL_IMM             = 0x30,
  BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM     = 0x40,
  BIND_OPCODE_SET_TYPE_IMM                      = 0x50,
  BIND_OPCODE_SET_ADDEND_SLEB                   = 0x60,
  BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB       = 0x70,
  BIND_OPCODE_ADD_ADDR_ULEB                     = 0x80,
  BIND_OPCODE_DO_BIND                           = 0x90,
  BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB             = 0xA0,
  BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED       = 0xB0,
  BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB  = 0xC0
};




enum {
  N_UNDF = 0x00,
  N_EXT  = 0x01,
  N_PEXT = 0x10,
  N_SECT = 0x0e
};

class nlist {
public:
  uint32_t  n_strx; 
  uint8_t   n_type; 
  uint8_t   n_sect;   
  uint16_t  n_desc;   
  uint64_t  n_value;    
  
  static unsigned size(bool is64) {
    return (is64 ? 16 : 12);
  }
  
  void copyTo(uint8_t *to, bool is64, bool swap=false) {
    if ( swap ) {
      assert(0 && "non-native endianness not supported yet");
    }
    else {
      if ( is64 ) {
        // in-memory matches on-disk, so just copy whole struct 
        ::memcpy(to, (uint8_t*)&n_strx, 16);
      }
      else {
        // on-disk uses 32-bit n_value, so special case n_value
        ::memcpy(to, (uint8_t*)&n_strx, 8);
        uint32_t value32 = n_value; // FIXME: range check
        ::memcpy(&to[8], &value32, sizeof(uint32_t));
      }
    }
  }
};





} // namespace mach_o
} // namespace lld



#endif // LLD_READER_WRITER_MACHO_FORMAT_H_

