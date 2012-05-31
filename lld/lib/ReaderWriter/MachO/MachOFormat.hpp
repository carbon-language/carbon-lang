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

class load_command {
public:
  uint32_t  cmd;  
  uint32_t  cmdsize;
  
  void copyTo(uint8_t* to, bool swap=false) {
    ::memcpy(to, (uint8_t*)&cmd, cmdsize);
  }
};

enum { 
  MH_MAGIC = 0xfeedface,
  MAGIC_64 = 0xfeedfacf 
};

enum {
  CPU_TYPE_I386 =   0x00000007,
  CPU_TYPE_X86_64 = 0x01000007
};

enum {
  CPU_SUBTYPE_X86_ALL = 0x00000003,
  CPU_SUBTYPE_X86_64_ALL = 0x00000003
};

enum {
  MH_OBJECT     = 0x1,
  MH_EXECUTE    = 0x2,
  MH_DYLIB      = 0x6,
  MH_DYLINKER   = 0x7,
  MH_BUNDLE     = 0x8,
  MH_DYLIB_STUB = 0x9,
  MH_KEXT_BUNDLE= 0xB
};


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
  
  void copyTo(uint8_t* to, bool swap=false) {
    ::memcpy(to, (char*)&magic, this->size());
  }
 
  void recordLoadCommand(const class load_command* lc) {
    ++ncmds;
    sizeofcmds += lc->cmdsize;
  }
    

};

enum {
  SECTION_TYPE              = 0x000000FF,
  S_REGULAR                 = 0x00000000,
  S_ZEROFILL                = 0x00000001,
  S_CSTRING_LITERALS        = 0x00000002,
  S_NON_LAZY_SYMBOL_POINTERS= 0x00000006,
  S_LAZY_SYMBOL_POINTERS    = 0x00000007,
  S_SYMBOL_STUBS            = 0x00000008,
  
  S_ATTR_PURE_INSTRUCTIONS  = 0x80000000,
  S_ATTR_SOME_INSTRUCTIONS  = 0x00000400
};

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

enum {
  LC_SEGMENT_64 = 0x19
};

enum {
  VM_PROT_NONE    = 0x0,
  VM_PROT_READ    = 0x1,
  VM_PROT_WRITE   = 0x2,
  VM_PROT_EXECUTE = 0x4,
};



class segment_command_64 : public load_command {
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
  section_64 sections[1];
  
  // The segment_command_64 load commands has a nsect trailing
  // section_64 records appended to the end.
  static segment_command_64* make(unsigned sectCount) {
    // Compute size in portable way.  Can't use offsetof() in non-POD class.
    // Can't use zero size sections[] array above.
    // So, since sizeof() already includes one section_64, subtract it off.
    unsigned size = sizeof(segment_command_64) 
                    + ((int)sectCount -1) * sizeof(section_64);
    segment_command_64* result = reinterpret_cast<segment_command_64*>
                                                          (::calloc(1, size));
    result->cmd = LC_SEGMENT_64;
    result->cmdsize = size;
    result->nsects = sectCount;
    return result;
  }
  
};


enum {
  LC_LOAD_DYLINKER = 0xe
};


class dylinker_command : public load_command {
public:
  uint32_t  name_offset;
  char      name[1];
  
  static dylinker_command* make(const char* path) {
    unsigned size = (sizeof(dylinker_command) + strlen(path) + 7) & (-8);
    dylinker_command* result = reinterpret_cast<dylinker_command*>
                                                          (::calloc(1, size));
    result->cmd = LC_LOAD_DYLINKER;
    result->cmdsize = size;
    result->name_offset = 12;
    strcpy(result->name, path);
    return result;
  }
};






enum {
  N_UNDF = 0x00,
  N_EXT  = 0x01,
  N_PEXT = 0x10,
  N_SECT = 0x0e
};

class nlist_64 {
public:
  uint32_t  n_strx; 
  uint8_t   n_type; 
  uint8_t   n_sect;   
  uint16_t  n_desc;   
  uint64_t  n_value;    

   void copyTo(uint8_t* to, bool swap=false) {
     ::memcpy(to, (uint8_t*)&n_strx, 16);
  }


};


enum {
  LC_SYMTAB  =  0x2
};

class symtab_command : public load_command {
public:
  uint32_t  symoff;  
  uint32_t  nsyms;  
  uint32_t  stroff;  
  uint32_t  strsize;  

  static symtab_command* make() {
    unsigned size = sizeof(symtab_command);
    symtab_command* result = reinterpret_cast<symtab_command*>
                                                          (::calloc(1, size));
    result->cmd = LC_SYMTAB;
    result->cmdsize = size;
    return result;
  }
};


enum {
  LC_MAIN = 0x80000028
};

class entry_point_command : public load_command {
public:
  uint64_t  entryoff;  /* file (__TEXT) offset of main() */
  uint64_t  stacksize;/* if not zero, initial stack size */

  static entry_point_command* make() {
    unsigned size = sizeof(entry_point_command);
    entry_point_command* result = reinterpret_cast<entry_point_command*>
                                                          (::calloc(1, size));
    result->cmd = LC_MAIN;
    result->cmdsize = size;
    return result;
  }
};

enum {
  LC_DYLD_INFO_ONLY = 0x80000022
};

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

  static dyld_info_command* make() {
    unsigned size = sizeof(dyld_info_command);
    dyld_info_command* result = reinterpret_cast<dyld_info_command*>
    (::calloc(1, size));
    result->cmd = LC_DYLD_INFO_ONLY;
    result->cmdsize = size;
    return result;
  }
};


enum {
  LC_LOAD_DYLIB = 0xC
};
  

struct dylib_command : public load_command {
  uint32_t  name_offset;
  uint32_t  timestamp;
  uint32_t  current_version;    
  uint32_t  compatibility_version;
  char      name[1];
  
  static dylib_command* make(const char* path) {
    unsigned size = (sizeof(dylib_command) + strlen(path) + 7) & (-8);
    dylib_command* result = reinterpret_cast<dylib_command*>
    (::calloc(1, size));
    result->cmd = LC_LOAD_DYLIB;
    result->cmdsize = size;
    result->name_offset = 24;
    result->name_offset = 24;
    result->timestamp = 0;
    result->current_version = 0x10000;
    result->compatibility_version = 0x10000;
    strcpy(result->name, path);
    return result;
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




} // namespace mach_o
} // namespace lld



#endif // LLD_READER_WRITER_MACHO_FORMAT_H_

