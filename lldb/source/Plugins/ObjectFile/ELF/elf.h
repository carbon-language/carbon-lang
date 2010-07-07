//===-- elf.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __elf_h__
#define __elf_h__

//----------------------------------------------------------------------
// Typedefs for ELF32.
//----------------------------------------------------------------------
typedef uint16_t    Elf32_Half;
typedef uint32_t    Elf32_Word;
typedef int32_t     Elf32_Sword;
typedef uint32_t    Elf32_Addr;
typedef uint32_t    Elf32_Off;

//----------------------------------------------------------------------
// Typedefs for ELF64.
//----------------------------------------------------------------------
typedef uint16_t    Elf64_Half;
typedef uint32_t    Elf64_Word;
typedef int32_t     Elf64_Sword;
typedef uint64_t    Elf64_Xword;
typedef int64_t     Elf64_Sxword;
typedef uint64_t    Elf64_Addr;
typedef uint64_t    Elf64_Off;

#define EI_NIDENT 16

//----------------------------------------------------------------------
// ELF Headers
//----------------------------------------------------------------------
typedef struct Elf32_Ehdr_Tag
{
    unsigned char e_ident[EI_NIDENT];
    Elf32_Half  e_type;
    Elf32_Half  e_machine;
    Elf32_Word  e_version;
    Elf32_Addr  e_entry;
    Elf32_Off   e_phoff;
    Elf32_Off   e_shoff;
    Elf32_Word  e_flags;
    Elf32_Half  e_ehsize;
    Elf32_Half  e_phentsize;
    Elf32_Half  e_phnum;
    Elf32_Half  e_shentsize;
    Elf32_Half  e_shnum;
    Elf32_Half  e_shstrndx;
} Elf32_Ehdr;

typedef struct Elf64_Ehdr_Tag
{
    unsigned char e_ident[EI_NIDENT];
    Elf64_Half  e_type;
    Elf64_Half  e_machine;
    Elf64_Word  e_version;
    Elf64_Addr  e_entry;
    Elf64_Off   e_phoff;
    Elf64_Off   e_shoff;
    Elf64_Word  e_flags;
    Elf64_Half  e_ehsize;
    Elf64_Half  e_phentsize;
    Elf64_Half  e_phnum;
    Elf64_Half  e_shentsize;
    Elf64_Half  e_shnum;
    Elf64_Half  e_shstrndx;
} Elf64_Ehdr;

//----------------------------------------------------------------------
// e_type
//
// This member identifies the object file type.
//----------------------------------------------------------------------
#define ET_NONE     0       // No file type
#define ET_REL      1       // Relocatable file
#define ET_EXEC     2       // Executable file
#define ET_DYN      3       // Shared object file
#define ET_CORE     4       // Core file
#define ET_LOPROC   0xff00  // Processor-specific
#define ET_HIPROC   0xffff  // Processor-specific

//----------------------------------------------------------------------
// e_machine
//
// Machine Type
//----------------------------------------------------------------------
#define EM_NONE     0   // No machine
#define EM_M32      1   // AT&T WE 32100
#define EM_SPARC    2   // SPARC
#define EM_386      3   // Intel 80386
#define EM_68K      4   // Motorola 68000
#define EM_88K      5   // Motorola 88000
#define EM_860      7   // Intel 80860
#define EM_MIPS     8   // MIPS RS3000
#define EM_PPC      20  // PowerPC
#define EM_PPC64    21  // PowerPC64
#define EM_ARM      40  // ARM
#define EM_X86_64   62  // AMD x86-64


//----------------------------------------------------------------------
// e_ident indexes
//----------------------------------------------------------------------
#define EI_MAG0     0   // File identification
#define EI_MAG1     1   // File identification
#define EI_MAG2     2   // File identification
#define EI_MAG3     3   // File identification
#define EI_CLASS    4   // File class
#define EI_DATA     5   // Data encoding
#define EI_VERSION  6   // File version
#define EI_PAD      7   // Start of padding bytes

//----------------------------------------------------------------------
// EI_CLASS definitions
//----------------------------------------------------------------------
#define ELFCLASS32 1 // 32-bit object file
#define ELFCLASS64 2 // 64-bit object file

//----------------------------------------------------------------------
// EI_DATA definitions
//----------------------------------------------------------------------
#define ELFDATANONE 0   // Invalid data encoding
#define ELFDATA2LSB 1   // Little Endian
#define ELFDATA2MSB 2   // Big Endian

//----------------------------------------------------------------------
// Section Headers
//----------------------------------------------------------------------
typedef struct Elf32_Shdr_Tag
{
    Elf32_Word  sh_name;
    Elf32_Word  sh_type;
    Elf32_Word  sh_flags;
    Elf32_Addr  sh_addr;
    Elf32_Off   sh_offset;
    Elf32_Word  sh_size;
    Elf32_Word  sh_link;
    Elf32_Word  sh_info;
    Elf32_Word  sh_addralign;
    Elf32_Word  sh_entsize;
} Elf32_Shdr;

typedef struct Elf64_Shdr_Tag
{
    Elf64_Word  sh_name;
    Elf64_Word  sh_type;
    Elf64_Xword sh_flags;
    Elf64_Addr  sh_addr;
    Elf64_Off   sh_offset;
    Elf64_Xword sh_size;
    Elf64_Word  sh_link;
    Elf64_Word  sh_info;
    Elf64_Xword sh_addralign;
    Elf64_Xword sh_entsize;
} Elf64_Shdr;

//----------------------------------------------------------------------
// Section Types (sh_type)
//----------------------------------------------------------------------
#define SHT_NULL        0
#define SHT_PROGBITS    1
#define SHT_SYMTAB      2
#define SHT_STRTAB      3
#define SHT_RELA        4
#define SHT_HASH        5
#define SHT_DYNAMIC     6
#define SHT_NOTE        7
#define SHT_NOBITS      8
#define SHT_REL         9
#define SHT_SHLIB       10
#define SHT_DYNSYM      11
#define SHT_LOPROC      0x70000000
#define SHT_HIPROC      0x7fffffff
#define SHT_LOUSER      0x80000000
#define SHT_HIUSER      0xffffffff

//----------------------------------------------------------------------
// Special Section Indexes
//----------------------------------------------------------------------
#define SHN_UNDEF       0
#define SHN_LORESERVE   0xff00
#define SHN_LOPROC      0xff00
#define SHN_HIPROC      0xff1f
#define SHN_ABS         0xfff1
#define SHN_COMMON      0xfff2
#define SHN_HIRESERVE   0xffff

//----------------------------------------------------------------------
// Section Attribute Flags (sh_flags)
//----------------------------------------------------------------------
#define SHF_WRITE       0x1
#define SHF_ALLOC       0x2
#define SHF_EXECINSTR   0x4
#define SHF_MASKPROC    0xf0000000


//----------------------------------------------------------------------
// Symbol Table Entry Headers
//----------------------------------------------------------------------
typedef struct Elf32_Sym_Tag
{
    Elf32_Word      st_name;
    Elf32_Addr      st_value;
    Elf32_Word      st_size;
    unsigned char   st_info;
    unsigned char   st_other;
    Elf32_Half      st_shndx;
} Elf32_Sym;

typedef struct Elf64_Sym_Tag
{
    Elf64_Word      st_name;
    unsigned char   st_info;
    unsigned char   st_other;
    Elf64_Half      st_shndx;
    Elf64_Addr      st_value;
    Elf64_Xword     st_size;
} Elf64_Sym;

//----------------------------------------------------------------------
// Accessors to the binding and type bits in the st_info field of a 
// symbol table entry.  Valid for both 32 and 64 bit variations.
//----------------------------------------------------------------------
#define ELF_ST_BIND(i)    ((i)>>4)
#define ELF_ST_TYPE(i)    ((i)&0xf)
#define ELF_ST_INFO(b,t)  (((b)<<4)+((t)&0xf))

// ST_BIND
#define STB_LOCAL   0
#define STB_GLOBAL  1
#define STB_WEAK    2
#define STB_LOPROC  13
#define STB_HIPROC  15

// ST_TYPE
#define STT_NOTYPE  0
#define STT_OBJECT  1
#define STT_FUNC    2
#define STT_SECTION 3
#define STT_FILE    4
#define STT_LOPROC  13
#define STT_HIPROC  15


//----------------------------------------------------------------------
// Relocation Entries
//----------------------------------------------------------------------
typedef struct Elf32_Rel_Tag
{
    Elf32_Addr  r_offset;
    Elf32_Word  r_info;
} Elf32_Rel;

typedef struct Elf32_Rela_Tag
{
    Elf32_Addr  r_offset;
    Elf32_Word  r_info;
    Elf32_Sword r_addend;
} Elf32_Rela;

typedef struct Elf64_Rel_Tag
{
    Elf64_Addr  r_offset;
    Elf64_Word  r_info;
} Elf64_Rel;

typedef struct Elf64_Rela_Tag
{
    Elf64_Addr  r_offset;
    Elf64_Word  r_info;
    Elf64_Sword r_addend;
} Elf64_Rela;

#define ELF32_R_SYM(i)      ((i)>>8)
#define ELF32_R_TYPE(i)     ((unsignedchar)(i))
#define ELF32_R_INFO(s,t)   (((s)<<8)+(unsignedchar)(t))

//----------------------------------------------------------------------
// Dynamic Table Entry Headers
//----------------------------------------------------------------------
typedef struct Elf64_Dyn_Tag
{
    Elf64_Sxword d_tag;
    union
    {
        Elf64_Xword d_val;
        Elf64_Addr  d_ptr;
    } d_un;
} Elf64_Dyn;

#define DT_NULL         0       // Marks end of dynamic array.
#define DT_NEEDED       1       // String table offset of needed library.
#define DT_PLTRELSZ     2       // Size of relocation entries in PLT.
#define DT_PLTGOT       3       // Address associated with linkage table.
#define DT_HASH         4       // Address of symbolic hash table.
#define DT_STRTAB       5       // Address of dynamic string table.
#define DT_SYMTAB       6       // Address of dynamic symbol table.
#define DT_RELA         7       // Address of relocation table (Rela entries).
#define DT_RELASZ       8       // Size of Rela relocation table.
#define DT_RELAENT      9       // Size of a Rela relocation entry.
#define DT_STRSZ        10      // Total size of the string table.
#define DT_SYMENT       11      // Size of a symbol table entry.
#define DT_INIT         12      // Address of initialization function.
#define DT_FINI         13      // Address of termination function.
#define DT_SONAME       14      // String table offset of a shared objects name.
#define DT_RPATH        15      // String table offset of library search path.
#define DT_SYMBOLIC     16      // Changes symbol resolution algorithm.
#define DT_REL          17      // Address of relocation table (Rel entries).
#define DT_RELSZ        18      // Size of Rel relocation table.
#define DT_RELENT       19      // Size of a Rel relocation entry.
#define DT_PLTREL       20      // Type of relocation entry used for linking.
#define DT_DEBUG        21      // Reserved for debugger.
#define DT_TEXTREL      22      // Relocations exist for non-writable segements.
#define DT_JMPREL       23      // Address of relocations associated with PLT.
#define DT_BIND_NOW     24      // Process all relocations before execution.
#define DT_INIT_ARRAY   25      // Pointer to array of initialization functions.
#define DT_FINI_ARRAY   26      // Pointer to array of termination functions.
#define DT_INIT_ARRAYSZ 27      // Size of DT_INIT_ARRAY.
#define DT_FINI_ARRAYSZ 28      // Size of DT_FINI_ARRAY.
#define DT_LOOS         0x60000000 // Start of environment specific tags.
#define DT_HIOS         0x6FFFFFFF // End of environment specific tags.
#define DT_LOPROC       0x70000000 // Start of processor specific tags.
#define DT_HIPROC       0x7FFFFFFF // End of processor specific tags.


//----------------------------------------------------------------------
// Program Headers
//----------------------------------------------------------------------
typedef struct Elf32_Phdr_Tag
{
    Elf32_Word  p_type;
    Elf32_Off   p_offset;
    Elf32_Addr  p_vaddr;
    Elf32_Addr  p_paddr;
    Elf32_Word  p_filesz;
    Elf32_Word  p_memsz;
    Elf32_Word  p_flags;
    Elf32_Word  p_align;
} Elf32_Phdr;

typedef struct Elf64_Phdr_Tag
{
    Elf64_Word  p_type;
    Elf64_Off   p_offset;
    Elf64_Addr  p_vaddr;
    Elf64_Addr  p_paddr;
    Elf64_Word  p_filesz;
    Elf64_Word  p_memsz;
    Elf64_Word  p_flags;
    Elf64_Word  p_align;
} Elf64_Phdr;

//----------------------------------------------------------------------
// Program Header Type (p_type)
//----------------------------------------------------------------------
#define PT_NULL     0
#define PT_LOAD     1
#define PT_DYNAMIC  2
#define PT_INTERP   3
#define PT_NOTE     4
#define PT_SHLIB    5
#define PT_PHDR     6
#define PT_LOPROC   0x70000000
#define PT_HIPROC   0x7fffffff

#define PF_X        (1 << 0)    // executable
#define PF_W        (1 << 1)    // writable
#define PF_R        (1 << 2)    // readable


#endif // __elf_h__
