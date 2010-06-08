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

typedef uint16_t    Elf32_Half;
typedef uint32_t    Elf32_Word;
typedef int32_t     Elf32_Sword;
typedef uint32_t    Elf32_Addr;
typedef uint32_t    Elf32_Off;


#define EI_NIDENT 16

//----------------------------------------------------------------------
// ELF Header
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
// EI_DATA definitions
//----------------------------------------------------------------------
#define ELFDATANONE 0   // Invalid data encoding
#define ELFDATA2LSB 1   // Little Endian
#define ELFDATA2MSB 2   // Big Endian

//----------------------------------------------------------------------
// Section Header
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
// Symbol Table Entry Header
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


#define ELF32_ST_BIND(i)    ((i)>>4)
#define ELF32_ST_TYPE(i)    ((i)&0xf)
#define ELF32_ST_INFO(b,t)  (((b)<<4)+((t)&0xf))

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

#define ELF32_R_SYM(i)      ((i)>>8)
#define ELF32_R_TYPE(i)     ((unsignedchar)(i))
#define ELF32_R_INFO(s,t)   (((s)<<8)+(unsignedchar)(t))


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
