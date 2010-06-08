/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- compact_unwind_encoding.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
 

#ifndef __COMPACT_UNWIND_ENCODING__
#define __COMPACT_UNWIND_ENCODING__

#include <stdint.h>

namespace lldb_private {

// 
// Each final linked mach-o image has an optional __TEXT, __unwind_info section.
// This section is much smaller and faster to use than the __eh_frame section.
//



//
// Compilers usually emit standard Dwarf FDEs.  The linker recognizes standard FDEs and
// synthesizes a matching compact_unwind_encoding_t and adds it to the __unwind_info table.
// It is also possible for the compiler to emit __unwind_info entries for functions that 
// have different unwind requirements at different ranges in the function.
//
typedef uint32_t compact_unwind_encoding_t;



//
// The __unwind_info section is laid out for an efficient two level lookup.
// The header of the section contains a coarse index that maps function address
// to the page (4096 byte block) containing the unwind info for that function.  
//

#define UNWIND_SECTION_VERSION 1
struct unwind_info_section_header
{
    uint32_t    version;            // UNWIND_SECTION_VERSION
    uint32_t    commonEncodingsArraySectionOffset;
    uint32_t    commonEncodingsArrayCount;
    uint32_t    personalityArraySectionOffset;
    uint32_t    personalityArrayCount;
    uint32_t    indexSectionOffset;
    uint32_t    indexCount;
    // compact_unwind_encoding_t[]
    // uintptr_t personalities[]
    // unwind_info_section_header_index_entry[]
    // unwind_info_section_header_lsda_index_entry[]
};

struct unwind_info_section_header_index_entry 
{
    uint32_t        functionOffset;
    uint32_t        secondLevelPagesSectionOffset;  // section offset to start of regular or compress page
    uint32_t        lsdaIndexArraySectionOffset;    // section offset to start of lsda_index array for this range
};

struct unwind_info_section_header_lsda_index_entry 
{
    uint32_t        functionOffset;
    uint32_t        lsdaOffset;
};

//
// There are two kinds of second level index pages: regular and compressed.
// A compressed page can hold up to 1021 entries, but it cannot be used
// if too many different encoding types are used.  The regular page holds
// 511 entries.  
//

struct unwind_info_regular_second_level_entry 
{
    uint32_t                    functionOffset;
    compact_unwind_encoding_t    encoding;
};

#define UNWIND_SECOND_LEVEL_REGULAR 2
struct unwind_info_regular_second_level_page_header
{
    uint32_t    kind;    // UNWIND_SECOND_LEVEL_REGULAR
    uint16_t    entryPageOffset;
    uint16_t    entryCount;
    // entry array
};

#define UNWIND_SECOND_LEVEL_COMPRESSED 3
struct unwind_info_compressed_second_level_page_header
{
    uint32_t    kind;    // UNWIND_SECOND_LEVEL_COMPRESSED
    uint16_t    entryPageOffset;
    uint16_t    entryCount;
    uint16_t    encodingsPageOffset;
    uint16_t    encodingsCount;
    // 32-bit entry array    
    // encodings array
};

#define UNWIND_INFO_COMPRESSED_ENTRY_FUNC_OFFSET(entry)            (entry & 0x00FFFFFF)
#define UNWIND_INFO_COMPRESSED_ENTRY_ENCODING_INDEX(entry)        ((entry >> 24) & 0xFF)



// architecture independent bits
enum {
    UNWIND_IS_NOT_FUNCTION_START           = 0x80000000,
    UNWIND_HAS_LSDA                        = 0x40000000,
    UNWIND_PERSONALITY_MASK                = 0x30000000,
};


// x86_64
//
// 1-bit: start
// 1-bit: has lsda
// 2-bit: personality index
//
// 4-bits: 0=old, 1=rbp based, 2=stack-imm, 3=stack-ind, 4=dwarf
//  rbp based:
//        15-bits (5*3-bits per reg) register permutation
//        8-bits for stack offset
//  frameless:
//        8-bits stack size
//        3-bits stack adjust
//        3-bits register count
//        10-bits register permutation
//
enum {
    UNWIND_X86_64_MODE_MASK                         = 0x0F000000,
    UNWIND_X86_64_MODE_COMPATIBILITY                = 0x00000000,
    UNWIND_X86_64_MODE_RBP_FRAME                    = 0x01000000,
    UNWIND_X86_64_MODE_STACK_IMMD                   = 0x02000000,
    UNWIND_X86_64_MODE_STACK_IND                    = 0x03000000,
    UNWIND_X86_64_MODE_DWARF                        = 0x04000000,
    
    UNWIND_X86_64_RBP_FRAME_REGISTERS               = 0x00007FFF,
    UNWIND_X86_64_RBP_FRAME_OFFSET                  = 0x00FF0000,

    UNWIND_X86_64_FRAMELESS_STACK_SIZE              = 0x00FF0000,
    UNWIND_X86_64_FRAMELESS_STACK_ADJUST            = 0x0000E000,
    UNWIND_X86_64_FRAMELESS_STACK_REG_COUNT         = 0x00001C00,
    UNWIND_X86_64_FRAMELESS_STACK_REG_PERMUTATION   = 0x000003FF,

    UNWIND_X86_64_DWARF_SECTION_OFFSET              = 0x00FFFFFF,
};

enum {
    UNWIND_X86_64_REG_NONE       = 0,
    UNWIND_X86_64_REG_RBX        = 1,
    UNWIND_X86_64_REG_R12        = 2,
    UNWIND_X86_64_REG_R13        = 3,
    UNWIND_X86_64_REG_R14        = 4,
    UNWIND_X86_64_REG_R15        = 5,
    UNWIND_X86_64_REG_RBP        = 6,
};


// x86
//
// 1-bit: start
// 1-bit: has lsda
// 2-bit: personality index
//
// 4-bits: 0=old, 1=ebp based, 2=stack-imm, 3=stack-ind, 4=dwarf
//  ebp based:
//        15-bits (5*3-bits per reg) register permutation
//        8-bits for stack offset
//  frameless:
//        8-bits stack size
//        3-bits stack adjust
//        3-bits register count
//        10-bits register permutation
//
enum {
    UNWIND_X86_MODE_MASK                         = 0x0F000000,
    UNWIND_X86_MODE_COMPATIBILITY                = 0x00000000,
    UNWIND_X86_MODE_EBP_FRAME                    = 0x01000000,
    UNWIND_X86_MODE_STACK_IMMD                   = 0x02000000,
    UNWIND_X86_MODE_STACK_IND                    = 0x03000000,
    UNWIND_X86_MODE_DWARF                        = 0x04000000,
    
    UNWIND_X86_EBP_FRAME_REGISTERS               = 0x00007FFF,
    UNWIND_X86_EBP_FRAME_OFFSET                  = 0x00FF0000,
    
    UNWIND_X86_FRAMELESS_STACK_SIZE              = 0x00FF0000,
    UNWIND_X86_FRAMELESS_STACK_ADJUST            = 0x0000E000,
    UNWIND_X86_FRAMELESS_STACK_REG_COUNT         = 0x00001C00,
    UNWIND_X86_FRAMELESS_STACK_REG_PERMUTATION   = 0x000003FF,
    
    UNWIND_X86_DWARF_SECTION_OFFSET              = 0x00FFFFFF,
};

enum {
    UNWIND_X86_REG_NONE     = 0,
    UNWIND_X86_REG_EBX      = 1,
    UNWIND_X86_REG_ECX      = 2,
    UNWIND_X86_REG_EDX      = 3,
    UNWIND_X86_REG_EDI      = 4,
    UNWIND_X86_REG_ESI      = 5,
    UNWIND_X86_REG_EBP      = 6,
};

}; // namespace lldb_private

#endif

