//===-- ArchSpec.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ArchSpec.h"

#include <stdio.h>

#include <string>

#include "llvm/Support/ELF.h"
#include "llvm/Support/MachO.h"

using namespace lldb;
using namespace lldb_private;

#define ARCH_SPEC_SEPARATOR_CHAR    '-'


//----------------------------------------------------------------------
// A structure that describes all of the information we want to know
// about each architecture.
//----------------------------------------------------------------------
struct ArchDefinition
{
    uint32_t cpu;
    uint32_t sub;
    const char *name;
};


static const char *g_arch_type_strings[] = 
{
    "invalid",
    "mach-o",
    "elf"
};

#define CPU_ANY		(UINT32_MAX)

//----------------------------------------------------------------------
// A table that gets searched linearly for matches. This table is used
// to convert cpu type and subtypes to architecture names, and to
// convert architecture names to cpu types and subtypes. The ordering
// is important and allows the precedence to be set when the table is
// built.
//----------------------------------------------------------------------
static ArchDefinition g_mach_arch_defs[] =
{
    { CPU_ANY,                          CPU_ANY , "all"         },
    { llvm::MachO::CPUTypeARM,          CPU_ANY , "arm"         },
    { llvm::MachO::CPUTypeARM,          0       , "arm"         },
    { llvm::MachO::CPUTypeARM,          5       , "armv4"       },
    { llvm::MachO::CPUTypeARM,          6       , "armv6"       },
    { llvm::MachO::CPUTypeARM,          7       , "armv5"       },
    { llvm::MachO::CPUTypeARM,          8       , "xscale"      },
    { llvm::MachO::CPUTypeARM,          9       , "armv7"       },
    { llvm::MachO::CPUTypePowerPC,      CPU_ANY , "ppc"         },
    { llvm::MachO::CPUTypePowerPC,      0       , "ppc"         },
    { llvm::MachO::CPUTypePowerPC,      1       , "ppc601"      },
    { llvm::MachO::CPUTypePowerPC,      2       , "ppc602"      },
    { llvm::MachO::CPUTypePowerPC,      3       , "ppc603"      },
    { llvm::MachO::CPUTypePowerPC,      4       , "ppc603e"     },
    { llvm::MachO::CPUTypePowerPC,      5       , "ppc603ev"    },
    { llvm::MachO::CPUTypePowerPC,      6       , "ppc604"      },
    { llvm::MachO::CPUTypePowerPC,      7       , "ppc604e"     },
    { llvm::MachO::CPUTypePowerPC,      8       , "ppc620"      },
    { llvm::MachO::CPUTypePowerPC,      9       , "ppc750"      },
    { llvm::MachO::CPUTypePowerPC,      10      , "ppc7400"     },
    { llvm::MachO::CPUTypePowerPC,      11      , "ppc7450"     },
    { llvm::MachO::CPUTypePowerPC,      100     , "ppc970"      },
    { llvm::MachO::CPUTypePowerPC64,    0       , "ppc64"       },
    { llvm::MachO::CPUTypePowerPC64,    100     , "ppc970-64"   },
    { llvm::MachO::CPUTypeI386,         3       , "i386"        },
    { llvm::MachO::CPUTypeI386,         4       , "i486"        },
    { llvm::MachO::CPUTypeI386,         0x84    , "i486sx"      },
    { llvm::MachO::CPUTypeI386,         CPU_ANY , "i386"        },
    { llvm::MachO::CPUTypeX86_64,       3       , "x86_64"      },
    { llvm::MachO::CPUTypeX86_64,       CPU_ANY , "x86_64"      },

    // TODO: when we get a platform that knows more about the host OS we should
    // let it call some accessor funcitons to set the default system arch for
    // the default, 32 and 64 bit cases instead of hard coding it in this
    // table.

#if defined (__i386__) || defined(__x86_64__)
    { llvm::MachO::CPUTypeX86_64,      3    , LLDB_ARCH_DEFAULT         },
    { llvm::MachO::CPUTypeI386,        3    , LLDB_ARCH_DEFAULT_32BIT   },
    { llvm::MachO::CPUTypeX86_64,      3    , LLDB_ARCH_DEFAULT_64BIT   },
#elif defined (__arm__)
    { llvm::MachO::CPUTypeARM,         6    , LLDB_ARCH_DEFAULT         },
    { llvm::MachO::CPUTypeARM,         6    , LLDB_ARCH_DEFAULT_32BIT   },
#elif defined (__powerpc__) || defined (__ppc__) || defined (__ppc64__)
    { llvm::MachO::CPUTypePowerPC,     10   , LLDB_ARCH_DEFAULT         },
    { llvm::MachO::CPUTypePowerPC,     10   , LLDB_ARCH_DEFAULT_32BIT   },
    { llvm::MachO::CPUTypePowerPC64,   100  , LLDB_ARCH_DEFAULT_64BIT   },
#endif
};

//----------------------------------------------------------------------
// Figure out how many architecture definitions we have
//----------------------------------------------------------------------
const size_t k_num_mach_arch_defs = sizeof(g_mach_arch_defs)/sizeof(ArchDefinition);



//----------------------------------------------------------------------
// A table that gets searched linearly for matches. This table is used
// to convert cpu type and subtypes to architecture names, and to
// convert architecture names to cpu types and subtypes. The ordering
// is important and allows the precedence to be set when the table is
// built.
//----------------------------------------------------------------------
static ArchDefinition g_elf_arch_defs[] =
{
    { llvm::ELF::EM_M32    , 0, "m32"      }, // AT&T WE 32100
    { llvm::ELF::EM_SPARC  , 0, "sparc"    }, // AT&T WE 32100
    { llvm::ELF::EM_386    , 0, "i386"     }, // Intel 80386
    { llvm::ELF::EM_68K    , 0, "68k"      }, // Motorola 68000
    { llvm::ELF::EM_88K    , 0, "88k"      }, // Motorola 88000
    { llvm::ELF::EM_486    , 0, "i486"     }, // Intel 486 (deprecated)
    { llvm::ELF::EM_860    , 0, "860"      }, // Intel 80860
    { llvm::ELF::EM_MIPS   , 0, "rs3000"   }, // MIPS RS3000
    { llvm::ELF::EM_PPC    , 0, "ppc"      }, // PowerPC
    { 21                   , 0, "ppc64"    }, // PowerPC64
    { llvm::ELF::EM_ARM    , 0, "arm"      }, // ARM
    { llvm::ELF::EM_ALPHA  , 0, "alpha"    }, // DEC Alpha
    { llvm::ELF::EM_SPARCV9, 0, "sparc9"   }, // SPARC V9
    { llvm::ELF::EM_X86_64 , 0, "x86_64"   }, // AMD64

#if defined (__i386__) || defined(__x86_64__)
    { llvm::ELF::EM_X86_64 , 0, LLDB_ARCH_DEFAULT         },
    { llvm::ELF::EM_386    , 0, LLDB_ARCH_DEFAULT_32BIT   },
    { llvm::ELF::EM_X86_64 , 0, LLDB_ARCH_DEFAULT_64BIT   },
#elif defined (__arm__)
    { llvm::ELF::EM_ARM    , 0, LLDB_ARCH_DEFAULT         },
    { llvm::ELF::EM_ARM    , 0, LLDB_ARCH_DEFAULT_32BIT   },
#elif defined (__powerpc__) || defined (__ppc__) || defined (__ppc64__)
    { llvm::ELF::EM_PPC    , 0, LLDB_ARCH_DEFAULT         },
    { llvm::ELF::EM_PPC    , 0, LLDB_ARCH_DEFAULT_32BIT   },
    { llvm::ELF::EM_PPC64  , 0, LLDB_ARCH_DEFAULT_64BIT   },
#endif
};

//----------------------------------------------------------------------
// Figure out how many architecture definitions we have
//----------------------------------------------------------------------
const size_t k_num_elf_arch_defs = sizeof(g_elf_arch_defs)/sizeof(ArchDefinition);

//----------------------------------------------------------------------
// Default constructor
//----------------------------------------------------------------------
ArchSpec::ArchSpec() :
    m_type (eArchTypeMachO),        // Use the most complete arch definition which will always be translatable to any other ArchitectureType values
    m_cpu (LLDB_INVALID_CPUTYPE),
    m_sub (0)
{
}

//----------------------------------------------------------------------
// Constructor that initializes the object with supplied cpu and
// subtypes.
//----------------------------------------------------------------------
ArchSpec::ArchSpec (lldb::ArchitectureType arch_type, uint32_t cpu, uint32_t sub) :
    m_type (arch_type),
    m_cpu (cpu),
    m_sub (sub)
{
}

//----------------------------------------------------------------------
// Constructor that initializes the object with supplied
// architecture name. There are also predefined values in
// Defines.h:
//  liblldb_ARCH_DEFAULT
//      The arch the current system defaults to when a program is
//      launched without any extra attributes or settings.
//
//  liblldb_ARCH_DEFAULT_32BIT
//      The 32 bit arch the current system defaults to (if any)
//
//  liblldb_ARCH_DEFAULT_32BIT
//      The 64 bit arch the current system defaults to (if any)
//----------------------------------------------------------------------
ArchSpec::ArchSpec (const char *arch_name) :
    m_type (eArchTypeMachO),        // Use the most complete arch definition which will always be translatable to any other ArchitectureType values
    m_cpu (LLDB_INVALID_CPUTYPE),
    m_sub (0)
{
    if (arch_name)
        SetArch (arch_name);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ArchSpec::~ArchSpec()
{
}

//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------
const ArchSpec&
ArchSpec::operator= (const ArchSpec& rhs)
{
    if (this != &rhs)
    {
        m_type = rhs.m_type;
        m_cpu = rhs.m_cpu;
        m_sub = rhs.m_sub;
    }
    return *this;
}

//----------------------------------------------------------------------
// Get a C string representation of the current architecture
//----------------------------------------------------------------------
const char *
ArchSpec::AsCString() const
{
    return ArchSpec::AsCString(m_type, m_cpu, m_sub);
}

//----------------------------------------------------------------------
// Class function to get a C string representation given a CPU type
// and subtype.
//----------------------------------------------------------------------
const char *
ArchSpec::AsCString (lldb::ArchitectureType arch_type, uint32_t cpu, uint32_t sub)
{
    if (arch_type >= kNumArchTypes)
        return NULL;

    switch (arch_type)
    {
    case kNumArchTypes:
    case eArchTypeInvalid:
        break;

    case eArchTypeMachO:
        for (uint32_t i=0; i<k_num_mach_arch_defs; i++)
        {
            if (cpu == g_mach_arch_defs[i].cpu)
            {
                if (sub == g_mach_arch_defs[i].sub)
                    return g_mach_arch_defs[i].name;
                else if (sub != CPU_ANY && sub != LLDB_INVALID_CPUTYPE)
                {
                    if ((sub & 0x00ffffff) == g_mach_arch_defs[i].sub)
                        return g_mach_arch_defs[i].name;
                }
            }
        }
        break;
    
    case eArchTypeELF:
        for (uint32_t i=0; i<k_num_elf_arch_defs; i++)
        {
            if (cpu == g_elf_arch_defs[i].cpu)
            {
                if (sub == g_elf_arch_defs[i].sub)
                    return g_elf_arch_defs[i].name;
            }
        }
        break;
    }

    const char *arch_type_cstr = g_arch_type_strings[arch_type];

    static char s_cpu_hex_str[128];
    ::snprintf(s_cpu_hex_str, 
               sizeof(s_cpu_hex_str), 
               "%s%c%u%c%u", 
               arch_type_cstr,
               ARCH_SPEC_SEPARATOR_CHAR, 
               cpu, 
               ARCH_SPEC_SEPARATOR_CHAR, 
               sub);
    return s_cpu_hex_str;
}

//----------------------------------------------------------------------
// Clears the object contents back to a default invalid state.
//----------------------------------------------------------------------
void
ArchSpec::Clear()
{
    m_type = eArchTypeInvalid;
    m_cpu = LLDB_INVALID_CPUTYPE;
    m_sub = 0;
}




//----------------------------------------------------------------------
// CPU subtype get accessor.
//----------------------------------------------------------------------
uint32_t
ArchSpec::GetCPUSubtype() const
{
    if (m_type == eArchTypeMachO)
    {
        if (m_sub == CPU_ANY || m_sub == LLDB_INVALID_CPUTYPE)
            return m_sub;
        return m_sub & 0xffffff;
    }
    return 0;
}


//----------------------------------------------------------------------
// CPU type get accessor.
//----------------------------------------------------------------------
uint32_t
ArchSpec::GetCPUType() const
{
    return m_cpu;
}

//----------------------------------------------------------------------
// This function is designed to abstract us from having to know any
// details about the current m_type, m_cpu, and m_sub values and 
// translate the result into a generic CPU type so LLDB core code can
// detect any CPUs that it supports.
//----------------------------------------------------------------------
ArchSpec::CPU
ArchSpec::GetGenericCPUType () const
{
    switch (m_type)
    {
    case kNumArchTypes:
    case eArchTypeInvalid:
        break;

    case eArchTypeMachO:
        switch (m_cpu)
        {
        case llvm::MachO::CPUTypeARM:       return eCPU_arm;
        case llvm::MachO::CPUTypeI386:      return eCPU_i386;
        case llvm::MachO::CPUTypeX86_64:    return eCPU_x86_64;
        case llvm::MachO::CPUTypePowerPC:   return eCPU_ppc;
        case llvm::MachO::CPUTypePowerPC64: return eCPU_ppc64;
        case llvm::MachO::CPUTypeSPARC:     return eCPU_sparc;
        }
        break;
    
    case eArchTypeELF:
        switch (m_cpu)
        {
        case llvm::ELF::EM_ARM:     return eCPU_arm;
        case llvm::ELF::EM_386:     return eCPU_i386;
        case llvm::ELF::EM_X86_64:  return eCPU_x86_64;
        case llvm::ELF::EM_PPC:     return eCPU_ppc;
        case 21:                    return eCPU_ppc64;
        case llvm::ELF::EM_SPARC: 	return eCPU_sparc;
        }
        break;
    }

    return eCPU_Unknown;
}




//----------------------------------------------------------------------
// Feature flags get accessor.
//----------------------------------------------------------------------
uint32_t
ArchSpec::GetFeatureFlags() const
{
    if (m_type == eArchTypeMachO)
    {
        if (m_sub == CPU_ANY || m_sub == LLDB_INVALID_CPUTYPE)
            return 0;
        return m_sub & 0xff000000;
    }
    return 0;
}


static const char * g_i386_dwarf_reg_names[] =
{
    "eax",
    "ecx",
    "edx",
    "ebx",
    "esp",
    "ebp",
    "esi",
    "edi",
    "eip",
    "eflags"
};

static const char * g_i386_gcc_reg_names[] =
{
    "eax",
    "ecx",
    "edx",
    "ebx",
    "ebp",
    "esp",
    "esi",
    "edi",
    "eip",
    "eflags"
};

static const char * g_x86_64_dwarf_and_gcc_reg_names[] = {
    "rax",
    "rdx",
    "rcx",
    "rbx",
    "rsi",
    "rdi",
    "rbp",
    "rsp",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
    "rip"
};

// Values take from:
// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0040a/IHI0040A_aadwarf.pdf

enum
{
    eRegNumARM_DWARF_r0         = 0,
    eRegNumARM_DWARF_r1         = 1,
    eRegNumARM_DWARF_r2         = 2,
    eRegNumARM_DWARF_r3         = 3,
    eRegNumARM_DWARF_r4         = 4,
    eRegNumARM_DWARF_r5         = 5,
    eRegNumARM_DWARF_r6         = 6,
    eRegNumARM_DWARF_r7         = 7,
    eRegNumARM_DWARF_r8         = 8,
    eRegNumARM_DWARF_r9         = 9,
    eRegNumARM_DWARF_r10        = 10,
    eRegNumARM_DWARF_r11        = 11,
    eRegNumARM_DWARF_r12        = 12,
    eRegNumARM_DWARF_r13        = 13,   // SP
    eRegNumARM_DWARF_r14        = 14,   // LR
    eRegNumARM_DWARF_r15        = 15,   // PC

    eRegNumARM_DWARF_f0_obsolete= 16,
    eRegNumARM_DWARF_f1_obsolete,
    eRegNumARM_DWARF_f2_obsolete,
    eRegNumARM_DWARF_f3_obsolete,
    eRegNumARM_DWARF_f4_obsolete,
    eRegNumARM_DWARF_f5_obsolete,
    eRegNumARM_DWARF_f6_obsolete,
    eRegNumARM_DWARF_f7_obsolete,

    eRegNumARM_DWARF_s0_obsolete = 16,
    eRegNumARM_DWARF_s1_obsolete,
    eRegNumARM_DWARF_s2_obsolete,
    eRegNumARM_DWARF_s3_obsolete,
    eRegNumARM_DWARF_s4_obsolete,
    eRegNumARM_DWARF_s5_obsolete,
    eRegNumARM_DWARF_s6_obsolete,
    eRegNumARM_DWARF_s7_obsolete,
    eRegNumARM_DWARF_s8_obsolete,
    eRegNumARM_DWARF_s9_obsolete,
    eRegNumARM_DWARF_s10_obsolete,
    eRegNumARM_DWARF_s11_obsolete,
    eRegNumARM_DWARF_s12_obsolete,
    eRegNumARM_DWARF_s13_obsolete,
    eRegNumARM_DWARF_s14_obsolete,
    eRegNumARM_DWARF_s15_obsolete,
    eRegNumARM_DWARF_s16_obsolete,
    eRegNumARM_DWARF_s17_obsolete,
    eRegNumARM_DWARF_s18_obsolete,
    eRegNumARM_DWARF_s19_obsolete,
    eRegNumARM_DWARF_s20_obsolete,
    eRegNumARM_DWARF_s21_obsolete,
    eRegNumARM_DWARF_s22_obsolete,
    eRegNumARM_DWARF_s23_obsolete,
    eRegNumARM_DWARF_s24_obsolete,
    eRegNumARM_DWARF_s25_obsolete,
    eRegNumARM_DWARF_s26_obsolete,
    eRegNumARM_DWARF_s27_obsolete,
    eRegNumARM_DWARF_s28_obsolete,
    eRegNumARM_DWARF_s29_obsolete,
    eRegNumARM_DWARF_s30_obsolete,
    eRegNumARM_DWARF_s31_obsolete,

    eRegNumARM_DWARF_s0 = 64,
    eRegNumARM_DWARF_s1,
    eRegNumARM_DWARF_s2,
    eRegNumARM_DWARF_s3,
    eRegNumARM_DWARF_s4,
    eRegNumARM_DWARF_s5,
    eRegNumARM_DWARF_s6,
    eRegNumARM_DWARF_s7,
    eRegNumARM_DWARF_s8,
    eRegNumARM_DWARF_s9,
    eRegNumARM_DWARF_s10,
    eRegNumARM_DWARF_s11,
    eRegNumARM_DWARF_s12,
    eRegNumARM_DWARF_s13,
    eRegNumARM_DWARF_s14,
    eRegNumARM_DWARF_s15,
    eRegNumARM_DWARF_s16,
    eRegNumARM_DWARF_s17,
    eRegNumARM_DWARF_s18,
    eRegNumARM_DWARF_s19,
    eRegNumARM_DWARF_s20,
    eRegNumARM_DWARF_s21,
    eRegNumARM_DWARF_s22,
    eRegNumARM_DWARF_s23,
    eRegNumARM_DWARF_s24,
    eRegNumARM_DWARF_s25,
    eRegNumARM_DWARF_s26,
    eRegNumARM_DWARF_s27,
    eRegNumARM_DWARF_s28,
    eRegNumARM_DWARF_s29,
    eRegNumARM_DWARF_s30,
    eRegNumARM_DWARF_s31,

    eRegNumARM_DWARF_f0         = 96,
    eRegNumARM_DWARF_f1,
    eRegNumARM_DWARF_f2,
    eRegNumARM_DWARF_f3,
    eRegNumARM_DWARF_f4,
    eRegNumARM_DWARF_f5,
    eRegNumARM_DWARF_f6,
    eRegNumARM_DWARF_f7,

    eRegNumARM_DWARF_ACC0       = 104,
    eRegNumARM_DWARF_ACC1,
    eRegNumARM_DWARF_ACC2,
    eRegNumARM_DWARF_ACC3,
    eRegNumARM_DWARF_ACC4,
    eRegNumARM_DWARF_ACC5,
    eRegNumARM_DWARF_ACC6,
    eRegNumARM_DWARF_ACC7,

    eRegNumARM_DWARF_wCGR0      = 104,  // These overlap with ACC0-ACC7
    eRegNumARM_DWARF_wCGR1,
    eRegNumARM_DWARF_wCGR2,
    eRegNumARM_DWARF_wCGR3,
    eRegNumARM_DWARF_wCGR4,
    eRegNumARM_DWARF_wCGR5,
    eRegNumARM_DWARF_wCGR6,
    eRegNumARM_DWARF_wCGR7,

    eRegNumARM_DWARF_wR0        = 112,
    eRegNumARM_DWARF_wR1,
    eRegNumARM_DWARF_wR2,
    eRegNumARM_DWARF_wR3,
    eRegNumARM_DWARF_wR4,
    eRegNumARM_DWARF_wR5,
    eRegNumARM_DWARF_wR6,
    eRegNumARM_DWARF_wR7,
    eRegNumARM_DWARF_wR8,
    eRegNumARM_DWARF_wR9,
    eRegNumARM_DWARF_wR10,
    eRegNumARM_DWARF_wR11,
    eRegNumARM_DWARF_wR12,
    eRegNumARM_DWARF_wR13,
    eRegNumARM_DWARF_wR14,
    eRegNumARM_DWARF_wR15,

    eRegNumARM_DWARF_spsr       = 128,
    eRegNumARM_DWARF_spsr_fiq,
    eRegNumARM_DWARF_spsr_irq,
    eRegNumARM_DWARF_spsr_abt,
    eRegNumARM_DWARF_spsr_und,
    eRegNumARM_DWARF_spsr_svc,

    eRegNumARM_DWARF_r8_usr     = 144,
    eRegNumARM_DWARF_r9_usr,
    eRegNumARM_DWARF_r10_usr,
    eRegNumARM_DWARF_r11_usr,
    eRegNumARM_DWARF_r12_usr,
    eRegNumARM_DWARF_r13_usr,
    eRegNumARM_DWARF_r14_usr,

    eRegNumARM_DWARF_r8_fiq     = 151,
    eRegNumARM_DWARF_r9_fiq,
    eRegNumARM_DWARF_r10_fiq,
    eRegNumARM_DWARF_r11_fiq,
    eRegNumARM_DWARF_r12_fiq,
    eRegNumARM_DWARF_r13_fiq,
    eRegNumARM_DWARF_r14_fiq,

    eRegNumARM_DWARF_r13_irq,
    eRegNumARM_DWARF_r14_irq,

    eRegNumARM_DWARF_r13_abt,
    eRegNumARM_DWARF_r14_abt,

    eRegNumARM_DWARF_r13_und,
    eRegNumARM_DWARF_r14_und,

    eRegNumARM_DWARF_r13_svc,
    eRegNumARM_DWARF_r14_svc,

    eRegNumARM_DWARF_wC0        = 192,
    eRegNumARM_DWARF_wC1,
    eRegNumARM_DWARF_wC2,
    eRegNumARM_DWARF_wC3,
    eRegNumARM_DWARF_wC4,
    eRegNumARM_DWARF_wC5,
    eRegNumARM_DWARF_wC6,
    eRegNumARM_DWARF_wC7,

    eRegNumARM_DWARF_d0         = 256,  // VFP-v3/NEON D0-D31 (32 64 bit registers)
    eRegNumARM_DWARF_d1,
    eRegNumARM_DWARF_d2,
    eRegNumARM_DWARF_d3,
    eRegNumARM_DWARF_d4,
    eRegNumARM_DWARF_d5,
    eRegNumARM_DWARF_d6,
    eRegNumARM_DWARF_d7,
    eRegNumARM_DWARF_d8,
    eRegNumARM_DWARF_d9,
    eRegNumARM_DWARF_d10,
    eRegNumARM_DWARF_d11,
    eRegNumARM_DWARF_d12,
    eRegNumARM_DWARF_d13,
    eRegNumARM_DWARF_d14,
    eRegNumARM_DWARF_d15,
    eRegNumARM_DWARF_d16,
    eRegNumARM_DWARF_d17,
    eRegNumARM_DWARF_d18,
    eRegNumARM_DWARF_d19,
    eRegNumARM_DWARF_d20,
    eRegNumARM_DWARF_d21,
    eRegNumARM_DWARF_d22,
    eRegNumARM_DWARF_d23,
    eRegNumARM_DWARF_d24,
    eRegNumARM_DWARF_d25,
    eRegNumARM_DWARF_d26,
    eRegNumARM_DWARF_d27,
    eRegNumARM_DWARF_d28,
    eRegNumARM_DWARF_d29,
    eRegNumARM_DWARF_d30,
    eRegNumARM_DWARF_d31
};

// Register numbering definitions for 32 and 64 bit ppc for RegisterNumberingType::Dwarf
enum
{
    eRegNumPPC_DWARF_r0         = 0,
    eRegNumPPC_DWARF_r1         = 1,
    eRegNumPPC_DWARF_r2         = 2,
    eRegNumPPC_DWARF_r3         = 3,
    eRegNumPPC_DWARF_r4         = 4,
    eRegNumPPC_DWARF_r5         = 5,
    eRegNumPPC_DWARF_r6         = 6,
    eRegNumPPC_DWARF_r7         = 7,
    eRegNumPPC_DWARF_r8         = 8,
    eRegNumPPC_DWARF_r9         = 9,
    eRegNumPPC_DWARF_r10        = 10,
    eRegNumPPC_DWARF_r11        = 11,
    eRegNumPPC_DWARF_r12        = 12,
    eRegNumPPC_DWARF_r13        = 13,
    eRegNumPPC_DWARF_r14        = 14,
    eRegNumPPC_DWARF_r15        = 15,
    eRegNumPPC_DWARF_r16        = 16,
    eRegNumPPC_DWARF_r17        = 17,
    eRegNumPPC_DWARF_r18        = 18,
    eRegNumPPC_DWARF_r19        = 19,
    eRegNumPPC_DWARF_r20        = 20,
    eRegNumPPC_DWARF_r21        = 21,
    eRegNumPPC_DWARF_r22        = 22,
    eRegNumPPC_DWARF_r23        = 23,
    eRegNumPPC_DWARF_r24        = 24,
    eRegNumPPC_DWARF_r25        = 25,
    eRegNumPPC_DWARF_r26        = 26,
    eRegNumPPC_DWARF_r27        = 27,
    eRegNumPPC_DWARF_r28        = 28,
    eRegNumPPC_DWARF_r29        = 29,
    eRegNumPPC_DWARF_r30        = 30,
    eRegNumPPC_DWARF_r31        = 31,

    eRegNumPPC_DWARF_fr0        = 32,
    eRegNumPPC_DWARF_fr1        = 33,
    eRegNumPPC_DWARF_fr2        = 34,
    eRegNumPPC_DWARF_fr3        = 35,
    eRegNumPPC_DWARF_fr4        = 36,
    eRegNumPPC_DWARF_fr5        = 37,
    eRegNumPPC_DWARF_fr6        = 38,
    eRegNumPPC_DWARF_fr7        = 39,
    eRegNumPPC_DWARF_fr8        = 40,
    eRegNumPPC_DWARF_fr9        = 41,
    eRegNumPPC_DWARF_fr10       = 42,
    eRegNumPPC_DWARF_fr11       = 43,
    eRegNumPPC_DWARF_fr12       = 44,
    eRegNumPPC_DWARF_fr13       = 45,
    eRegNumPPC_DWARF_fr14       = 46,
    eRegNumPPC_DWARF_fr15       = 47,
    eRegNumPPC_DWARF_fr16       = 48,
    eRegNumPPC_DWARF_fr17       = 49,
    eRegNumPPC_DWARF_fr18       = 50,
    eRegNumPPC_DWARF_fr19       = 51,
    eRegNumPPC_DWARF_fr20       = 52,
    eRegNumPPC_DWARF_fr21       = 53,
    eRegNumPPC_DWARF_fr22       = 54,
    eRegNumPPC_DWARF_fr23       = 55,
    eRegNumPPC_DWARF_fr24       = 56,
    eRegNumPPC_DWARF_fr25       = 57,
    eRegNumPPC_DWARF_fr26       = 58,
    eRegNumPPC_DWARF_fr27       = 59,
    eRegNumPPC_DWARF_fr28       = 60,
    eRegNumPPC_DWARF_fr29       = 61,
    eRegNumPPC_DWARF_fr30       = 62,
    eRegNumPPC_DWARF_fr31       = 63,

    eRegNumPPC_DWARF_cr         = 64,
    eRegNumPPC_DWARF_fpscr      = 65,
    eRegNumPPC_DWARF_msr        = 66,
    eRegNumPPC_DWARF_vscr       = 67,

    eRegNumPPC_DWARF_sr0        = 70,
    eRegNumPPC_DWARF_sr1,
    eRegNumPPC_DWARF_sr2,
    eRegNumPPC_DWARF_sr3,
    eRegNumPPC_DWARF_sr4,
    eRegNumPPC_DWARF_sr5,
    eRegNumPPC_DWARF_sr6,
    eRegNumPPC_DWARF_sr7,
    eRegNumPPC_DWARF_sr8,
    eRegNumPPC_DWARF_sr9,
    eRegNumPPC_DWARF_sr10,
    eRegNumPPC_DWARF_sr11,
    eRegNumPPC_DWARF_sr12,
    eRegNumPPC_DWARF_sr13,
    eRegNumPPC_DWARF_sr14,
    eRegNumPPC_DWARF_sr15,


    eRegNumPPC_DWARF_acc        = 99,
    eRegNumPPC_DWARF_mq         = 100,
    eRegNumPPC_DWARF_xer        = 101,
    eRegNumPPC_DWARF_rtcu       = 104,
    eRegNumPPC_DWARF_rtcl       = 105,

    eRegNumPPC_DWARF_lr         = 108,
    eRegNumPPC_DWARF_ctr        = 109,

    eRegNumPPC_DWARF_dsisr      = 118,
    eRegNumPPC_DWARF_dar        = 119,
    eRegNumPPC_DWARF_dec        = 122,
    eRegNumPPC_DWARF_sdr1       = 125,
    eRegNumPPC_DWARF_srr0       = 126,
    eRegNumPPC_DWARF_srr1       = 127,

    eRegNumPPC_DWARF_vrsave     = 356,
    eRegNumPPC_DWARF_sprg0      = 372,
    eRegNumPPC_DWARF_sprg1,
    eRegNumPPC_DWARF_sprg2,
    eRegNumPPC_DWARF_sprg3,

    eRegNumPPC_DWARF_asr        = 380,
    eRegNumPPC_DWARF_ear        = 382,
    eRegNumPPC_DWARF_tb         = 384,
    eRegNumPPC_DWARF_tbu        = 385,
    eRegNumPPC_DWARF_pvr        = 387,

    eRegNumPPC_DWARF_spefscr    = 612,

    eRegNumPPC_DWARF_ibat0u     = 628,
    eRegNumPPC_DWARF_ibat0l     = 629,
    eRegNumPPC_DWARF_ibat1u     = 630,
    eRegNumPPC_DWARF_ibat1l     = 631,
    eRegNumPPC_DWARF_ibat2u     = 632,
    eRegNumPPC_DWARF_ibat2l     = 633,
    eRegNumPPC_DWARF_ibat3u     = 634,
    eRegNumPPC_DWARF_ibat3l     = 635,
    eRegNumPPC_DWARF_dbat0u     = 636,
    eRegNumPPC_DWARF_dbat0l     = 637,
    eRegNumPPC_DWARF_dbat1u     = 638,
    eRegNumPPC_DWARF_dbat1l     = 639,
    eRegNumPPC_DWARF_dbat2u     = 640,
    eRegNumPPC_DWARF_dbat2l     = 641,
    eRegNumPPC_DWARF_dbat3u     = 642,
    eRegNumPPC_DWARF_dbat3l     = 643,

    eRegNumPPC_DWARF_hid0       = 1108,
    eRegNumPPC_DWARF_hid1,
    eRegNumPPC_DWARF_hid2,
    eRegNumPPC_DWARF_hid3,
    eRegNumPPC_DWARF_hid4,
    eRegNumPPC_DWARF_hid5,
    eRegNumPPC_DWARF_hid6,
    eRegNumPPC_DWARF_hid7,
    eRegNumPPC_DWARF_hid8,
    eRegNumPPC_DWARF_hid9,
    eRegNumPPC_DWARF_hid10,
    eRegNumPPC_DWARF_hid11,
    eRegNumPPC_DWARF_hid12,
    eRegNumPPC_DWARF_hid13,
    eRegNumPPC_DWARF_hid14,
    eRegNumPPC_DWARF_hid15,

    eRegNumPPC_DWARF_vr0        = 1124,
    eRegNumPPC_DWARF_vr1,
    eRegNumPPC_DWARF_vr2,
    eRegNumPPC_DWARF_vr3,
    eRegNumPPC_DWARF_vr4,
    eRegNumPPC_DWARF_vr5,
    eRegNumPPC_DWARF_vr6,
    eRegNumPPC_DWARF_vr7,
    eRegNumPPC_DWARF_vr8,
    eRegNumPPC_DWARF_vr9,
    eRegNumPPC_DWARF_vr10,
    eRegNumPPC_DWARF_vr11,
    eRegNumPPC_DWARF_vr12,
    eRegNumPPC_DWARF_vr13,
    eRegNumPPC_DWARF_vr14,
    eRegNumPPC_DWARF_vr15,
    eRegNumPPC_DWARF_vr16,
    eRegNumPPC_DWARF_vr17,
    eRegNumPPC_DWARF_vr18,
    eRegNumPPC_DWARF_vr19,
    eRegNumPPC_DWARF_vr20,
    eRegNumPPC_DWARF_vr21,
    eRegNumPPC_DWARF_vr22,
    eRegNumPPC_DWARF_vr23,
    eRegNumPPC_DWARF_vr24,
    eRegNumPPC_DWARF_vr25,
    eRegNumPPC_DWARF_vr26,
    eRegNumPPC_DWARF_vr27,
    eRegNumPPC_DWARF_vr28,
    eRegNumPPC_DWARF_vr29,
    eRegNumPPC_DWARF_vr30,
    eRegNumPPC_DWARF_vr31,

    eRegNumPPC_DWARF_ev0        = 1200,
    eRegNumPPC_DWARF_ev1,
    eRegNumPPC_DWARF_ev2,
    eRegNumPPC_DWARF_ev3,
    eRegNumPPC_DWARF_ev4,
    eRegNumPPC_DWARF_ev5,
    eRegNumPPC_DWARF_ev6,
    eRegNumPPC_DWARF_ev7,
    eRegNumPPC_DWARF_ev8,
    eRegNumPPC_DWARF_ev9,
    eRegNumPPC_DWARF_ev10,
    eRegNumPPC_DWARF_ev11,
    eRegNumPPC_DWARF_ev12,
    eRegNumPPC_DWARF_ev13,
    eRegNumPPC_DWARF_ev14,
    eRegNumPPC_DWARF_ev15,
    eRegNumPPC_DWARF_ev16,
    eRegNumPPC_DWARF_ev17,
    eRegNumPPC_DWARF_ev18,
    eRegNumPPC_DWARF_ev19,
    eRegNumPPC_DWARF_ev20,
    eRegNumPPC_DWARF_ev21,
    eRegNumPPC_DWARF_ev22,
    eRegNumPPC_DWARF_ev23,
    eRegNumPPC_DWARF_ev24,
    eRegNumPPC_DWARF_ev25,
    eRegNumPPC_DWARF_ev26,
    eRegNumPPC_DWARF_ev27,
    eRegNumPPC_DWARF_ev28,
    eRegNumPPC_DWARF_ev29,
    eRegNumPPC_DWARF_ev30,
    eRegNumPPC_DWARF_ev31
};

// Register numbering definitions for 32 and 64 bit ppc for RegisterNumberingType::GCC
enum
{
    eRegNumPPC_GCC_r0       = 0,
    eRegNumPPC_GCC_r1       = 1,
    eRegNumPPC_GCC_r2       = 2,
    eRegNumPPC_GCC_r3       = 3,
    eRegNumPPC_GCC_r4       = 4,
    eRegNumPPC_GCC_r5       = 5,
    eRegNumPPC_GCC_r6       = 6,
    eRegNumPPC_GCC_r7       = 7,
    eRegNumPPC_GCC_r8       = 8,
    eRegNumPPC_GCC_r9       = 9,
    eRegNumPPC_GCC_r10      = 10,
    eRegNumPPC_GCC_r11      = 11,
    eRegNumPPC_GCC_r12      = 12,
    eRegNumPPC_GCC_r13      = 13,
    eRegNumPPC_GCC_r14      = 14,
    eRegNumPPC_GCC_r15      = 15,
    eRegNumPPC_GCC_r16      = 16,
    eRegNumPPC_GCC_r17      = 17,
    eRegNumPPC_GCC_r18      = 18,
    eRegNumPPC_GCC_r19      = 19,
    eRegNumPPC_GCC_r20      = 20,
    eRegNumPPC_GCC_r21      = 21,
    eRegNumPPC_GCC_r22      = 22,
    eRegNumPPC_GCC_r23      = 23,
    eRegNumPPC_GCC_r24      = 24,
    eRegNumPPC_GCC_r25      = 25,
    eRegNumPPC_GCC_r26      = 26,
    eRegNumPPC_GCC_r27      = 27,
    eRegNumPPC_GCC_r28      = 28,
    eRegNumPPC_GCC_r29      = 29,
    eRegNumPPC_GCC_r30      = 30,
    eRegNumPPC_GCC_r31      = 31,
    eRegNumPPC_GCC_fr0      = 32,
    eRegNumPPC_GCC_fr1      = 33,
    eRegNumPPC_GCC_fr2      = 34,
    eRegNumPPC_GCC_fr3      = 35,
    eRegNumPPC_GCC_fr4      = 36,
    eRegNumPPC_GCC_fr5      = 37,
    eRegNumPPC_GCC_fr6      = 38,
    eRegNumPPC_GCC_fr7      = 39,
    eRegNumPPC_GCC_fr8      = 40,
    eRegNumPPC_GCC_fr9      = 41,
    eRegNumPPC_GCC_fr10     = 42,
    eRegNumPPC_GCC_fr11     = 43,
    eRegNumPPC_GCC_fr12     = 44,
    eRegNumPPC_GCC_fr13     = 45,
    eRegNumPPC_GCC_fr14     = 46,
    eRegNumPPC_GCC_fr15     = 47,
    eRegNumPPC_GCC_fr16     = 48,
    eRegNumPPC_GCC_fr17     = 49,
    eRegNumPPC_GCC_fr18     = 50,
    eRegNumPPC_GCC_fr19     = 51,
    eRegNumPPC_GCC_fr20     = 52,
    eRegNumPPC_GCC_fr21     = 53,
    eRegNumPPC_GCC_fr22     = 54,
    eRegNumPPC_GCC_fr23     = 55,
    eRegNumPPC_GCC_fr24     = 56,
    eRegNumPPC_GCC_fr25     = 57,
    eRegNumPPC_GCC_fr26     = 58,
    eRegNumPPC_GCC_fr27     = 59,
    eRegNumPPC_GCC_fr28     = 60,
    eRegNumPPC_GCC_fr29     = 61,
    eRegNumPPC_GCC_fr30     = 62,
    eRegNumPPC_GCC_fr31     = 63,
    eRegNumPPC_GCC_mq       = 64,
    eRegNumPPC_GCC_lr       = 65,
    eRegNumPPC_GCC_ctr      = 66,
    eRegNumPPC_GCC_ap       = 67,
    eRegNumPPC_GCC_cr0      = 68,
    eRegNumPPC_GCC_cr1      = 69,
    eRegNumPPC_GCC_cr2      = 70,
    eRegNumPPC_GCC_cr3      = 71,
    eRegNumPPC_GCC_cr4      = 72,
    eRegNumPPC_GCC_cr5      = 73,
    eRegNumPPC_GCC_cr6      = 74,
    eRegNumPPC_GCC_cr7      = 75,
    eRegNumPPC_GCC_xer      = 76,
    eRegNumPPC_GCC_v0       = 77,
    eRegNumPPC_GCC_v1       = 78,
    eRegNumPPC_GCC_v2       = 79,
    eRegNumPPC_GCC_v3       = 80,
    eRegNumPPC_GCC_v4       = 81,
    eRegNumPPC_GCC_v5       = 82,
    eRegNumPPC_GCC_v6       = 83,
    eRegNumPPC_GCC_v7       = 84,
    eRegNumPPC_GCC_v8       = 85,
    eRegNumPPC_GCC_v9       = 86,
    eRegNumPPC_GCC_v10      = 87,
    eRegNumPPC_GCC_v11      = 88,
    eRegNumPPC_GCC_v12      = 89,
    eRegNumPPC_GCC_v13      = 90,
    eRegNumPPC_GCC_v14      = 91,
    eRegNumPPC_GCC_v15      = 92,
    eRegNumPPC_GCC_v16      = 93,
    eRegNumPPC_GCC_v17      = 94,
    eRegNumPPC_GCC_v18      = 95,
    eRegNumPPC_GCC_v19      = 96,
    eRegNumPPC_GCC_v20      = 97,
    eRegNumPPC_GCC_v21      = 98,
    eRegNumPPC_GCC_v22      = 99,
    eRegNumPPC_GCC_v23      = 100,
    eRegNumPPC_GCC_v24      = 101,
    eRegNumPPC_GCC_v25      = 102,
    eRegNumPPC_GCC_v26      = 103,
    eRegNumPPC_GCC_v27      = 104,
    eRegNumPPC_GCC_v28      = 105,
    eRegNumPPC_GCC_v29      = 106,
    eRegNumPPC_GCC_v30      = 107,
    eRegNumPPC_GCC_v31      = 108,
    eRegNumPPC_GCC_vrsave   = 109,
    eRegNumPPC_GCC_vscr     = 110,
    eRegNumPPC_GCC_spe_acc  = 111,
    eRegNumPPC_GCC_spefscr  = 112,
    eRegNumPPC_GCC_sfp      = 113
};

static const char * g_arm_gcc_reg_names[] = {
    "r0",   "r1",   "r2",   "r3",   "r4",   "r5",   "r6",   "r7",
    "r8",   "r9",   "r10",  "r11",  "r12",  "sp",   "lr",   "pc",
    "f0",   "f1",   "f2",   "f3",   "f4",   "f5",   "f6",   "f7",
    "cc",   "sfp",  "afp",
    "mv0",  "mv1",  "mv2",  "mv3",  "mv4",  "mv5",  "mv6",  "mv7",
    "mv8",  "mv9",  "mv10", "mv11", "mv12", "mv13", "mv14", "mv15",
    "wcgr0","wcgr1","wcgr2","wcgr3",
    "wr0",  "wr1",  "wr2",  "wr3",  "wr4",  "wr5",  "wr6",  "wr7",
    "wr8",  "wr9",  "wr10", "wr11", "wr12", "wr13", "wr14", "wr15",
    "s0",   "s1",   "s2",   "s3",   "s4",   "s5",   "s6",   "s7",
    "s8",   "s9",   "s10",  "s11",  "s12",  "s13",  "s14",  "s15",
    "s16",  "s17",  "s18",  "s19",  "s20",  "s21",  "s22",  "s23",
    "s24",  "s25",  "s26",  "s27",  "s28",  "s29",  "s30",  "s31",
    "vfpcc"
};

//----------------------------------------------------------------------
// Get register names for the current object architecture given
// a register number, and a reg_kind for that register number.
//----------------------------------------------------------------------
const char *
ArchSpec::GetRegisterName(uint32_t reg_num, uint32_t reg_kind) const
{
    return ArchSpec::GetRegisterName(m_type, m_cpu, m_sub, reg_num, reg_kind);
}


//----------------------------------------------------------------------
// Get register names for the specified CPU type and subtype given
// a register number, and a reg_kind for that register number.
//----------------------------------------------------------------------
const char *
ArchSpec::GetRegisterName (ArchitectureType arch_type, uint32_t cpu, uint32_t subtype, uint32_t reg_num, uint32_t reg_kind)
{
    if ((arch_type == eArchTypeMachO && cpu == llvm::MachO::CPUTypeI386) ||
        (arch_type == eArchTypeELF   && cpu == llvm::ELF::EM_386))
    {
        switch (reg_kind)
        {
        case eRegisterKindGCC:
            if (reg_num < sizeof(g_i386_gcc_reg_names)/sizeof(const char *))
                return g_i386_gcc_reg_names[reg_num];
            break;
        case eRegisterKindDWARF:
            if (reg_num < sizeof(g_i386_dwarf_reg_names)/sizeof(const char *))
                return g_i386_dwarf_reg_names[reg_num];
            break;
        default:
            break;
        }
    }
    else if ((arch_type == eArchTypeMachO && cpu == llvm::MachO::CPUTypeX86_64) ||
             (arch_type == eArchTypeELF   && cpu == llvm::ELF::EM_X86_64))
    {
        switch (reg_kind)
        {
        case eRegisterKindGCC:
        case eRegisterKindDWARF:
            if (reg_num < sizeof(g_x86_64_dwarf_and_gcc_reg_names)/sizeof(const char *))
                return g_x86_64_dwarf_and_gcc_reg_names[reg_num];
            break;
        default:
            break;
        }
    }
    else if ((arch_type == eArchTypeMachO && cpu == llvm::MachO::CPUTypeARM) ||
             (arch_type == eArchTypeELF   && cpu == llvm::ELF::EM_ARM))
    {
        switch (reg_kind)
        {
        case eRegisterKindGCC:
            if (reg_num < sizeof(g_arm_gcc_reg_names)/sizeof(const char *))
                return g_arm_gcc_reg_names[reg_num];
            break;

        case eRegisterKindDWARF:
            switch (reg_num)
            {
            case eRegNumARM_DWARF_r0:       return "r0";
            case eRegNumARM_DWARF_r1:       return "r1";
            case eRegNumARM_DWARF_r2:       return "r2";
            case eRegNumARM_DWARF_r3:       return "r3";
            case eRegNumARM_DWARF_r4:       return "r4";
            case eRegNumARM_DWARF_r5:       return "r5";
            case eRegNumARM_DWARF_r6:       return "r6";
            case eRegNumARM_DWARF_r7:       return "r7";
            case eRegNumARM_DWARF_r8:       return "r8";
            case eRegNumARM_DWARF_r9:       return "r9";
            case eRegNumARM_DWARF_r10:      return "r10";
            case eRegNumARM_DWARF_r11:      return "r11";
            case eRegNumARM_DWARF_r12:      return "r12";
            case eRegNumARM_DWARF_r13:      return "sp";
            case eRegNumARM_DWARF_r14:      return "lr";
            case eRegNumARM_DWARF_r15:      return "pc";
            case eRegNumARM_DWARF_s0_obsolete:  case eRegNumARM_DWARF_s0:       return "s0";
            case eRegNumARM_DWARF_s1_obsolete:  case eRegNumARM_DWARF_s1:       return "s1";
            case eRegNumARM_DWARF_s2_obsolete:  case eRegNumARM_DWARF_s2:       return "s2";
            case eRegNumARM_DWARF_s3_obsolete:  case eRegNumARM_DWARF_s3:       return "s3";
            case eRegNumARM_DWARF_s4_obsolete:  case eRegNumARM_DWARF_s4:       return "s4";
            case eRegNumARM_DWARF_s5_obsolete:  case eRegNumARM_DWARF_s5:       return "s5";
            case eRegNumARM_DWARF_s6_obsolete:  case eRegNumARM_DWARF_s6:       return "s6";
            case eRegNumARM_DWARF_s7_obsolete:  case eRegNumARM_DWARF_s7:       return "s7";
            case eRegNumARM_DWARF_s8_obsolete:  case eRegNumARM_DWARF_s8:       return "s8";
            case eRegNumARM_DWARF_s9_obsolete:  case eRegNumARM_DWARF_s9:       return "s9";
            case eRegNumARM_DWARF_s10_obsolete: case eRegNumARM_DWARF_s10:      return "s10";
            case eRegNumARM_DWARF_s11_obsolete: case eRegNumARM_DWARF_s11:      return "s11";
            case eRegNumARM_DWARF_s12_obsolete: case eRegNumARM_DWARF_s12:      return "s12";
            case eRegNumARM_DWARF_s13_obsolete: case eRegNumARM_DWARF_s13:      return "s13";
            case eRegNumARM_DWARF_s14_obsolete: case eRegNumARM_DWARF_s14:      return "s14";
            case eRegNumARM_DWARF_s15_obsolete: case eRegNumARM_DWARF_s15:      return "s15";
            case eRegNumARM_DWARF_s16_obsolete: case eRegNumARM_DWARF_s16:      return "s16";
            case eRegNumARM_DWARF_s17_obsolete: case eRegNumARM_DWARF_s17:      return "s17";
            case eRegNumARM_DWARF_s18_obsolete: case eRegNumARM_DWARF_s18:      return "s18";
            case eRegNumARM_DWARF_s19_obsolete: case eRegNumARM_DWARF_s19:      return "s19";
            case eRegNumARM_DWARF_s20_obsolete: case eRegNumARM_DWARF_s20:      return "s20";
            case eRegNumARM_DWARF_s21_obsolete: case eRegNumARM_DWARF_s21:      return "s21";
            case eRegNumARM_DWARF_s22_obsolete: case eRegNumARM_DWARF_s22:      return "s22";
            case eRegNumARM_DWARF_s23_obsolete: case eRegNumARM_DWARF_s23:      return "s23";
            case eRegNumARM_DWARF_s24_obsolete: case eRegNumARM_DWARF_s24:      return "s24";
            case eRegNumARM_DWARF_s25_obsolete: case eRegNumARM_DWARF_s25:      return "s25";
            case eRegNumARM_DWARF_s26_obsolete: case eRegNumARM_DWARF_s26:      return "s26";
            case eRegNumARM_DWARF_s27_obsolete: case eRegNumARM_DWARF_s27:      return "s27";
            case eRegNumARM_DWARF_s28_obsolete: case eRegNumARM_DWARF_s28:      return "s28";
            case eRegNumARM_DWARF_s29_obsolete: case eRegNumARM_DWARF_s29:      return "s29";
            case eRegNumARM_DWARF_s30_obsolete: case eRegNumARM_DWARF_s30:      return "s30";
            case eRegNumARM_DWARF_s31_obsolete: case eRegNumARM_DWARF_s31:      return "s31";
            case eRegNumARM_DWARF_f0:       return "f0";
            case eRegNumARM_DWARF_f1:       return "f1";
            case eRegNumARM_DWARF_f2:       return "f2";
            case eRegNumARM_DWARF_f3:       return "f3";
            case eRegNumARM_DWARF_f4:       return "f4";
            case eRegNumARM_DWARF_f5:       return "f5";
            case eRegNumARM_DWARF_f6:       return "f6";
            case eRegNumARM_DWARF_f7:       return "f7";
            case eRegNumARM_DWARF_wCGR0:    return "wCGR0/ACC0";
            case eRegNumARM_DWARF_wCGR1:    return "wCGR1/ACC1";
            case eRegNumARM_DWARF_wCGR2:    return "wCGR2/ACC2";
            case eRegNumARM_DWARF_wCGR3:    return "wCGR3/ACC3";
            case eRegNumARM_DWARF_wCGR4:    return "wCGR4/ACC4";
            case eRegNumARM_DWARF_wCGR5:    return "wCGR5/ACC5";
            case eRegNumARM_DWARF_wCGR6:    return "wCGR6/ACC6";
            case eRegNumARM_DWARF_wCGR7:    return "wCGR7/ACC7";
            case eRegNumARM_DWARF_wR0:      return "wR0";
            case eRegNumARM_DWARF_wR1:      return "wR1";
            case eRegNumARM_DWARF_wR2:      return "wR2";
            case eRegNumARM_DWARF_wR3:      return "wR3";
            case eRegNumARM_DWARF_wR4:      return "wR4";
            case eRegNumARM_DWARF_wR5:      return "wR5";
            case eRegNumARM_DWARF_wR6:      return "wR6";
            case eRegNumARM_DWARF_wR7:      return "wR7";
            case eRegNumARM_DWARF_wR8:      return "wR8";
            case eRegNumARM_DWARF_wR9:      return "wR9";
            case eRegNumARM_DWARF_wR10:     return "wR10";
            case eRegNumARM_DWARF_wR11:     return "wR11";
            case eRegNumARM_DWARF_wR12:     return "wR12";
            case eRegNumARM_DWARF_wR13:     return "wR13";
            case eRegNumARM_DWARF_wR14:     return "wR14";
            case eRegNumARM_DWARF_wR15:     return "wR15";
            case eRegNumARM_DWARF_spsr:     return "spsr";
            case eRegNumARM_DWARF_spsr_fiq: return "spsr_fiq";
            case eRegNumARM_DWARF_spsr_irq: return "spsr_irq";
            case eRegNumARM_DWARF_spsr_abt: return "spsr_abt";
            case eRegNumARM_DWARF_spsr_und: return "spsr_und";
            case eRegNumARM_DWARF_spsr_svc: return "spsr_svc";
            case eRegNumARM_DWARF_r8_usr:   return "r8_usr";
            case eRegNumARM_DWARF_r9_usr:   return "r9_usr";
            case eRegNumARM_DWARF_r10_usr:  return "r10_usr";
            case eRegNumARM_DWARF_r11_usr:  return "r11_usr";
            case eRegNumARM_DWARF_r12_usr:  return "r12_usr";
            case eRegNumARM_DWARF_r13_usr:  return "sp_usr";
            case eRegNumARM_DWARF_r14_usr:  return "lr_usr";
            case eRegNumARM_DWARF_r8_fiq:   return "r8_fiq";
            case eRegNumARM_DWARF_r9_fiq:   return "r9_fiq";
            case eRegNumARM_DWARF_r10_fiq:  return "r10_fiq";
            case eRegNumARM_DWARF_r11_fiq:  return "r11_fiq";
            case eRegNumARM_DWARF_r12_fiq:  return "r12_fiq";
            case eRegNumARM_DWARF_r13_fiq:  return "sp_fiq";
            case eRegNumARM_DWARF_r14_fiq:  return "lr_fiq";
            case eRegNumARM_DWARF_r13_irq:  return "sp_irq";
            case eRegNumARM_DWARF_r14_irq:  return "lr_irq";
            case eRegNumARM_DWARF_r13_abt:  return "sp_abt";
            case eRegNumARM_DWARF_r14_abt:  return "lr_abt";
            case eRegNumARM_DWARF_r13_und:  return "sp_und";
            case eRegNumARM_DWARF_r14_und:  return "lr_und";
            case eRegNumARM_DWARF_r13_svc:  return "sp_svc";
            case eRegNumARM_DWARF_r14_svc:  return "lr_svc";
            case eRegNumARM_DWARF_wC0:      return "wC0";
            case eRegNumARM_DWARF_wC1:      return "wC1";
            case eRegNumARM_DWARF_wC2:      return "wC2";
            case eRegNumARM_DWARF_wC3:      return "wC3";
            case eRegNumARM_DWARF_wC4:      return "wC4";
            case eRegNumARM_DWARF_wC5:      return "wC5";
            case eRegNumARM_DWARF_wC6:      return "wC6";
            case eRegNumARM_DWARF_wC7:      return "wC7";
            case eRegNumARM_DWARF_d0:       return "d0";
            case eRegNumARM_DWARF_d1:       return "d1";
            case eRegNumARM_DWARF_d2:       return "d2";
            case eRegNumARM_DWARF_d3:       return "d3";
            case eRegNumARM_DWARF_d4:       return "d4";
            case eRegNumARM_DWARF_d5:       return "d5";
            case eRegNumARM_DWARF_d6:       return "d6";
            case eRegNumARM_DWARF_d7:       return "d7";
            case eRegNumARM_DWARF_d8:       return "d8";
            case eRegNumARM_DWARF_d9:       return "d9";
            case eRegNumARM_DWARF_d10:      return "d10";
            case eRegNumARM_DWARF_d11:      return "d11";
            case eRegNumARM_DWARF_d12:      return "d12";
            case eRegNumARM_DWARF_d13:      return "d13";
            case eRegNumARM_DWARF_d14:      return "d14";
            case eRegNumARM_DWARF_d15:      return "d15";
            case eRegNumARM_DWARF_d16:      return "d16";
            case eRegNumARM_DWARF_d17:      return "d17";
            case eRegNumARM_DWARF_d18:      return "d18";
            case eRegNumARM_DWARF_d19:      return "d19";
            case eRegNumARM_DWARF_d20:      return "d20";
            case eRegNumARM_DWARF_d21:      return "d21";
            case eRegNumARM_DWARF_d22:      return "d22";
            case eRegNumARM_DWARF_d23:      return "d23";
            case eRegNumARM_DWARF_d24:      return "d24";
            case eRegNumARM_DWARF_d25:      return "d25";
            case eRegNumARM_DWARF_d26:      return "d26";
            case eRegNumARM_DWARF_d27:      return "d27";
            case eRegNumARM_DWARF_d28:      return "d28";
            case eRegNumARM_DWARF_d29:      return "d29";
            case eRegNumARM_DWARF_d30:      return "d30";
            case eRegNumARM_DWARF_d31:      return "d31";
            }
            break;
        default:
            break;
        }
    }
    else if ((arch_type == eArchTypeMachO && (cpu == llvm::MachO::CPUTypePowerPC || cpu == llvm::MachO::CPUTypePowerPC64)) ||
             (arch_type == eArchTypeELF   && cpu == llvm::ELF::EM_PPC))
    {
        switch (reg_kind)
        {
        case eRegisterKindGCC:
            switch (reg_num)
            {
            case eRegNumPPC_GCC_r0:         return "r0";
            case eRegNumPPC_GCC_r1:         return "r1";
            case eRegNumPPC_GCC_r2:         return "r2";
            case eRegNumPPC_GCC_r3:         return "r3";
            case eRegNumPPC_GCC_r4:         return "r4";
            case eRegNumPPC_GCC_r5:         return "r5";
            case eRegNumPPC_GCC_r6:         return "r6";
            case eRegNumPPC_GCC_r7:         return "r7";
            case eRegNumPPC_GCC_r8:         return "r8";
            case eRegNumPPC_GCC_r9:         return "r9";
            case eRegNumPPC_GCC_r10:        return "r10";
            case eRegNumPPC_GCC_r11:        return "r11";
            case eRegNumPPC_GCC_r12:        return "r12";
            case eRegNumPPC_GCC_r13:        return "r13";
            case eRegNumPPC_GCC_r14:        return "r14";
            case eRegNumPPC_GCC_r15:        return "r15";
            case eRegNumPPC_GCC_r16:        return "r16";
            case eRegNumPPC_GCC_r17:        return "r17";
            case eRegNumPPC_GCC_r18:        return "r18";
            case eRegNumPPC_GCC_r19:        return "r19";
            case eRegNumPPC_GCC_r20:        return "r20";
            case eRegNumPPC_GCC_r21:        return "r21";
            case eRegNumPPC_GCC_r22:        return "r22";
            case eRegNumPPC_GCC_r23:        return "r23";
            case eRegNumPPC_GCC_r24:        return "r24";
            case eRegNumPPC_GCC_r25:        return "r25";
            case eRegNumPPC_GCC_r26:        return "r26";
            case eRegNumPPC_GCC_r27:        return "r27";
            case eRegNumPPC_GCC_r28:        return "r28";
            case eRegNumPPC_GCC_r29:        return "r29";
            case eRegNumPPC_GCC_r30:        return "r30";
            case eRegNumPPC_GCC_r31:        return "r31";
            case eRegNumPPC_GCC_fr0:        return "fr0";
            case eRegNumPPC_GCC_fr1:        return "fr1";
            case eRegNumPPC_GCC_fr2:        return "fr2";
            case eRegNumPPC_GCC_fr3:        return "fr3";
            case eRegNumPPC_GCC_fr4:        return "fr4";
            case eRegNumPPC_GCC_fr5:        return "fr5";
            case eRegNumPPC_GCC_fr6:        return "fr6";
            case eRegNumPPC_GCC_fr7:        return "fr7";
            case eRegNumPPC_GCC_fr8:        return "fr8";
            case eRegNumPPC_GCC_fr9:        return "fr9";
            case eRegNumPPC_GCC_fr10:       return "fr10";
            case eRegNumPPC_GCC_fr11:       return "fr11";
            case eRegNumPPC_GCC_fr12:       return "fr12";
            case eRegNumPPC_GCC_fr13:       return "fr13";
            case eRegNumPPC_GCC_fr14:       return "fr14";
            case eRegNumPPC_GCC_fr15:       return "fr15";
            case eRegNumPPC_GCC_fr16:       return "fr16";
            case eRegNumPPC_GCC_fr17:       return "fr17";
            case eRegNumPPC_GCC_fr18:       return "fr18";
            case eRegNumPPC_GCC_fr19:       return "fr19";
            case eRegNumPPC_GCC_fr20:       return "fr20";
            case eRegNumPPC_GCC_fr21:       return "fr21";
            case eRegNumPPC_GCC_fr22:       return "fr22";
            case eRegNumPPC_GCC_fr23:       return "fr23";
            case eRegNumPPC_GCC_fr24:       return "fr24";
            case eRegNumPPC_GCC_fr25:       return "fr25";
            case eRegNumPPC_GCC_fr26:       return "fr26";
            case eRegNumPPC_GCC_fr27:       return "fr27";
            case eRegNumPPC_GCC_fr28:       return "fr28";
            case eRegNumPPC_GCC_fr29:       return "fr29";
            case eRegNumPPC_GCC_fr30:       return "fr30";
            case eRegNumPPC_GCC_fr31:       return "fr31";
            case eRegNumPPC_GCC_mq:         return "mq";
            case eRegNumPPC_GCC_lr:         return "lr";
            case eRegNumPPC_GCC_ctr:        return "ctr";
            case eRegNumPPC_GCC_ap:         return "ap";
            case eRegNumPPC_GCC_cr0:        return "cr0";
            case eRegNumPPC_GCC_cr1:        return "cr1";
            case eRegNumPPC_GCC_cr2:        return "cr2";
            case eRegNumPPC_GCC_cr3:        return "cr3";
            case eRegNumPPC_GCC_cr4:        return "cr4";
            case eRegNumPPC_GCC_cr5:        return "cr5";
            case eRegNumPPC_GCC_cr6:        return "cr6";
            case eRegNumPPC_GCC_cr7:        return "cr7";
            case eRegNumPPC_GCC_xer:        return "xer";
            case eRegNumPPC_GCC_v0:         return "v0";
            case eRegNumPPC_GCC_v1:         return "v1";
            case eRegNumPPC_GCC_v2:         return "v2";
            case eRegNumPPC_GCC_v3:         return "v3";
            case eRegNumPPC_GCC_v4:         return "v4";
            case eRegNumPPC_GCC_v5:         return "v5";
            case eRegNumPPC_GCC_v6:         return "v6";
            case eRegNumPPC_GCC_v7:         return "v7";
            case eRegNumPPC_GCC_v8:         return "v8";
            case eRegNumPPC_GCC_v9:         return "v9";
            case eRegNumPPC_GCC_v10:        return "v10";
            case eRegNumPPC_GCC_v11:        return "v11";
            case eRegNumPPC_GCC_v12:        return "v12";
            case eRegNumPPC_GCC_v13:        return "v13";
            case eRegNumPPC_GCC_v14:        return "v14";
            case eRegNumPPC_GCC_v15:        return "v15";
            case eRegNumPPC_GCC_v16:        return "v16";
            case eRegNumPPC_GCC_v17:        return "v17";
            case eRegNumPPC_GCC_v18:        return "v18";
            case eRegNumPPC_GCC_v19:        return "v19";
            case eRegNumPPC_GCC_v20:        return "v20";
            case eRegNumPPC_GCC_v21:        return "v21";
            case eRegNumPPC_GCC_v22:        return "v22";
            case eRegNumPPC_GCC_v23:        return "v23";
            case eRegNumPPC_GCC_v24:        return "v24";
            case eRegNumPPC_GCC_v25:        return "v25";
            case eRegNumPPC_GCC_v26:        return "v26";
            case eRegNumPPC_GCC_v27:        return "v27";
            case eRegNumPPC_GCC_v28:        return "v28";
            case eRegNumPPC_GCC_v29:        return "v29";
            case eRegNumPPC_GCC_v30:        return "v30";
            case eRegNumPPC_GCC_v31:        return "v31";
            case eRegNumPPC_GCC_vrsave:     return "vrsave";
            case eRegNumPPC_GCC_vscr:       return "vscr";
            case eRegNumPPC_GCC_spe_acc:    return "spe_acc";
            case eRegNumPPC_GCC_spefscr:    return "spefscr";
            case eRegNumPPC_GCC_sfp:        return "sfp";
            default:
                break;
            }
            break;

        case eRegisterKindDWARF:
            switch (reg_num)
            {
            case eRegNumPPC_DWARF_r0:       return "r0";
            case eRegNumPPC_DWARF_r1:       return "r1";
            case eRegNumPPC_DWARF_r2:       return "r2";
            case eRegNumPPC_DWARF_r3:       return "r3";
            case eRegNumPPC_DWARF_r4:       return "r4";
            case eRegNumPPC_DWARF_r5:       return "r5";
            case eRegNumPPC_DWARF_r6:       return "r6";
            case eRegNumPPC_DWARF_r7:       return "r7";
            case eRegNumPPC_DWARF_r8:       return "r8";
            case eRegNumPPC_DWARF_r9:       return "r9";
            case eRegNumPPC_DWARF_r10:      return "r10";
            case eRegNumPPC_DWARF_r11:      return "r11";
            case eRegNumPPC_DWARF_r12:      return "r12";
            case eRegNumPPC_DWARF_r13:      return "r13";
            case eRegNumPPC_DWARF_r14:      return "r14";
            case eRegNumPPC_DWARF_r15:      return "r15";
            case eRegNumPPC_DWARF_r16:      return "r16";
            case eRegNumPPC_DWARF_r17:      return "r17";
            case eRegNumPPC_DWARF_r18:      return "r18";
            case eRegNumPPC_DWARF_r19:      return "r19";
            case eRegNumPPC_DWARF_r20:      return "r20";
            case eRegNumPPC_DWARF_r21:      return "r21";
            case eRegNumPPC_DWARF_r22:      return "r22";
            case eRegNumPPC_DWARF_r23:      return "r23";
            case eRegNumPPC_DWARF_r24:      return "r24";
            case eRegNumPPC_DWARF_r25:      return "r25";
            case eRegNumPPC_DWARF_r26:      return "r26";
            case eRegNumPPC_DWARF_r27:      return "r27";
            case eRegNumPPC_DWARF_r28:      return "r28";
            case eRegNumPPC_DWARF_r29:      return "r29";
            case eRegNumPPC_DWARF_r30:      return "r30";
            case eRegNumPPC_DWARF_r31:      return "r31";

            case eRegNumPPC_DWARF_fr0:      return "fr0";
            case eRegNumPPC_DWARF_fr1:      return "fr1";
            case eRegNumPPC_DWARF_fr2:      return "fr2";
            case eRegNumPPC_DWARF_fr3:      return "fr3";
            case eRegNumPPC_DWARF_fr4:      return "fr4";
            case eRegNumPPC_DWARF_fr5:      return "fr5";
            case eRegNumPPC_DWARF_fr6:      return "fr6";
            case eRegNumPPC_DWARF_fr7:      return "fr7";
            case eRegNumPPC_DWARF_fr8:      return "fr8";
            case eRegNumPPC_DWARF_fr9:      return "fr9";
            case eRegNumPPC_DWARF_fr10:     return "fr10";
            case eRegNumPPC_DWARF_fr11:     return "fr11";
            case eRegNumPPC_DWARF_fr12:     return "fr12";
            case eRegNumPPC_DWARF_fr13:     return "fr13";
            case eRegNumPPC_DWARF_fr14:     return "fr14";
            case eRegNumPPC_DWARF_fr15:     return "fr15";
            case eRegNumPPC_DWARF_fr16:     return "fr16";
            case eRegNumPPC_DWARF_fr17:     return "fr17";
            case eRegNumPPC_DWARF_fr18:     return "fr18";
            case eRegNumPPC_DWARF_fr19:     return "fr19";
            case eRegNumPPC_DWARF_fr20:     return "fr20";
            case eRegNumPPC_DWARF_fr21:     return "fr21";
            case eRegNumPPC_DWARF_fr22:     return "fr22";
            case eRegNumPPC_DWARF_fr23:     return "fr23";
            case eRegNumPPC_DWARF_fr24:     return "fr24";
            case eRegNumPPC_DWARF_fr25:     return "fr25";
            case eRegNumPPC_DWARF_fr26:     return "fr26";
            case eRegNumPPC_DWARF_fr27:     return "fr27";
            case eRegNumPPC_DWARF_fr28:     return "fr28";
            case eRegNumPPC_DWARF_fr29:     return "fr29";
            case eRegNumPPC_DWARF_fr30:     return "fr30";
            case eRegNumPPC_DWARF_fr31:     return "fr31";

            case eRegNumPPC_DWARF_cr:       return "cr";
            case eRegNumPPC_DWARF_fpscr:    return "fpscr";
            case eRegNumPPC_DWARF_msr:      return "msr";
            case eRegNumPPC_DWARF_vscr:     return "vscr";

            case eRegNumPPC_DWARF_sr0:      return "sr0";
            case eRegNumPPC_DWARF_sr1:      return "sr1";
            case eRegNumPPC_DWARF_sr2:      return "sr2";
            case eRegNumPPC_DWARF_sr3:      return "sr3";
            case eRegNumPPC_DWARF_sr4:      return "sr4";
            case eRegNumPPC_DWARF_sr5:      return "sr5";
            case eRegNumPPC_DWARF_sr6:      return "sr6";
            case eRegNumPPC_DWARF_sr7:      return "sr7";
            case eRegNumPPC_DWARF_sr8:      return "sr8";
            case eRegNumPPC_DWARF_sr9:      return "sr9";
            case eRegNumPPC_DWARF_sr10:     return "sr10";
            case eRegNumPPC_DWARF_sr11:     return "sr11";
            case eRegNumPPC_DWARF_sr12:     return "sr12";
            case eRegNumPPC_DWARF_sr13:     return "sr13";
            case eRegNumPPC_DWARF_sr14:     return "sr14";
            case eRegNumPPC_DWARF_sr15:     return "sr15";

            case eRegNumPPC_DWARF_acc:      return "acc";
            case eRegNumPPC_DWARF_mq:       return "mq";
            case eRegNumPPC_DWARF_xer:      return "xer";
            case eRegNumPPC_DWARF_rtcu:     return "rtcu";
            case eRegNumPPC_DWARF_rtcl:     return "rtcl";

            case eRegNumPPC_DWARF_lr:       return "lr";
            case eRegNumPPC_DWARF_ctr:      return "ctr";

            case eRegNumPPC_DWARF_dsisr:    return "dsisr";
            case eRegNumPPC_DWARF_dar:      return "dar";
            case eRegNumPPC_DWARF_dec:      return "dec";
            case eRegNumPPC_DWARF_sdr1:     return "sdr1";
            case eRegNumPPC_DWARF_srr0:     return "srr0";
            case eRegNumPPC_DWARF_srr1:     return "srr1";

            case eRegNumPPC_DWARF_vrsave:   return "vrsave";

            case eRegNumPPC_DWARF_sprg0:    return "sprg0";
            case eRegNumPPC_DWARF_sprg1:    return "sprg1";
            case eRegNumPPC_DWARF_sprg2:    return "sprg2";
            case eRegNumPPC_DWARF_sprg3:    return "sprg3";

            case eRegNumPPC_DWARF_asr:      return "asr";
            case eRegNumPPC_DWARF_ear:      return "ear";
            case eRegNumPPC_DWARF_tb:       return "tb";
            case eRegNumPPC_DWARF_tbu:      return "tbu";
            case eRegNumPPC_DWARF_pvr:      return "pvr";

            case eRegNumPPC_DWARF_spefscr:  return "spefscr";

            case eRegNumPPC_DWARF_ibat0u:   return "ibat0u";
            case eRegNumPPC_DWARF_ibat0l:   return "ibat0l";
            case eRegNumPPC_DWARF_ibat1u:   return "ibat1u";
            case eRegNumPPC_DWARF_ibat1l:   return "ibat1l";
            case eRegNumPPC_DWARF_ibat2u:   return "ibat2u";
            case eRegNumPPC_DWARF_ibat2l:   return "ibat2l";
            case eRegNumPPC_DWARF_ibat3u:   return "ibat3u";
            case eRegNumPPC_DWARF_ibat3l:   return "ibat3l";
            case eRegNumPPC_DWARF_dbat0u:   return "dbat0u";
            case eRegNumPPC_DWARF_dbat0l:   return "dbat0l";
            case eRegNumPPC_DWARF_dbat1u:   return "dbat1u";
            case eRegNumPPC_DWARF_dbat1l:   return "dbat1l";
            case eRegNumPPC_DWARF_dbat2u:   return "dbat2u";
            case eRegNumPPC_DWARF_dbat2l:   return "dbat2l";
            case eRegNumPPC_DWARF_dbat3u:   return "dbat3u";
            case eRegNumPPC_DWARF_dbat3l:   return "dbat3l";

            case eRegNumPPC_DWARF_hid0:     return "hid0";
            case eRegNumPPC_DWARF_hid1:     return "hid1";
            case eRegNumPPC_DWARF_hid2:     return "hid2";
            case eRegNumPPC_DWARF_hid3:     return "hid3";
            case eRegNumPPC_DWARF_hid4:     return "hid4";
            case eRegNumPPC_DWARF_hid5:     return "hid5";
            case eRegNumPPC_DWARF_hid6:     return "hid6";
            case eRegNumPPC_DWARF_hid7:     return "hid7";
            case eRegNumPPC_DWARF_hid8:     return "hid8";
            case eRegNumPPC_DWARF_hid9:     return "hid9";
            case eRegNumPPC_DWARF_hid10:    return "hid10";
            case eRegNumPPC_DWARF_hid11:    return "hid11";
            case eRegNumPPC_DWARF_hid12:    return "hid12";
            case eRegNumPPC_DWARF_hid13:    return "hid13";
            case eRegNumPPC_DWARF_hid14:    return "hid14";
            case eRegNumPPC_DWARF_hid15:    return "hid15";

            case eRegNumPPC_DWARF_vr0:      return "vr0";
            case eRegNumPPC_DWARF_vr1:      return "vr1";
            case eRegNumPPC_DWARF_vr2:      return "vr2";
            case eRegNumPPC_DWARF_vr3:      return "vr3";
            case eRegNumPPC_DWARF_vr4:      return "vr4";
            case eRegNumPPC_DWARF_vr5:      return "vr5";
            case eRegNumPPC_DWARF_vr6:      return "vr6";
            case eRegNumPPC_DWARF_vr7:      return "vr7";
            case eRegNumPPC_DWARF_vr8:      return "vr8";
            case eRegNumPPC_DWARF_vr9:      return "vr9";
            case eRegNumPPC_DWARF_vr10:     return "vr10";
            case eRegNumPPC_DWARF_vr11:     return "vr11";
            case eRegNumPPC_DWARF_vr12:     return "vr12";
            case eRegNumPPC_DWARF_vr13:     return "vr13";
            case eRegNumPPC_DWARF_vr14:     return "vr14";
            case eRegNumPPC_DWARF_vr15:     return "vr15";
            case eRegNumPPC_DWARF_vr16:     return "vr16";
            case eRegNumPPC_DWARF_vr17:     return "vr17";
            case eRegNumPPC_DWARF_vr18:     return "vr18";
            case eRegNumPPC_DWARF_vr19:     return "vr19";
            case eRegNumPPC_DWARF_vr20:     return "vr20";
            case eRegNumPPC_DWARF_vr21:     return "vr21";
            case eRegNumPPC_DWARF_vr22:     return "vr22";
            case eRegNumPPC_DWARF_vr23:     return "vr23";
            case eRegNumPPC_DWARF_vr24:     return "vr24";
            case eRegNumPPC_DWARF_vr25:     return "vr25";
            case eRegNumPPC_DWARF_vr26:     return "vr26";
            case eRegNumPPC_DWARF_vr27:     return "vr27";
            case eRegNumPPC_DWARF_vr28:     return "vr28";
            case eRegNumPPC_DWARF_vr29:     return "vr29";
            case eRegNumPPC_DWARF_vr30:     return "vr30";
            case eRegNumPPC_DWARF_vr31:     return "vr31";

            case eRegNumPPC_DWARF_ev0:      return "ev0";
            case eRegNumPPC_DWARF_ev1:      return "ev1";
            case eRegNumPPC_DWARF_ev2:      return "ev2";
            case eRegNumPPC_DWARF_ev3:      return "ev3";
            case eRegNumPPC_DWARF_ev4:      return "ev4";
            case eRegNumPPC_DWARF_ev5:      return "ev5";
            case eRegNumPPC_DWARF_ev6:      return "ev6";
            case eRegNumPPC_DWARF_ev7:      return "ev7";
            case eRegNumPPC_DWARF_ev8:      return "ev8";
            case eRegNumPPC_DWARF_ev9:      return "ev9";
            case eRegNumPPC_DWARF_ev10:     return "ev10";
            case eRegNumPPC_DWARF_ev11:     return "ev11";
            case eRegNumPPC_DWARF_ev12:     return "ev12";
            case eRegNumPPC_DWARF_ev13:     return "ev13";
            case eRegNumPPC_DWARF_ev14:     return "ev14";
            case eRegNumPPC_DWARF_ev15:     return "ev15";
            case eRegNumPPC_DWARF_ev16:     return "ev16";
            case eRegNumPPC_DWARF_ev17:     return "ev17";
            case eRegNumPPC_DWARF_ev18:     return "ev18";
            case eRegNumPPC_DWARF_ev19:     return "ev19";
            case eRegNumPPC_DWARF_ev20:     return "ev20";
            case eRegNumPPC_DWARF_ev21:     return "ev21";
            case eRegNumPPC_DWARF_ev22:     return "ev22";
            case eRegNumPPC_DWARF_ev23:     return "ev23";
            case eRegNumPPC_DWARF_ev24:     return "ev24";
            case eRegNumPPC_DWARF_ev25:     return "ev25";
            case eRegNumPPC_DWARF_ev26:     return "ev26";
            case eRegNumPPC_DWARF_ev27:     return "ev27";
            case eRegNumPPC_DWARF_ev28:     return "ev28";
            case eRegNumPPC_DWARF_ev29:     return "ev29";
            case eRegNumPPC_DWARF_ev30:     return "ev30";
            case eRegNumPPC_DWARF_ev31:     return "ev31";
            default:
                break;
            }
            break;
        default:
            break;
        }

    }
    return NULL;
}

//----------------------------------------------------------------------
// Returns true if this object contains a valid architecture, false
// otherwise.
//----------------------------------------------------------------------
bool
ArchSpec::IsValid() const
{
    return !(m_cpu == LLDB_INVALID_CPUTYPE);
}

//----------------------------------------------------------------------
// Returns true if this architecture is 64 bit, otherwise 32 bit is
// assumed and false is returned.
//----------------------------------------------------------------------
uint32_t
ArchSpec::GetAddressByteSize() const
{
    switch (m_type)
    {
    case kNumArchTypes:
    case eArchTypeInvalid:
        break;

    case eArchTypeMachO:
        if (GetCPUType() & llvm::MachO::CPUArchABI64)
            return 8;
        else
            return 4;
        break;
    
    case eArchTypeELF:
        switch (m_cpu)
        {
        case llvm::ELF::EM_M32:
        case llvm::ELF::EM_SPARC:
        case llvm::ELF::EM_386:
        case llvm::ELF::EM_68K:
        case llvm::ELF::EM_88K:
        case llvm::ELF::EM_486:
        case llvm::ELF::EM_860:
        case llvm::ELF::EM_MIPS:
        case llvm::ELF::EM_PPC:
        case llvm::ELF::EM_ARM:
        case llvm::ELF::EM_ALPHA:
        case llvm::ELF::EM_SPARCV9:
            return 4;
        case llvm::ELF::EM_X86_64:
            return 8;
        }
        break;
    }

    return 0;
}

//----------------------------------------------------------------------
// Returns the number of bytes that this object takes when an
// instance exists in memory.
//----------------------------------------------------------------------
size_t
ArchSpec::MemorySize() const
{
    return sizeof(ArchSpec);
}

bool
ArchSpec::SetArchFromTargetTriple (const char *target_triple)
{
    if (target_triple)
    {
        const char *hyphen = strchr(target_triple, '-');
        if (hyphen)
        {
            std::string arch_only (target_triple, hyphen);
            return SetArch (arch_only.c_str());
        }
    }
    return SetArch (target_triple);
}

//----------------------------------------------------------------------
// Change the CPU type and subtype given an architecture name.
//----------------------------------------------------------------------
bool
ArchSpec::SetArch (const char *arch_name)
{
    if (arch_name && arch_name[0] != '\0')
    {
        size_t i;

        switch (m_type)
        {
        case eArchTypeInvalid:
        case eArchTypeMachO:
            for (i=0; i<k_num_mach_arch_defs; i++)
            {
                if (strcasecmp(arch_name, g_mach_arch_defs[i].name) == 0)
                {
                    m_type = eArchTypeMachO;
                    m_cpu = g_mach_arch_defs[i].cpu;
                    m_sub = g_mach_arch_defs[i].sub;
                    return true;
                }
            }
            break;
        
        case eArchTypeELF:
            for (i=0; i<k_num_elf_arch_defs; i++)
            {
                if (strcasecmp(arch_name, g_elf_arch_defs[i].name) == 0)
                {
                    m_cpu = g_elf_arch_defs[i].cpu;
                    m_sub = g_elf_arch_defs[i].sub;
                    return true;
                }
            }
            break;

        case kNumArchTypes:
            break;
        }

        const char *str = arch_name;
        // Check for a numeric cpu followed by an optional separator char and numeric subtype.
        // This allows for support of new cpu type/subtypes without having to have
        // a recompiled debug core.
        // Examples:
        //  "12.6" is armv6
        //  "0x0000000c-0x00000006" is also armv6
        
        m_type = eArchTypeInvalid;
        for (i=1; i<kNumArchTypes; ++i)
        {
            const char *arch_type_cstr = g_arch_type_strings[i];
            if (strstr(str, arch_type_cstr))
            {
                m_type = (ArchitectureType)i;
                str += strlen(arch_type_cstr) + 1; // Also skip separator char
            }
        }
        
        if (m_type != eArchTypeInvalid)
        {
            char *end = NULL;
            m_cpu = ::strtoul (str, &end, 0);
            if (str != end)
            {
                if (*end == ARCH_SPEC_SEPARATOR_CHAR)
                {
                    // We have a cputype.cpusubtype format
                    str = end + 1;
                    if (*str != '\0')
                    {
                        m_sub = strtoul(str, &end, 0);
                        if (*end == '\0')
                        {
                            // We consumed the entire string and got a cpu type and subtype
                            return true;
                        }
                    }
                }

                // If we reach this point we have a valid cpu type, but no cpu subtype.
                // Search for the first matching cpu type and use the corresponding cpu
                // subtype. This setting should typically be the _ALL variant and should
                // appear first in the list for each cpu type in the g_mach_arch_defs
                // structure.
                for (i=0; i<k_num_mach_arch_defs; ++i)
                {
                    if (m_cpu == g_mach_arch_defs[i].cpu)
                    {
                        m_sub = g_mach_arch_defs[i].sub;
                        return true;
                    }
                }

                // Default the cpu subtype to zero when we don't have a matching
                // cpu type in our architecture defs structure (g_mach_arch_defs).
                m_sub = 0;
                return true;

            }
        }
    }
    Clear();
    return false;
}

//----------------------------------------------------------------------
// CPU type and subtype set accessor.
//----------------------------------------------------------------------
void
ArchSpec::SetArch (uint32_t cpu_type, uint32_t cpu_subtype)
{
    m_cpu = cpu_type;
    m_sub = cpu_subtype;
}

//----------------------------------------------------------------------
// CPU type set accessor.
//----------------------------------------------------------------------
void
ArchSpec::SetCPUType (uint32_t cpu)
{
    m_cpu = cpu;
}

//----------------------------------------------------------------------
// CPU subtype set accessor.
//----------------------------------------------------------------------
void
ArchSpec::SetCPUSubtype (uint32_t subtype)
{
    m_sub = subtype;
}

ByteOrder
ArchSpec::GetDefaultEndian () const
{
    switch (GetGenericCPUType ())
    {
    case eCPU_ppc:
    case eCPU_ppc64:
        return eByteOrderBig;

    case eCPU_arm:
    case eCPU_i386:
    case eCPU_x86_64:
        return eByteOrderLittle;

    default:
        break;
    }
    return eByteOrderInvalid;
}

//----------------------------------------------------------------------
// Equal operator
//----------------------------------------------------------------------
bool
lldb_private::operator== (const ArchSpec& lhs, const ArchSpec& rhs)
{
    uint32_t lhs_cpu = lhs.GetCPUType();
    uint32_t rhs_cpu = rhs.GetCPUType();

    if (lhs_cpu == CPU_ANY || rhs_cpu == CPU_ANY)
        return true;

    else if (lhs_cpu == rhs_cpu)
    {
        uint32_t lhs_subtype = lhs.GetCPUSubtype();
        uint32_t rhs_subtype = rhs.GetCPUSubtype();
        if (lhs_subtype == CPU_ANY || rhs_subtype == CPU_ANY)
            return true;
        return lhs_subtype == rhs_subtype;
    }
    return false;
}


//----------------------------------------------------------------------
// Not Equal operator
//----------------------------------------------------------------------
bool
lldb_private::operator!= (const ArchSpec& lhs, const ArchSpec& rhs)
{
    return !(lhs == rhs);
}

//----------------------------------------------------------------------
// Less than operator
//----------------------------------------------------------------------
bool
lldb_private::operator<(const ArchSpec& lhs, const ArchSpec& rhs)
{
    uint32_t lhs_cpu = lhs.GetCPUType();
    uint32_t rhs_cpu = rhs.GetCPUType();

    if (lhs_cpu == rhs_cpu)
        return lhs.GetCPUSubtype() < rhs.GetCPUSubtype();

    return lhs_cpu < rhs_cpu;
}

