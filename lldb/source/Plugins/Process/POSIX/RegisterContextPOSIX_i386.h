//===-- RegisterContextPOSIX_i386.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContext_i386_h_
#define liblldb_RegisterContext_i386_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "RegisterContextPOSIX.h"
#include "RegisterContext_x86.h"

//---------------------------------------------------------------------------
// Internal codes for all i386 registers.
//---------------------------------------------------------------------------
enum
{
    k_first_gpr_i386,
    gpr_eax_i386 = k_first_gpr_i386,
    gpr_ebx_i386,
    gpr_ecx_i386,
    gpr_edx_i386,
    gpr_edi_i386,
    gpr_esi_i386,
    gpr_ebp_i386,
    gpr_esp_i386,
    gpr_eip_i386,
    gpr_eflags_i386,
    gpr_cs_i386,
    gpr_fs_i386,
    gpr_gs_i386,
    gpr_ss_i386,
    gpr_ds_i386,
    gpr_es_i386,

    k_first_alias_i386,
    gpr_ax_i386 = k_first_alias_i386,
    gpr_bx_i386,
    gpr_cx_i386,
    gpr_dx_i386,
    gpr_di_i386,
    gpr_si_i386,
    gpr_bp_i386,
    gpr_sp_i386,
    gpr_ah_i386,
    gpr_bh_i386,
    gpr_ch_i386,
    gpr_dh_i386,
    gpr_al_i386,
    gpr_bl_i386,
    gpr_cl_i386,
    gpr_dl_i386,
    k_last_alias_i386 = gpr_dl_i386,

    k_last_gpr_i386 = k_last_alias_i386,

    k_first_fpr_i386,
    fpu_fctrl_i386 = k_first_fpr_i386,
    fpu_fstat_i386,
    fpu_ftag_i386,
    fpu_fop_i386,
    fpu_fiseg_i386,
    fpu_fioff_i386,
    fpu_foseg_i386,
    fpu_fooff_i386,
    fpu_mxcsr_i386,
    fpu_mxcsrmask_i386,
    fpu_st0_i386,
    fpu_st1_i386,
    fpu_st2_i386,
    fpu_st3_i386,
    fpu_st4_i386,
    fpu_st5_i386,
    fpu_st6_i386,
    fpu_st7_i386,
    fpu_mm0_i386,
    fpu_mm1_i386,
    fpu_mm2_i386,
    fpu_mm3_i386,
    fpu_mm4_i386,
    fpu_mm5_i386,
    fpu_mm6_i386,
    fpu_mm7_i386,
    fpu_xmm0_i386,
    fpu_xmm1_i386,
    fpu_xmm2_i386,
    fpu_xmm3_i386,
    fpu_xmm4_i386,
    fpu_xmm5_i386,
    fpu_xmm6_i386,
    fpu_xmm7_i386,
    k_last_fpr_i386 = fpu_xmm7_i386,

    k_first_avx_i386,
    fpu_ymm0_i386 = k_first_avx_i386,
    fpu_ymm1_i386,
    fpu_ymm2_i386,
    fpu_ymm3_i386,
    fpu_ymm4_i386,
    fpu_ymm5_i386,
    fpu_ymm6_i386,
    fpu_ymm7_i386,
    k_last_avx_i386 = fpu_ymm7_i386,

    dr0_i386,
    dr1_i386,
    dr2_i386,
    dr3_i386,
    dr4_i386,
    dr5_i386,
    dr6_i386,
    dr7_i386,

    k_num_registers_i386,
    k_num_gpr_registers_i386 = k_last_gpr_i386 - k_first_gpr_i386 + 1,
    k_num_fpr_registers_i386 = k_last_fpr_i386 - k_first_fpr_i386 + 1,
    k_num_avx_registers_i386 = k_last_avx_i386 - k_first_avx_i386 + 1
};

class RegisterContextPOSIX_i386 :
    public lldb_private::RegisterContext
{
public:
    RegisterContextPOSIX_i386(lldb_private::Thread &thread,
                              uint32_t concreate_frame_idx,
                              RegisterInfoInterface *register_info);

    ~RegisterContextPOSIX_i386();

    void
    Invalidate();

    void
    InvalidateAllRegisters();

    size_t
    GetRegisterCount();

    virtual unsigned
    GetRegisterSize(unsigned reg);

    virtual unsigned
    GetRegisterOffset(unsigned reg);

    const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex(size_t reg);

    size_t
    GetRegisterSetCount();

    const lldb_private::RegisterSet *
    GetRegisterSet(size_t set);

    const char *
    GetRegisterName(unsigned reg);

    bool
    ReadRegisterValue(uint32_t reg, lldb_private::Scalar &value);

    bool
    ReadRegisterBytes(uint32_t reg, lldb_private::DataExtractor &data);

    virtual bool
    ReadRegister(const lldb_private::RegisterInfo *reg_info,
                 lldb_private::RegisterValue &value) = 0;

    bool
    ReadAllRegisterValues(lldb::DataBufferSP &data_sp);

    bool
    WriteRegisterValue(uint32_t reg, const lldb_private::Scalar &value);

    bool
    WriteRegisterBytes(uint32_t reg, lldb_private::DataExtractor &data,
                       uint32_t data_offset = 0);

    virtual bool
    WriteRegister(const lldb_private::RegisterInfo *reg_info,
                  const lldb_private::RegisterValue &value) = 0;

    bool
    WriteAllRegisterValues(const lldb::DataBufferSP &data_sp);

    uint32_t
    ConvertRegisterKindToRegisterNumber(uint32_t kind, uint32_t num);

    bool
    HardwareSingleStep(bool enable);

    bool
    UpdateAfterBreakpoint();

    //---------------------------------------------------------------------------
    // Note: prefer kernel definitions over user-land
    //---------------------------------------------------------------------------
    enum FPRType
    {
        eNotValid = 0,
        eFSAVE,  // TODO
        eFXSAVE,
        eSOFT,   // TODO
        eXSAVE
    };

    static uint32_t g_contained_eax[];
    static uint32_t g_contained_ebx[];
    static uint32_t g_contained_ecx[];
    static uint32_t g_contained_edx[];
    static uint32_t g_contained_edi[];
    static uint32_t g_contained_esi[];
    static uint32_t g_contained_ebp[];
    static uint32_t g_contained_esp[];

    static uint32_t g_invalidate_eax[];
    static uint32_t g_invalidate_ebx[];
    static uint32_t g_invalidate_ecx[];
    static uint32_t g_invalidate_edx[];
    static uint32_t g_invalidate_edi[];
    static uint32_t g_invalidate_esi[];
    static uint32_t g_invalidate_ebp[];
    static uint32_t g_invalidate_esp[];

    static const uint32_t g_gpr_regnums[]; // k_num_gpr_registers_i386 
    static const uint32_t g_fpu_regnums[]; // k_num_fpr_registers_i386 
    static const uint32_t g_avx_regnums[]; // k_num_avx_registers_i386 

protected:
    virtual const lldb_private::RegisterInfo *
    GetRegisterInfo();

    FPRType m_fpr_type;                                        // determines the type of data stored by union FPR, if any.
    std::unique_ptr<RegisterInfoInterface> m_register_info_ap; // Register Info Interface (FreeBSD or Linux)

    FPRType GetFPRType();

    virtual bool ReadGPR() = 0;
    virtual bool ReadFPR() = 0;
    virtual bool WriteGPR() = 0;
    virtual bool WriteFPR() = 0;
};

#endif // #ifndef liblldb_RegisterContext_i386_h_
