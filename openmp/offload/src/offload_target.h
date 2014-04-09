//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


// The parts of the offload library used only on the target

#ifndef OFFLOAD_TARGET_H_INCLUDED
#define OFFLOAD_TARGET_H_INCLUDED

#include "offload_common.h"
#include "coi/coi_server.h"

// The offload descriptor.
class OffloadDescriptor
{
public:
    ~OffloadDescriptor() {
        if (m_vars != 0) {
            free(m_vars);
        }
    }

    // Entry point for COI. Synchronously execute offloaded region given
    // the provided buffers, misc and return data.
    static void offload(
        uint32_t  buffer_count,
        void**    buffers,
        void*     misc_data,
        uint16_t  misc_data_len,
        void*     return_data,
        uint16_t  return_data_len
    );

    // scatters input data from in buffer to target variables
    void scatter_copyin_data();

    // gathers output data to the buffer
    void gather_copyout_data();

    // merges local variable descriptors with the descriptors received from
    // host
    void merge_var_descs(VarDesc *vars, VarDesc2 *vars2, int vars_total);

    int get_offload_number() const {
        return m_offload_number;
    }

    void set_offload_number(int number) {
        m_offload_number = number;
    }

private:
    // Constructor
    OffloadDescriptor() : m_vars(0)
    {}

private:
    typedef std::list<void*> BufferList;

    // The Marshaller for the inputs of the offloaded region.
    Marshaller m_in;

    // The Marshaller for the outputs of the offloaded region.
    Marshaller m_out;

    // List of buffers that are passed to dispatch call
    BufferList m_buffers;

    // Variable descriptors received from host
    VarDesc* m_vars;
    int      m_vars_total;
    int      m_offload_number;
};

// one time target initialization in main
extern void __offload_target_init(void);

// logical device index
extern int mic_index;

// total number of available logical devices
extern int mic_engines_total;

// device frequency (from COI)
extern uint64_t mic_frequency;

struct RefInfo {
    RefInfo(bool is_add, long amount):is_added(is_add),count(amount)
    {}
    bool is_added;
    long count;
};

#endif // OFFLOAD_TARGET_H_INCLUDED
