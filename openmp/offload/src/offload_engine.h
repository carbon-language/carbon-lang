//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef OFFLOAD_ENGINE_H_INCLUDED
#define OFFLOAD_ENGINE_H_INCLUDED

#include <limits.h>

#include <list>
#include <set>
#include <map>
#include "offload_common.h"
#include "coi/coi_client.h"

// Address range
class MemRange {
public:
    MemRange() : m_start(0), m_length(0) {}
    MemRange(const void *addr, uint64_t len) : m_start(addr), m_length(len) {}

    const void* start() const {
        return m_start;
    }

    const void* end() const {
        return static_cast<const char*>(m_start) + m_length;
    }

    uint64_t length() const {
        return m_length;
    }

    // returns true if given range overlaps with another one
    bool overlaps(const MemRange &o) const {
        // Two address ranges A[start, end) and B[start,end) overlap
        // if A.start < B.end and A.end > B.start.
        return start() < o.end() && end() > o.start();
    }

    // returns true if given range contains the other range
    bool contains(const MemRange &o) const {
        return start() <= o.start() && o.end() <= end();
    }

private:
    const void* m_start;
    uint64_t    m_length;
};

// Data associated with a pointer variable
class PtrData {
public:
    PtrData(const void *addr, uint64_t len) :
        cpu_addr(addr, len), cpu_buf(0),
        mic_addr(0), alloc_disp(0), mic_buf(0), mic_offset(0),
        ref_count(0), is_static(false)
    {}

    //
    // Copy constructor
    //
    PtrData(const PtrData& ptr):
        cpu_addr(ptr.cpu_addr), cpu_buf(ptr.cpu_buf),
        mic_addr(ptr.mic_addr), alloc_disp(ptr.alloc_disp),
        mic_buf(ptr.mic_buf), mic_offset(ptr.mic_offset),
        ref_count(ptr.ref_count), is_static(ptr.is_static)
    {}

    bool operator<(const PtrData &o) const {
        // Variables are sorted by the CPU start address.
        // Overlapping memory ranges are considered equal.
        return (cpu_addr.start() < o.cpu_addr.start()) &&
               !cpu_addr.overlaps(o.cpu_addr);
    }

    long add_reference() {
        if (is_static) {
            return LONG_MAX;
        }
#ifndef TARGET_WINNT
        return __sync_fetch_and_add(&ref_count, 1);
#else // TARGET_WINNT
        return _InterlockedIncrement(&ref_count) - 1;
#endif // TARGET_WINNT
    }

    long remove_reference() {
        if (is_static) {
            return LONG_MAX;
        }
#ifndef TARGET_WINNT
        return __sync_sub_and_fetch(&ref_count, 1);
#else // TARGET_WINNT
        return _InterlockedDecrement(&ref_count);
#endif // TARGET_WINNT
    }

    long get_reference() const {
        if (is_static) {
            return LONG_MAX;
        }
        return ref_count;
    }

public:
    // CPU address range
    const MemRange  cpu_addr;

    // CPU and MIC buffers
    COIBUFFER       cpu_buf;
    COIBUFFER       mic_buf;

    // placeholder for buffer address on mic
    uint64_t        mic_addr;

    uint64_t        alloc_disp;

    // additional offset to pointer data on MIC for improving bandwidth for
    // data which is not 4K aligned
    uint32_t        mic_offset;

    // if true buffers are created from static memory
    bool            is_static;
    mutex_t         alloc_ptr_data_lock;

private:
    // reference count for the entry
    long            ref_count;
};

typedef std::list<PtrData*> PtrDataList;

// Data associated with automatic variable
class AutoData {
public:
    AutoData(const void *addr, uint64_t len) :
        cpu_addr(addr, len), ref_count(0)
    {}

    bool operator<(const AutoData &o) const {
        // Variables are sorted by the CPU start address.
        // Overlapping memory ranges are considered equal.
        return (cpu_addr.start() < o.cpu_addr.start()) &&
               !cpu_addr.overlaps(o.cpu_addr);
    }

    long add_reference() {
#ifndef TARGET_WINNT
        return __sync_fetch_and_add(&ref_count, 1);
#else // TARGET_WINNT
        return _InterlockedIncrement(&ref_count) - 1;
#endif // TARGET_WINNT
    }

    long remove_reference() {
#ifndef TARGET_WINNT
        return __sync_sub_and_fetch(&ref_count, 1);
#else // TARGET_WINNT
        return _InterlockedDecrement(&ref_count);
#endif // TARGET_WINNT
    }

    long get_reference() const {
        return ref_count;
    }

public:
    // CPU address range
    const MemRange cpu_addr;

private:
    // reference count for the entry
    long ref_count;
};

// Set of autimatic variables
typedef std::set<AutoData> AutoSet;

// Target image data
struct TargetImage
{
    TargetImage(const char *_name, const void *_data, uint64_t _size,
                const char *_origin, uint64_t _offset) :
        name(_name), data(_data), size(_size),
        origin(_origin), offset(_offset)
    {}

    // library name
    const char* name;

    // contents and size
    const void* data;
    uint64_t    size;

    // file of origin and offset within that file
    const char* origin;
    uint64_t    offset;
};

typedef std::list<TargetImage> TargetImageList;

// Data associated with persistent auto objects
struct PersistData
{
    PersistData(const void *addr, uint64_t routine_num, uint64_t size) :
        stack_cpu_addr(addr), routine_id(routine_num)
    {
        stack_ptr_data = new PtrData(0, size);
    }
    // 1-st key value - beginning of the stack at CPU
    const void *   stack_cpu_addr;
    // 2-nd key value - identifier of routine invocation at CPU
    uint64_t   routine_id;
    // corresponded PtrData; only stack_ptr_data->mic_buf is used
    PtrData * stack_ptr_data;
    // used to get offset of the variable in stack buffer
    char * cpu_stack_addr;
};

typedef std::list<PersistData> PersistDataList;

// class representing a single engine
struct Engine {
    friend void __offload_init_library_once(void);
    friend void __offload_fini_library(void);

#define check_result(res, tag, ...) \
    { \
        if (res == COI_PROCESS_DIED) { \
            fini_process(true); \
            exit(1); \
        } \
        if (res != COI_SUCCESS) { \
            __liboffload_error_support(tag, __VA_ARGS__); \
            exit(1); \
        } \
    }

    int get_logical_index() const {
        return m_index;
    }

    int get_physical_index() const {
        return m_physical_index;
    }

    const COIPROCESS& get_process() const {
        return m_process;
    }

    // initialize device
    void init(void);

    // add new library
    void add_lib(const TargetImage &lib)
    {
        m_lock.lock();
        m_ready = false;
        m_images.push_back(lib);
        m_lock.unlock();
    }

    COIRESULT compute(
        const std::list<COIBUFFER> &buffers,
        const void*         data,
        uint16_t            data_size,
        void*               ret,
        uint16_t            ret_size,
        uint32_t            num_deps,
        const COIEVENT*     deps,
        COIEVENT*           event
    );

#ifdef MYO_SUPPORT
    // temporary workaround for blocking behavior for myoiLibInit/Fini calls
    void init_myo(COIEVENT *event) {
        COIRESULT res;
        res = COI::PipelineRunFunction(get_pipeline(),
                                       m_funcs[c_func_myo_init],
                                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       event);
        check_result(res, c_pipeline_run_func, m_index, res);
    }

    void fini_myo(COIEVENT *event) {
        COIRESULT res;
        res = COI::PipelineRunFunction(get_pipeline(),
                                       m_funcs[c_func_myo_fini],
                                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       event);
        check_result(res, c_pipeline_run_func, m_index, res);
    }
#endif // MYO_SUPPORT

    //
    // Memory association table
    //
    PtrData* find_ptr_data(const void *ptr) {
        m_ptr_lock.lock();
        PtrSet::iterator res = m_ptr_set.find(PtrData(ptr, 0));
        m_ptr_lock.unlock();
        if (res == m_ptr_set.end()) {
            return 0;
        }
        return const_cast<PtrData*>(res.operator->());
    }

    PtrData* insert_ptr_data(const void *ptr, uint64_t len, bool &is_new) {
        m_ptr_lock.lock();
        std::pair<PtrSet::iterator, bool> res =
            m_ptr_set.insert(PtrData(ptr, len));
        PtrData* ptr_data = const_cast<PtrData*>(res.first.operator->());
        m_ptr_lock.unlock();

        is_new = res.second;
        if (is_new) {
            // It's necessary to lock as soon as possible.
            // unlock must be done at call site of insert_ptr_data at
            // branch for is_new
            ptr_data->alloc_ptr_data_lock.lock();
        }
        return ptr_data;
    }

    void remove_ptr_data(const void *ptr) {
        m_ptr_lock.lock();
        m_ptr_set.erase(PtrData(ptr, 0));
        m_ptr_lock.unlock();
    }

    //
    // Automatic variables
    //
    AutoData* find_auto_data(const void *ptr) {
        AutoSet &auto_vars = get_auto_vars();
        AutoSet::iterator res = auto_vars.find(AutoData(ptr, 0));
        if (res == auto_vars.end()) {
            return 0;
        }
        return const_cast<AutoData*>(res.operator->());
    }

    AutoData* insert_auto_data(const void *ptr, uint64_t len) {
        AutoSet &auto_vars = get_auto_vars();
        std::pair<AutoSet::iterator, bool> res =
            auto_vars.insert(AutoData(ptr, len));
        return const_cast<AutoData*>(res.first.operator->());
    }

    void remove_auto_data(const void *ptr) {
        get_auto_vars().erase(AutoData(ptr, 0));
    }

    //
    // Signals
    //
    void add_signal(const void *signal, OffloadDescriptor *desc) {
        m_signal_lock.lock();
        m_signal_map[signal] = desc;
        m_signal_lock.unlock();
    }

    OffloadDescriptor* find_signal(const void *signal, bool remove) {
        OffloadDescriptor *desc = 0;

        m_signal_lock.lock();
        {
            SignalMap::iterator it = m_signal_map.find(signal);
            if (it != m_signal_map.end()) {
                desc = it->second;
                if (remove) {
                    m_signal_map.erase(it);
                }
            }
        }
        m_signal_lock.unlock();

        return desc;
    }

    // stop device process
    void fini_process(bool verbose);

    // list of stacks active at the engine
    PersistDataList m_persist_list;

private:
    Engine() : m_index(-1), m_physical_index(-1), m_process(0), m_ready(false),
               m_proc_number(0)
    {}

    ~Engine() {
        if (m_process != 0) {
            fini_process(false);
        }
    }

    // set indexes
    void set_indexes(int logical_index, int physical_index) {
        m_index = logical_index;
        m_physical_index = physical_index;
    }

    // start process on device
    void init_process();

    void load_libraries(void);
    void init_ptr_data(void);

    // performs library intialization on the device side
    pid_t init_device(void);

private:
    // get pipeline associated with a calling thread
    COIPIPELINE get_pipeline(void);

    // get automatic vars set associated with the calling thread
    AutoSet& get_auto_vars(void);

    // destructor for thread data
    static void destroy_thread_data(void *data);

private:
    typedef std::set<PtrData> PtrSet;
    typedef std::map<const void*, OffloadDescriptor*> SignalMap;

    // device indexes
    int         m_index;
    int         m_physical_index;

    // number of COI pipes created for the engine
    long        m_proc_number;

    // process handle
    COIPROCESS  m_process;

    // If false, device either has not been initialized or new libraries
    // have been added.
    bool        m_ready;
    mutex_t     m_lock;

    // List of libraries to be loaded
    TargetImageList m_images;

    // var table
    PtrSet      m_ptr_set;
    mutex_t     m_ptr_lock;

    // signals
    SignalMap m_signal_map;
    mutex_t   m_signal_lock;

    // constants for accessing device function handles
    enum {
        c_func_compute = 0,
#ifdef MYO_SUPPORT
        c_func_myo_init,
        c_func_myo_fini,
#endif // MYO_SUPPORT
        c_func_init,
        c_func_var_table_size,
        c_func_var_table_copy,
        c_funcs_total
    };
    static const char* m_func_names[c_funcs_total];

    // device function handles
    COIFUNCTION m_funcs[c_funcs_total];

    // int -> name mapping for device signals
    static const int   c_signal_max = 32;
    static const char* c_signal_names[c_signal_max];
};

#endif // OFFLOAD_ENGINE_H_INCLUDED
