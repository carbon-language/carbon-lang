//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_myo_host.h"
#include <errno.h>
#include <malloc.h>
#include "offload_host.h"

#if defined(LINUX) || defined(FREEBSD)
#include <mm_malloc.h>
#endif

#define MYO_VERSION1    "MYO_1.0"

extern "C" void __cilkrts_cilk_for_32(void*, void*, uint32_t, int32_t);
extern "C" void __cilkrts_cilk_for_64(void*, void*, uint64_t, int32_t);

#ifndef TARGET_WINNT
#pragma weak __cilkrts_cilk_for_32
#pragma weak __cilkrts_cilk_for_64
#endif // TARGET_WINNT

#ifdef TARGET_WINNT
#define MYO_TABLE_END_MARKER() reinterpret_cast<const char*>(-1)
#else // TARGET_WINNT
#define MYO_TABLE_END_MARKER() reinterpret_cast<const char*>(0)
#endif // TARGET_WINNT

class MyoWrapper {
public:
    MyoWrapper() : m_lib_handle(0), m_is_available(false)
    {}

    bool is_available() const {
        return m_is_available;
    }

    bool LoadLibrary(void);

    // unloads the library
    void UnloadLibrary(void) {
//        if (m_lib_handle != 0) {
//            DL_close(m_lib_handle);
//            m_lib_handle = 0;
//        }
    }

    // Wrappers for MYO client functions
    void LibInit(void *arg, void *func) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myoinit,
                                 "%s(%p, %p)\n", __func__, arg, func);
        CheckResult(__func__, m_lib_init(arg, func));
    }

    void LibFini(void) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myofini, "%s()\n", __func__);
        m_lib_fini();
    }

    void* SharedMalloc(size_t size) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myosharedmalloc,
                                 "%s(%lld)\n", __func__, size);
        return m_shared_malloc(size);
    }

    void SharedFree(void *ptr) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myosharedfree,
                                 "%s(%p)\n", __func__, ptr);
        m_shared_free(ptr);
    }

    void* SharedAlignedMalloc(size_t size, size_t align) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myosharedalignedmalloc,
                                 "%s(%lld, %lld)\n", __func__, size, align);
        return m_shared_aligned_malloc(size, align);
    }

    void SharedAlignedFree(void *ptr) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myosharedalignedfree,
                              "%s(%p)\n", __func__, ptr);
        m_shared_aligned_free(ptr);
    }

    void Acquire(void) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myoacquire,
                              "%s()\n", __func__);
        CheckResult(__func__, m_acquire());
    }

    void Release(void) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myorelease,
                            "%s()\n", __func__);
        CheckResult(__func__, m_release());
    }

    void HostVarTablePropagate(void *table, int num_entries) const {
        OFFLOAD_DEBUG_TRACE(4, "%s(%p, %d)\n", __func__, table, num_entries);
        CheckResult(__func__, m_host_var_table_propagate(table, num_entries));
    }

    void HostFptrTableRegister(void *table, int num_entries,
                               int ordered) const {
        OFFLOAD_DEBUG_TRACE_1(4, 0, c_offload_myoregister,
                            "%s(%p, %d, %d)\n", __func__, table,
                            num_entries, ordered);
        CheckResult(__func__,
                    m_host_fptr_table_register(table, num_entries, ordered));
    }

    void RemoteThunkCall(void *thunk, void *args, int device) {
        OFFLOAD_DEBUG_TRACE(4, "%s(%p, %p, %d)\n", __func__, thunk, args,
                            device);
        CheckResult(__func__, m_remote_thunk_call(thunk, args, device));
    }

    MyoiRFuncCallHandle RemoteCall(char *func, void *args, int device) const {
        OFFLOAD_DEBUG_TRACE(4, "%s(%s, %p, %d)\n", __func__, func, args,
                            device);
        return m_remote_call(func, args, device);
    }

    void GetResult(MyoiRFuncCallHandle handle) const {
        OFFLOAD_DEBUG_TRACE(4, "%s(%p)\n", __func__, handle);
        CheckResult(__func__, m_get_result(handle));
    }

private:
    void CheckResult(const char *func, MyoError error) const {
        if (error != MYO_SUCCESS) {
             LIBOFFLOAD_ERROR(c_myowrapper_checkresult, func, error);
            exit(1);
        }
    }

private:
    void* m_lib_handle;
    bool  m_is_available;

    // pointers to functions from myo library
    MyoError (*m_lib_init)(void*, void*);
    void     (*m_lib_fini)(void);
    void*    (*m_shared_malloc)(size_t);
    void     (*m_shared_free)(void*);
    void*    (*m_shared_aligned_malloc)(size_t, size_t);
    void     (*m_shared_aligned_free)(void*);
    MyoError (*m_acquire)(void);
    MyoError (*m_release)(void);
    MyoError (*m_host_var_table_propagate)(void*, int);
    MyoError (*m_host_fptr_table_register)(void*, int, int);
    MyoError (*m_remote_thunk_call)(void*, void*, int);
    MyoiRFuncCallHandle (*m_remote_call)(char*, void*, int);
    MyoError (*m_get_result)(MyoiRFuncCallHandle);
};

bool MyoWrapper::LoadLibrary(void)
{
#ifndef TARGET_WINNT
    const char *lib_name = "libmyo-client.so";
#else // TARGET_WINNT
    const char *lib_name = "myo-client.dll";
#endif // TARGET_WINNT

    OFFLOAD_DEBUG_TRACE(2, "Loading MYO library %s ...\n", lib_name);

    m_lib_handle = DL_open(lib_name);
    if (m_lib_handle == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to load the library. errno = %d\n",
                            errno);
        return false;
    }

    m_lib_init = (MyoError (*)(void*, void*))
        DL_sym(m_lib_handle, "myoiLibInit", MYO_VERSION1);
    if (m_lib_init == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoiLibInit");
        UnloadLibrary();
        return false;
    }

    m_lib_fini = (void (*)(void))
        DL_sym(m_lib_handle, "myoiLibFini", MYO_VERSION1);
    if (m_lib_fini == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoiLibFini");
        UnloadLibrary();
        return false;
    }

    m_shared_malloc = (void* (*)(size_t))
        DL_sym(m_lib_handle, "myoSharedMalloc", MYO_VERSION1);
    if (m_shared_malloc == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoSharedMalloc");
        UnloadLibrary();
        return false;
    }

    m_shared_free = (void (*)(void*))
        DL_sym(m_lib_handle, "myoSharedFree", MYO_VERSION1);
    if (m_shared_free == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoSharedFree");
        UnloadLibrary();
        return false;
    }

    m_shared_aligned_malloc = (void* (*)(size_t, size_t))
        DL_sym(m_lib_handle, "myoSharedAlignedMalloc", MYO_VERSION1);
    if (m_shared_aligned_malloc == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoSharedAlignedMalloc");
        UnloadLibrary();
        return false;
    }

    m_shared_aligned_free = (void (*)(void*))
        DL_sym(m_lib_handle, "myoSharedAlignedFree", MYO_VERSION1);
    if (m_shared_aligned_free == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoSharedAlignedFree");
        UnloadLibrary();
        return false;
    }

    m_acquire = (MyoError (*)(void))
        DL_sym(m_lib_handle, "myoAcquire", MYO_VERSION1);
    if (m_acquire == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoAcquire");
        UnloadLibrary();
        return false;
    }

    m_release = (MyoError (*)(void))
        DL_sym(m_lib_handle, "myoRelease", MYO_VERSION1);
    if (m_release == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoRelease");
        UnloadLibrary();
        return false;
    }

    m_host_var_table_propagate = (MyoError (*)(void*, int))
        DL_sym(m_lib_handle, "myoiHostVarTablePropagate", MYO_VERSION1);
    if (m_host_var_table_propagate == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoiHostVarTablePropagate");
        UnloadLibrary();
        return false;
    }

    m_host_fptr_table_register = (MyoError (*)(void*, int, int))
        DL_sym(m_lib_handle, "myoiHostFptrTableRegister", MYO_VERSION1);
    if (m_host_fptr_table_register == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoiHostFptrTableRegister");
        UnloadLibrary();
        return false;
    }

    m_remote_thunk_call = (MyoError (*)(void*, void*, int))
        DL_sym(m_lib_handle, "myoiRemoteThunkCall", MYO_VERSION1);
    if (m_remote_thunk_call == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoiRemoteThunkCall");
        UnloadLibrary();
        return false;
    }

    m_remote_call = (MyoiRFuncCallHandle (*)(char*, void*, int))
        DL_sym(m_lib_handle, "myoiRemoteCall", MYO_VERSION1);
    if (m_remote_call == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoiRemoteCall");
        UnloadLibrary();
        return false;
    }

    m_get_result = (MyoError (*)(MyoiRFuncCallHandle))
        DL_sym(m_lib_handle, "myoiGetResult", MYO_VERSION1);
    if (m_get_result == 0) {
        OFFLOAD_DEBUG_TRACE(2, "Failed to find %s in MYO library\n",
                            "myoiGetResult");
        UnloadLibrary();
        return false;
    }

    OFFLOAD_DEBUG_TRACE(2, "The library was successfully loaded\n");

    m_is_available = true;

    return true;
}

static bool myo_is_available;
static MyoWrapper myo_wrapper;

struct MyoTable
{
    MyoTable(SharedTableEntry *tab, int len) : var_tab(tab), var_tab_len(len)
    {}

    SharedTableEntry*   var_tab;
    int                 var_tab_len;
};

typedef std::list<MyoTable> MyoTableList;
static MyoTableList __myo_table_list;
static mutex_t      __myo_table_lock;
static bool         __myo_tables = false;

static void __offload_myo_shared_table_register(SharedTableEntry *entry);
static void __offload_myo_shared_init_table_register(InitTableEntry* entry);
static void __offload_myo_fptr_table_register(FptrTableEntry *entry);

static void __offload_myoLoadLibrary_once(void)
{
    if (__offload_init_library()) {
        myo_wrapper.LoadLibrary();
    }
}

static bool __offload_myoLoadLibrary(void)
{
    static OffloadOnceControl ctrl = OFFLOAD_ONCE_CONTROL_INIT;
    __offload_run_once(&ctrl, __offload_myoLoadLibrary_once);

    return myo_wrapper.is_available();
}

static void __offload_myoInit_once(void)
{
    if (!__offload_myoLoadLibrary()) {
        return;
    }

    // initialize all devices
    for (int i = 0; i < mic_engines_total; i++) {
        mic_engines[i].init();
    }

    // load and initialize MYO library
    OFFLOAD_DEBUG_TRACE(2, "Initializing MYO library ...\n");

    COIEVENT events[MIC_ENGINES_MAX];
    MyoiUserParams params[MIC_ENGINES_MAX+1];

    // load target library to all devices
    for (int i = 0; i < mic_engines_total; i++) {
        mic_engines[i].init_myo(&events[i]);

        params[i].type = MYOI_USERPARAMS_DEVID;
        params[i].nodeid = mic_engines[i].get_physical_index() + 1;
    }

    params[mic_engines_total].type = MYOI_USERPARAMS_LAST_MSG;

    // initialize myo runtime on host
    myo_wrapper.LibInit(params, 0);

    // wait for the target init calls to finish
    COIRESULT res;
    res = COI::EventWait(mic_engines_total, events, -1, 1, 0, 0);
    if (res != COI_SUCCESS) {
        LIBOFFLOAD_ERROR(c_event_wait, res);
        exit(1);
    }

    myo_is_available = true;

    OFFLOAD_DEBUG_TRACE(2, "Initializing MYO library ... done\n");
}

static bool __offload_myoInit(void)
{
    static OffloadOnceControl ctrl = OFFLOAD_ONCE_CONTROL_INIT;
    __offload_run_once(&ctrl, __offload_myoInit_once);

    // register pending shared var tables
    if (myo_is_available && __myo_tables) {
        mutex_locker_t locker(__myo_table_lock);

        if (__myo_tables) {
            //  Register tables with MYO so it can propagate to target.
            for(MyoTableList::const_iterator it = __myo_table_list.begin();
                it != __myo_table_list.end(); ++it) {
#ifdef TARGET_WINNT
                for (SharedTableEntry *entry = it->var_tab;
                     entry->varName != MYO_TABLE_END_MARKER(); entry++) {
                    if (entry->varName == 0) {
                        continue;
                    }
                    myo_wrapper.HostVarTablePropagate(entry, 1);
                }
#else // TARGET_WINNT
                myo_wrapper.HostVarTablePropagate(it->var_tab,
                                                  it->var_tab_len);
#endif // TARGET_WINNT
            }

            __myo_table_list.clear();
            __myo_tables = false;
        }
    }

    return myo_is_available;
}

static bool shared_table_entries(
    SharedTableEntry *entry
)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p)\n", __func__, entry);

    for (; entry->varName != MYO_TABLE_END_MARKER(); entry++) {
#ifdef TARGET_WINNT
        if (entry->varName == 0) {
            continue;
        }
#endif // TARGET_WINNT

        return true;
    }

    return false;
}

static bool fptr_table_entries(
    FptrTableEntry *entry
)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p)\n", __func__, entry);

    for (; entry->funcName != MYO_TABLE_END_MARKER(); entry++) {
#ifdef TARGET_WINNT
        if (entry->funcName == 0) {
            continue;
        }
#endif // TARGET_WINNT

        return true;
    }

    return false;
}

extern "C" void __offload_myoRegisterTables(
    InitTableEntry* init_table,
    SharedTableEntry *shared_table,
    FptrTableEntry *fptr_table
)
{
    // check whether we need to initialize MYO library. It is
    // initialized only if at least one myo table is not empty
    if (shared_table_entries(shared_table) || fptr_table_entries(fptr_table)) {
        // make sure myo library is loaded
        __offload_myoLoadLibrary();

        // register tables
        __offload_myo_shared_table_register(shared_table);
        __offload_myo_fptr_table_register(fptr_table);
        __offload_myo_shared_init_table_register(init_table);
    }
}

void __offload_myoFini(void)
{
    if (myo_is_available) {
        OFFLOAD_DEBUG_TRACE(3, "%s\n", __func__);

        COIEVENT events[MIC_ENGINES_MAX];

        // kick off myoiLibFini calls on all devices
        for (int i = 0; i < mic_engines_total; i++) {
            mic_engines[i].fini_myo(&events[i]);
        }

        // cleanup myo runtime on host
        myo_wrapper.LibFini();

        // wait for the target fini calls to finish
        COIRESULT res;
        res = COI::EventWait(mic_engines_total, events, -1, 1, 0, 0);
        if (res != COI_SUCCESS) {
            LIBOFFLOAD_ERROR(c_event_wait, res);
            exit(1);
        }
    }
}

static void __offload_myo_shared_table_register(
    SharedTableEntry *entry
)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p)\n", __func__, entry);

    SharedTableEntry *start = entry;
    int entries = 0;

    // allocate shared memory for vars
    for (; entry->varName != MYO_TABLE_END_MARKER(); entry++) {
#ifdef TARGET_WINNT
        if (entry->varName == 0) {
            OFFLOAD_DEBUG_TRACE(4, "skip registering a NULL MyoSharedTable entry\n");
            continue;
        }
#endif // TARGET_WINNT

        OFFLOAD_DEBUG_TRACE(4, "registering MyoSharedTable entry for %s @%p\n",
                            entry->varName, entry);

        // Invoke the function to create shared memory
        reinterpret_cast<void(*)(void)>(entry->sharedAddr)();
        entries++;
    }

    // and table to the list if it is not empty
    if (entries > 0) {
        mutex_locker_t locker(__myo_table_lock);
        __myo_table_list.push_back(MyoTable(start, entries));
        __myo_tables = true;
    }
}

static void __offload_myo_shared_init_table_register(InitTableEntry* entry)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p)\n", __func__, entry);

#ifdef TARGET_WINNT
    for (; entry->funcName != MYO_TABLE_END_MARKER(); entry++) {
        if (entry->funcName == 0) {
            OFFLOAD_DEBUG_TRACE(4, "skip registering a NULL MyoSharedInit entry\n");
            continue;
        }

        //  Invoke the function to init the shared memory
        entry->func();
    }
#else // TARGET_WINNT
    for (; entry->func != 0; entry++) {
        // Invoke the function to init the shared memory
        entry->func();
    }
#endif // TARGET_WINNT
}

static void __offload_myo_fptr_table_register(
    FptrTableEntry *entry
)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p)\n", __func__, entry);

    FptrTableEntry *start = entry;
    int entries = 0;

    for (; entry->funcName != MYO_TABLE_END_MARKER(); entry++) {
#ifdef TARGET_WINNT
        if (entry->funcName == 0) {
            OFFLOAD_DEBUG_TRACE(4, "skip registering a NULL MyoFptrTable entry\n");
            continue;
        }
#endif // TARGET_WINNT

        if (!myo_wrapper.is_available()) {
            *(static_cast<void**>(entry->localThunkAddr)) = entry->funcAddr;
        }

        OFFLOAD_DEBUG_TRACE(4, "registering MyoFptrTable entry for %s @%p\n",
                            entry->funcName, entry);

#ifdef TARGET_WINNT
        if (myo_wrapper.is_available()) {
            myo_wrapper.HostFptrTableRegister(entry, 1, false);
        }
#endif // TARGET_WINNT

        entries++;
    }

#ifndef TARGET_WINNT
    if (myo_wrapper.is_available() && entries > 0) {
        myo_wrapper.HostFptrTableRegister(start, entries, false);
    }
#endif // TARGET_WINNT
}

extern "C" int __offload_myoIsAvailable(int target_number)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%d)\n", __func__, target_number);

    if (target_number >= -2) {
        bool is_default_number = (target_number == -2);

        if (__offload_myoInit()) {
            if (target_number >= 0) {
                // User provided the device number
                int num = target_number % mic_engines_total;

                // reserve device in ORSL
                target_number = ORSL::reserve(num) ? num : -1;
            }
            else {
                // try to use device 0
                target_number = ORSL::reserve(0) ? 0 : -1;
            }

            // make sure device is initialized
            if (target_number >= 0) {
                mic_engines[target_number].init();
            }
        }
        else {
            // fallback to CPU
            target_number = -1;
        }

        if (target_number < 0 && !is_default_number) {
            LIBOFFLOAD_ERROR(c_device_is_not_available);
            exit(1);
        }
    }
    else {
        LIBOFFLOAD_ERROR(c_invalid_device_number);
        exit(1);
    }

    return target_number;
}

extern "C" void __offload_myoiRemoteIThunkCall(
    void *thunk,
    void *arg,
    int target_number
)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p, %p, %d)\n", __func__, thunk, arg,
                        target_number);

    myo_wrapper.Release();
    myo_wrapper.RemoteThunkCall(thunk, arg, target_number);
    myo_wrapper.Acquire();

    ORSL::release(target_number);
}

extern "C" void* _Offload_shared_malloc(size_t size)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%lld)\n", __func__, size);

    if (__offload_myoLoadLibrary()) {
        return myo_wrapper.SharedMalloc(size);
    }
    else {
        return malloc(size);
    }
}

extern "C" void _Offload_shared_free(void *ptr)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p)\n", __func__, ptr);

    if (__offload_myoLoadLibrary()) {
        myo_wrapper.SharedFree(ptr);
    }
    else {
        free(ptr);
    }
}

extern "C" void* _Offload_shared_aligned_malloc(size_t size, size_t align)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%lld, %lld)\n", __func__, size, align);

    if (__offload_myoLoadLibrary()) {
        return myo_wrapper.SharedAlignedMalloc(size, align);
    }
    else {
        if (align < sizeof(void*)) {
            align = sizeof(void*);
        }
        return _mm_malloc(size, align);
    }
}

extern "C" void _Offload_shared_aligned_free(void *ptr)
{
    OFFLOAD_DEBUG_TRACE(3, "%s(%p)\n", __func__, ptr);

    if (__offload_myoLoadLibrary()) {
        myo_wrapper.SharedAlignedFree(ptr);
    }
    else {
        _mm_free(ptr);
    }
}

extern "C" void __intel_cilk_for_32_offload(
    int size,
    void (*copy_constructor)(void*, void*),
    int target_number,
    void *raddr,
    void *closure_object,
    unsigned int iters,
    unsigned int grain_size)
{
    OFFLOAD_DEBUG_TRACE(3, "%s\n", __func__);

    target_number = __offload_myoIsAvailable(target_number);
    if (target_number >= 0) {
        struct S {
            void *M1;
            unsigned int M2;
            unsigned int M3;
            char closure[];
        } *args;

        args = (struct S*) _Offload_shared_malloc(sizeof(struct S) + size);
        args->M1 = raddr;
        args->M2 = iters;
        args->M3 = grain_size;

        if (copy_constructor == 0) {
            memcpy(args->closure, closure_object, size);
        }
        else {
            copy_constructor(args->closure, closure_object);
        }

        myo_wrapper.Release();
        myo_wrapper.GetResult(
            myo_wrapper.RemoteCall("__intel_cilk_for_32_offload",
                                   args, target_number)
        );
        myo_wrapper.Acquire();

        _Offload_shared_free(args);

        ORSL::release(target_number);
    }
    else {
        __cilkrts_cilk_for_32(raddr,
                              closure_object,
                              iters,
                              grain_size);
    }
}

extern "C" void __intel_cilk_for_64_offload(
    int size,
    void (*copy_constructor)(void*, void*),
    int target_number,
    void *raddr,
    void *closure_object,
    uint64_t iters,
    uint64_t grain_size)
{
    OFFLOAD_DEBUG_TRACE(3, "%s\n", __func__);

    target_number = __offload_myoIsAvailable(target_number);
    if (target_number >= 0) {
        struct S {
            void *M1;
            uint64_t M2;
            uint64_t M3;
            char closure[];
        } *args;

        args = (struct S*) _Offload_shared_malloc(sizeof(struct S) + size);
        args->M1 = raddr;
        args->M2 = iters;
        args->M3 = grain_size;

        if (copy_constructor == 0) {
            memcpy(args->closure, closure_object, size);
        }
        else {
            copy_constructor(args->closure, closure_object);
        }

        myo_wrapper.Release();
        myo_wrapper.GetResult(
            myo_wrapper.RemoteCall("__intel_cilk_for_64_offload", args,
                                   target_number)
        );
        myo_wrapper.Acquire();

        _Offload_shared_free(args);

        ORSL::release(target_number);
    }
    else {
        __cilkrts_cilk_for_64(raddr,
                              closure_object,
                              iters,
                              grain_size);
    }
}
