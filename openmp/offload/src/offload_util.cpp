//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_util.h"
#include <errno.h>
#include "liboffload_error_codes.h"

#ifdef TARGET_WINNT
void *thread_getspecific(pthread_key_t key)
{
    if (key == 0) {
        return NULL;
    }
    else {
        return TlsGetValue(key);
    }
}

int thread_setspecific(pthread_key_t key, const void *value)
{
    return (TlsSetValue(key, (LPVOID)value)) ? 0 : GetLastError();
}
#endif // TARGET_WINNT

bool __offload_parse_size_string(const char *str, uint64_t &new_size)
{
    uint64_t val;
    char *suffix;

    errno = 0;
#ifdef TARGET_WINNT
    val = strtoul(str, &suffix, 10);
#else // TARGET_WINNT
    val = strtoull(str, &suffix, 10);
#endif // TARGET_WINNT
    if (errno != 0 || suffix == str) {
        return false;
    }

    if (suffix[0] == '\0') {
        // default is Kilobytes
        new_size = val * 1024;
        return true;
    }
    else if (suffix[1] == '\0') {
        // Optional suffixes: B (bytes), K (Kilobytes), M (Megabytes),
        // G (Gigabytes), or T (Terabytes) specify the units.
        switch (suffix[0]) {
            case 'b':
            case 'B':
                new_size = val;
                break;

            case 'k':
            case 'K':
                new_size = val * 1024;
                break;

            case 'm':
            case 'M':
                new_size = val * 1024 * 1024;
                break;

            case 'g':
            case 'G':
                new_size = val * 1024 * 1024 * 1024;
                break;

            case 't':
            case 'T':
                new_size = val * 1024 * 1024 * 1024 * 1024;
                break;

            default:
                return false;
        }
        return true;
    }

    return false;
}

bool __offload_parse_int_string(const char *str, int64_t &value)
{
    int64_t val;
    char *suffix;

    errno = 0;
#ifdef TARGET_WINNT
    val = strtol(str, &suffix, 0);
#else
    val = strtoll(str, &suffix, 0);
#endif
    if (errno == 0 && suffix != str && *suffix == '\0') {
        value = val;
        return true;
    }
    return false;
}

#ifdef TARGET_WINNT
extern void* DL_open(const char *path)
{
    void *handle;
    int error_mode;

    /*
     * do not display message box with error if it the call below fails to
     * load dynamic library.
     */
    error_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);

    /* load dynamic library */
    handle = (void*) LoadLibrary(path);

    /* restore error mode */
    SetErrorMode(error_mode);

    return handle;
}

extern int DL_addr(const void *addr, Dl_info *dl_info)
{
    MEMORY_BASIC_INFORMATION mem_info;
    char mod_name[MAX_PATH];
    HMODULE mod_handle;

    /* Fill MEMORY_BASIC_INFORMATION struct */
    if (!VirtualQuery(addr, &mem_info, sizeof(mem_info))) {
        return 0;
    }
    mod_handle = (HMODULE)mem_info.AllocationBase;

    /* ANSI file name for module */
    if (!GetModuleFileNameA(mod_handle, (char*) mod_name, sizeof(mod_name))) {
        return 0;
    }
    strcpy(dl_info->dli_fname, mod_name);
    dl_info->dli_fbase = mem_info.BaseAddress;
    dl_info->dli_saddr = addr;
    strcpy(dl_info->dli_sname, mod_name);
    return 1;
}

// Run once
static BOOL CALLBACK __offload_run_once_wrapper(
    PINIT_ONCE initOnce,
    PVOID parameter,
    PVOID *context
)
{
    void (*init_routine)(void) = (void(*)(void)) parameter;
    init_routine();
    return true;
}

void __offload_run_once(OffloadOnceControl *ctrl, void (*func)(void))
{
    InitOnceExecuteOnce(ctrl, __offload_run_once_wrapper, (void*) func, 0);
}
#endif // TARGET_WINNT

/* ARGSUSED */ // version is not used on windows
void* DL_sym(void *handle, const char *name, const char *version)
{
#ifdef TARGET_WINNT
    return GetProcAddress((HMODULE) handle, name);
#else // TARGET_WINNT
    if (version == 0) {
        return dlsym(handle, name);
    }
    else {
        return dlvsym(handle, name, version);
    }
#endif // TARGET_WINNT
}

int64_t get_el_value(
                     char *base,
                     int64_t offset,
                     int64_t size)
{
    int64_t val = 0;
    switch (size) {
        case 1:
            val = static_cast<int64_t>(*((char *)(base + offset)));
            break;
        case 2:
            val = static_cast<int64_t>(*((short *)(base + offset)));
            break;
        case 4:
            val = static_cast<int64_t>(*((int *)(base + offset)));
            break;
        default:
            val = *((int64_t *)(base + offset));
            break;
    }
    return val;
}
