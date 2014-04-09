//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#if defined(LINUX) || defined(FREEBSD)
#include <mm_malloc.h>
#endif

#include "offload_common.h"

// The debug routines

#if OFFLOAD_DEBUG > 0

void __dump_bytes(
    int trace_level,
    const void *data,
    int len
)
{
    if (console_enabled > trace_level) {
        const uint8_t *arr = (const uint8_t*) data;
        char buffer[4096];
        char *bufferp;
        int count = 0;

        bufferp = buffer;
        while (len--) {
            sprintf(bufferp, "%02x", *arr++);
            bufferp += 2;
            count++;
            if ((count&3) == 0) {
                sprintf(bufferp, " ");
                bufferp++;
            }
            if ((count&63) == 0) {
                OFFLOAD_DEBUG_TRACE(trace_level, "%s\n", buffer);
                bufferp = buffer;
                count = 0;
            }
        }
        if (count) {
            OFFLOAD_DEBUG_TRACE(trace_level, "%s\n", buffer);
        }
    }
}
#endif // OFFLOAD_DEBUG

// The Marshaller and associated routines

void Marshaller::send_data(
    const void *data,
    int64_t length
)
{
    OFFLOAD_DEBUG_TRACE(2, "send_data(%p, %lld)\n",
                        data, length);
    memcpy(buffer_ptr, data, (size_t)length);
    buffer_ptr += length;
    tfr_size += length;
}

void Marshaller::receive_data(
    void *data,
    int64_t length
)
{
    OFFLOAD_DEBUG_TRACE(2, "receive_data(%p, %lld)\n",
                        data, length);
    memcpy(data, buffer_ptr, (size_t)length);
    buffer_ptr += length;
    tfr_size += length;
}

// Send function pointer
void Marshaller::send_func_ptr(
    const void* data
)
{
    const char* name;
    size_t      length;

    if (data != 0) {
        name = __offload_funcs.find_name(data);
        if (name == 0) {
#if OFFLOAD_DEBUG > 0
            if (console_enabled > 2) {
                __offload_funcs.dump();
            }
#endif // OFFLOAD_DEBUG > 0

            LIBOFFLOAD_ERROR(c_send_func_ptr, data);
            exit(1);
        }
        length = strlen(name) + 1;
    }
    else {
        name = "";
        length = 1;
    }

    memcpy(buffer_ptr, name, length);
    buffer_ptr += length;
    tfr_size += length;
}

// Receive function pointer
void Marshaller::receive_func_ptr(
    const void** data
)
{
    const char* name;
    size_t      length;

    name = (const char*) buffer_ptr;
    if (name[0] != '\0') {
        *data = __offload_funcs.find_addr(name);
        if (*data == 0) {
#if OFFLOAD_DEBUG > 0
            if (console_enabled > 2) {
                __offload_funcs.dump();
            }
#endif // OFFLOAD_DEBUG > 0

            LIBOFFLOAD_ERROR(c_receive_func_ptr, name);
            exit(1);
        }
        length = strlen(name) + 1;
    }
    else {
        *data = 0;
        length = 1;
    }

    buffer_ptr += length;
    tfr_size += length;
}

// End of the Marshaller and associated routines

extern void *OFFLOAD_MALLOC(
    size_t size,
    size_t align
)
{
    void *ptr;
    int   err;

    OFFLOAD_DEBUG_TRACE(2, "%s(%lld, %lld)\n", __func__, size, align);

    if (align < sizeof(void*)) {
        align = sizeof(void*);
    }

    ptr = _mm_malloc(size, align);
    if (ptr == NULL) {
        LIBOFFLOAD_ERROR(c_offload_malloc, size, align);
        exit(1);
    }

    OFFLOAD_DEBUG_TRACE(2, "%s returned %p\n", __func__, ptr);

    return ptr;
}
