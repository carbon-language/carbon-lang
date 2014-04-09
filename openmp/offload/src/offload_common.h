//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


/*! \file
    \brief The parts of the runtime library common to host and target
*/

#ifndef OFFLOAD_COMMON_H_INCLUDED
#define OFFLOAD_COMMON_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>

#include "offload.h"
#include "offload_table.h"
#include "offload_trace.h"
#include "offload_timer.h"
#include "offload_util.h"
#include "cean_util.h"
#include "dv_util.h"
#include "liboffload_error_codes.h"

#include <stdarg.h>

// The debug routines

// Host console and file logging
extern int console_enabled;
extern int offload_report_level;

#define OFFLOAD_DO_TRACE (offload_report_level == 3)

extern const char *prefix;
extern int offload_number;
#if !HOST_LIBRARY
extern int mic_index;
#endif

#if HOST_LIBRARY
void Offload_Report_Prolog(OffloadHostTimerData* timer_data);
void Offload_Report_Epilog(OffloadHostTimerData* timer_data);
void offload_report_free_data(OffloadHostTimerData * timer_data);
void Offload_Timer_Print(void);

#ifndef TARGET_WINNT
#define OFFLOAD_DEBUG_INCR_OFLD_NUM() \
        __sync_add_and_fetch(&offload_number, 1)
#else
#define OFFLOAD_DEBUG_INCR_OFLD_NUM() \
        _InterlockedIncrement(reinterpret_cast<long*>(&offload_number))
#endif

#define OFFLOAD_DEBUG_PRINT_TAG_PREFIX() \
        printf("%s:  ", prefix);

#define OFFLOAD_DEBUG_PRINT_PREFIX() \
        printf("%s:  ", prefix);
#else
#define OFFLOAD_DEBUG_PRINT_PREFIX() \
        printf("%s%d:  ", prefix, mic_index);
#endif // HOST_LIBRARY

#define OFFLOAD_TRACE(trace_level, ...)  \
    if (console_enabled >= trace_level) { \
        OFFLOAD_DEBUG_PRINT_PREFIX(); \
        printf(__VA_ARGS__); \
        fflush(NULL); \
    }

#if OFFLOAD_DEBUG > 0

#define OFFLOAD_DEBUG_TRACE(level, ...) \
    OFFLOAD_TRACE(level, __VA_ARGS__)

#define OFFLOAD_REPORT(level, offload_number, stage, ...) \
    if (OFFLOAD_DO_TRACE) { \
        offload_stage_print(stage, offload_number, __VA_ARGS__); \
        fflush(NULL); \
    }

#define OFFLOAD_DEBUG_TRACE_1(level, offload_number, stage, ...) \
    if (OFFLOAD_DO_TRACE) { \
        offload_stage_print(stage, offload_number, __VA_ARGS__); \
        fflush(NULL); \
    } \
    if (!OFFLOAD_DO_TRACE) { \
        OFFLOAD_TRACE(level, __VA_ARGS__) \
    }

#define OFFLOAD_DEBUG_DUMP_BYTES(level, a, b) \
    __dump_bytes(level, a, b)

extern void __dump_bytes(
    int level,
    const void *data,
    int len
);

#else

#define OFFLOAD_DEBUG_LOG(level, ...)
#define OFFLOAD_DEBUG_DUMP_BYTES(level, a, b)

#endif

// Runtime interface

#define OFFLOAD_PREFIX(a) __offload_##a

#define OFFLOAD_MALLOC            OFFLOAD_PREFIX(malloc)
#define OFFLOAD_FREE(a)           _mm_free(a)

// Forward functions

extern void *OFFLOAD_MALLOC(size_t size, size_t align);

// The Marshaller

//! \enum Indicator for the type of entry on an offload item list.
enum OffloadItemType {
    c_data =   1,       //!< Plain data
    c_data_ptr,         //!< Pointer data
    c_func_ptr,         //!< Function pointer
    c_void_ptr,         //!< void*
    c_string_ptr,       //!< C string
    c_dv,               //!< Dope vector variable
    c_dv_data,          //!< Dope-vector data
    c_dv_data_slice,    //!< Dope-vector data's slice
    c_dv_ptr,           //!< Dope-vector variable pointer
    c_dv_ptr_data,      //!< Dope-vector pointer data
    c_dv_ptr_data_slice,//!< Dope-vector pointer data's slice
    c_cean_var,         //!< CEAN variable
    c_cean_var_ptr,     //!< Pointer to CEAN variable
    c_data_ptr_array,   //!< Pointer to data pointer array
    c_func_ptr_array,   //!< Pointer to function pointer array
    c_void_ptr_array,   //!< Pointer to void* pointer array
    c_string_ptr_array  //!< Pointer to char* pointer array
};

#define VAR_TYPE_IS_PTR(t) ((t) == c_string_ptr || \
                            (t) == c_data_ptr || \
                            (t) == c_cean_var_ptr || \
                            (t) == c_dv_ptr)

#define VAR_TYPE_IS_SCALAR(t) ((t) == c_data || \
                               (t) == c_void_ptr || \
                               (t) == c_cean_var || \
                               (t) == c_dv)

#define VAR_TYPE_IS_DV_DATA(t) ((t) == c_dv_data || \
                                (t) == c_dv_ptr_data)

#define VAR_TYPE_IS_DV_DATA_SLICE(t) ((t) == c_dv_data_slice || \
                                      (t) == c_dv_ptr_data_slice)


//! \enum Specify direction to copy offloaded variable.
enum OffloadParameterType {
    c_parameter_unknown = -1, //!< Unknown clause
    c_parameter_nocopy,       //!< Variable listed in "nocopy" clause
    c_parameter_in,           //!< Variable listed in "in" clause
    c_parameter_out,          //!< Variable listed in "out" clause
    c_parameter_inout         //!< Variable listed in "inout" clause
};

//! An Offload Variable descriptor
struct VarDesc {
    //! OffloadItemTypes of source and destination
    union {
        struct {
            uint8_t dst : 4; //!< OffloadItemType of destination
            uint8_t src : 4; //!< OffloadItemType of source
        };
        uint8_t bits;
    } type;

    //! OffloadParameterType that describes direction of data transfer
    union {
        struct {
            uint8_t in  : 1; //!< Set if IN or INOUT
            uint8_t out : 1; //!< Set if OUT or INOUT
        };
        uint8_t bits;
    } direction;

    uint8_t alloc_if;        //!< alloc_if modifier value
    uint8_t free_if;         //!< free_if modifier value
    uint32_t align;          //!< MIC alignment requested for pointer data
    //! Not used by compiler; set to 0
    /*! Used by runtime as offset to data from start of MIC buffer */
    uint32_t mic_offset;
    //! Flags describing this variable
    union {
        struct {
            //! source variable has persistent storage
            uint32_t is_static : 1;
            //! destination variable has persistent storage
            uint32_t is_static_dstn : 1;
            //! has length for c_dv && c_dv_ptr
            uint32_t has_length : 1;
            //! persisted local scalar is in stack buffer
            uint32_t is_stack_buf : 1;
            //! buffer address is sent in data
            uint32_t sink_addr : 1;
            //! alloc displacement is sent in data
            uint32_t alloc_disp : 1;
            //! source data is noncontiguous
            uint32_t is_noncont_src : 1;
            //! destination data is noncontiguous
            uint32_t is_noncont_dst : 1;
        };
        uint32_t bits;
    } flags;
    //! Not used by compiler; set to 0
    /*! Used by runtime as offset to base from data stored in a buffer */
    int64_t offset;
    //! Element byte-size of data to be transferred
    /*! For dope-vector, the size of the dope-vector      */
    int64_t size;
    union {
        //! Set to 0 for array expressions and dope-vectors
        /*! Set to 1 for scalars                          */
        /*! Set to value of length modifier for pointers  */
        int64_t count;
        //! Displacement not used by compiler
        int64_t disp;
    };

    //! This field not used by OpenMP 4.0
    /*! The alloc section expression in #pragma offload   */
    union {
       void *alloc;
       int64_t ptr_arr_offset;
    };

    //! This field not used by OpenMP 4.0
    /*! The into section expression in #pragma offload    */
    /*! For c_data_ptr_array this is the into ptr array   */
    void *into;

    //! For an ordinary variable, address of the variable
    /*! For c_cean_var (C/C++ array expression),
        pointer to arr_desc, which is an array descriptor. */
    /*! For c_data_ptr_array (array of data pointers),
        pointer to ptr_array_descriptor,
        which is a descriptor for pointer array transfers. */
    void *ptr;
};

//! Auxiliary struct used when -g is enabled that holds variable names
struct VarDesc2 {
    const char *sname; //!< Source name
    const char *dname; //!< Destination name (when "into" is used)
};

/*! When the OffloadItemType is c_data_ptr_array
    the ptr field of the main descriptor points to this struct.          */
/*! The type in VarDesc1 merely says c_cean_data_ptr, but the pointer
    type can be c_data_ptr, c_func_ptr, c_void_ptr, or c_string_ptr.
    Therefore the actual pointer type is in the flags field of VarDesc3. */
/*! If flag_align_is_array/flag_alloc_if_is_array/flag_free_if_is_array
    is 0 then alignment/alloc_if/free_if are specified in VarDesc1.      */
/*! If flag_align_is_array/flag_alloc_if_is_array/flag_free_if_is_array
    is 1 then align_array/alloc_if_array/free_if_array specify
    the set of alignment/alloc_if/free_if values.                        */
/*! For the other fields, if neither the scalar nor the array flag
    is set, then that modifier was not specified. If the bits are set
    they specify which modifier was set and whether it was a
    scalar or an array expression.                                       */
struct VarDesc3
{
    void *ptr_array;        //!< Pointer to arr_desc of array of pointers
    void *align_array;      //!< Scalar value or pointer to arr_desc
    void *alloc_if_array;   //!< Scalar value or pointer to arr_desc
    void *free_if_array;    //!< Scalar value or pointer to arr_desc
    void *extent_start;     //!< Scalar value or pointer to arr_desc
    void *extent_elements;  //!< Scalar value or pointer to arr_desc
    void *into_start;       //!< Scalar value or pointer to arr_desc
    void *into_elements;    //!< Scalar value or pointer to arr_desc
    void *alloc_start;      //!< Scalar value or pointer to arr_desc
    void *alloc_elements;   //!< Scalar value or pointer to arr_desc
    /*! Flags that describe the pointer type and whether each field
        is a scalar value or an array expression.        */
    /*! First 6 bits are pointer array element type:
        c_data_ptr, c_func_ptr, c_void_ptr, c_string_ptr */
    /*! Then single bits specify:                        */
    /*!     align_array is an array                      */
    /*!     alloc_if_array is an array                   */
    /*!     free_if_array is an array                    */
    /*!     extent_start is a scalar expression          */
    /*!     extent_start is an array expression          */
    /*!     extent_elements is a scalar expression       */
    /*!     extent_elements is an array expression       */
    /*!     into_start is a scalar expression            */
    /*!     into_start is an array expression            */
    /*!     into_elements is a scalar expression         */
    /*!     into_elements is an array expression         */
    /*!     alloc_start is a scalar expression           */
    /*!     alloc_start is an array expression           */
    /*!     alloc_elements is a scalar expression        */
    /*!     alloc_elements is an array expression        */
    uint32_t array_fields;
};
const int flag_align_is_array = 6;
const int flag_alloc_if_is_array = 7;
const int flag_free_if_is_array = 8;
const int flag_extent_start_is_scalar = 9;
const int flag_extent_start_is_array = 10;
const int flag_extent_elements_is_scalar = 11;
const int flag_extent_elements_is_array = 12;
const int flag_into_start_is_scalar = 13;
const int flag_into_start_is_array = 14;
const int flag_into_elements_is_scalar = 15;
const int flag_into_elements_is_array = 16;
const int flag_alloc_start_is_scalar = 17;
const int flag_alloc_start_is_array = 18;
const int flag_alloc_elements_is_scalar = 19;
const int flag_alloc_elements_is_array = 20;

// The Marshaller
class Marshaller
{
private:
    // Start address of buffer
    char *buffer_start;

    // Current pointer within buffer
    char *buffer_ptr;

    // Physical size of data sent (including flags)
    long long buffer_size;

    // User data sent/received
    long long tfr_size;

public:
    // Constructor
    Marshaller() :
        buffer_start(0), buffer_ptr(0),
        buffer_size(0), tfr_size(0)
    {
    }

    // Return count of user data sent/received
    long long get_tfr_size() const
    {
        return tfr_size;
    }

    // Return pointer to buffer
    char *get_buffer_start() const
    {
        return buffer_start;
    }

    // Return current size of data in buffer
    long long get_buffer_size() const
    {
        return buffer_size;
    }

    // Set buffer pointer
    void init_buffer(
        char *d,
        long long s
    )
    {
        buffer_start = buffer_ptr = d;
        buffer_size = s;
    }

    // Send data
    void send_data(
        const void *data,
        int64_t length
    );

    // Receive data
    void receive_data(
        void *data,
        int64_t length
    );

    // Send function pointer
    void send_func_ptr(
        const void* data
    );

    // Receive function pointer
    void receive_func_ptr(
        const void** data
    );
};

// End of the Marshaller

// The offloaded function descriptor.
// Sent from host to target to specify which function to run.
// Also, sets console and file tracing levels.
struct FunctionDescriptor
{
    // Input data size.
    long long in_datalen;

    // Output data size.
    long long out_datalen;

    // Whether trace is requested on console.
    // A value of 1 produces only function name and data sent/received.
    // Values > 1 produce copious trace information.
    uint8_t console_enabled;

    // Flag controlling timing on the target side.
    // Values > 0 enable timing on sink.
    uint8_t timer_enabled;

    int offload_report_level;
    int offload_number;

    // number of variable descriptors
    int vars_num;

    // inout data offset if data is passed as misc/return data
    // otherwise it should be zero.
    int data_offset;

    // The name of the offloaded function
    char data[];
};

// typedef OFFLOAD.
// Pointer to OffloadDescriptor.
typedef struct OffloadDescriptor *OFFLOAD;

#endif // OFFLOAD_COMMON_H_INCLUDED
