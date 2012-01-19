/*===---- unwind.h - Stack unwinding ----------------------------------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

/* See "Data Definitions for libgcc_s" in the Linux Standard Base.*/

#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && \
    __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >=  1070
/* Starting in OS X Lion the SDK includes a unwind.h. We should just use it. */
#include_next <unwind.h>
#else

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* It is a bit strange for a header to play with the visibility of the
   symbols it declares, but this matches gcc's behavior and some programs
   depend on it */
#pragma GCC visibility push(default)

struct _Unwind_Context;
typedef enum {
  _URC_NO_REASON = 0,
  _URC_FOREIGN_EXCEPTION_CAUGHT = 1,

  _URC_FATAL_PHASE2_ERROR = 2,
  _URC_FATAL_PHASE1_ERROR = 3,
  _URC_NORMAL_STOP = 4,

  _URC_END_OF_STACK = 5,
  _URC_HANDLER_FOUND = 6,
  _URC_INSTALL_CONTEXT = 7,
  _URC_CONTINUE_UNWIND = 8
} _Unwind_Reason_Code;


#ifdef __arm__

typedef enum { 
  _UVRSC_CORE = 0,        /* integer register */ 
  _UVRSC_VFP = 1,         /* vfp */ 
  _UVRSC_WMMXD = 3,       /* Intel WMMX data register */ 
  _UVRSC_WMMXC = 4        /* Intel WMMX control register */ 
} _Unwind_VRS_RegClass; 

typedef enum { 
  _UVRSD_UINT32 = 0,  
  _UVRSD_VFPX = 1,  
  _UVRSD_UINT64 = 3,  
  _UVRSD_FLOAT = 4,  
  _UVRSD_DOUBLE = 5 
} _Unwind_VRS_DataRepresentation; 

typedef enum { 
  _UVRSR_OK = 0,  
  _UVRSR_NOT_IMPLEMENTED = 1,  
  _UVRSR_FAILED = 2  
} _Unwind_VRS_Result; 

_Unwind_VRS_Result _Unwind_VRS_Get(_Unwind_Context *context,
  _Unwind_VRS_RegClass regclass,
  uint32_t regno,
  _Unwind_VRS_DataRepresentation representation,
  void *valuep);

#else

uintptr_t _Unwind_GetIP(struct _Unwind_Context* context);

#endif

typedef _Unwind_Reason_Code (*_Unwind_Trace_Fn)(struct _Unwind_Context*, void*);
_Unwind_Reason_Code _Unwind_Backtrace(_Unwind_Trace_Fn, void*);

#pragma GCC visibility pop

#ifdef __cplusplus
}
#endif

#endif
