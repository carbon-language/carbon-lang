// clang-format off
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ittnotify_config.h"

#if ITT_PLATFORM == ITT_PLATFORM_WIN

#pragma warning(disable: 593) /* parameter "XXXX" was set but never used */
#pragma warning(disable: 344) /* typedef name has already been declared (with
                                  same type) */
#pragma warning(disable: 174) /* expression has no effect */
#pragma warning(disable: 4127) /* conditional expression is constant */
#pragma warning(disable: 4306) /* conversion from '?' to '?' of greater size */

#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

#if defined __INTEL_COMPILER

#pragma warning(disable: 869) /* parameter "XXXXX" was never referenced */
#pragma warning(disable: 1418) /* external function definition with no prior
                                  declaration  */
#pragma warning(disable: 1419) /* external declaration in primary source file */

#endif /* __INTEL_COMPILER */
