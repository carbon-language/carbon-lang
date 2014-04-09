//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//



#include <stdio.h>

// ===========================================================================
// Bring in the static string table and the enumerations for indexing into
// it.
// ===========================================================================

#include "liboffload_msg.h"

# define DYNART_STDERR_PUTS(__message_text__) fputs((__message_text__),stderr)

// ===========================================================================
// Now the code for accessing the message catalogs
// ===========================================================================


    void write_message(FILE * file, int msgCode) {
        fputs(MESSAGE_TABLE_NAME[ msgCode ], file);
        fflush(file);
    }

    char const *offload_get_message_str(int msgCode) {
        return MESSAGE_TABLE_NAME[ msgCode ];
    }
