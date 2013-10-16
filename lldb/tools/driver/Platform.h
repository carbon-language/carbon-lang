//===-- Platform.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Platform_h_
#define lldb_Platform_h_

#if defined( _MSC_VER )

    #define PRIu32 "u"
    #define PRId64 "I64d"
    #define PRIi64 "I64i"
    #define PRIo64 "I64o"
    #define PRIu64 "I64u"
    #define PRIx64 "I64x"
    #define PRIX64 "I64X"

    // this will stop signal.h being included
    #define _INC_SIGNAL

    #include <io.h>
    #include <eh.h>
    #include "ELWrapper.h"
    #include "lldb/Host/windows/Windows.h"
    #include "GetOptWrapper.h"

    struct timeval
    {
        long tv_sec;
        long tv_usec;
    };

    struct winsize
    {
        long ws_col;
    };

    typedef unsigned char   cc_t;
    typedef unsigned int    speed_t;
    typedef unsigned int    tcflag_t;

    // fcntl.h
    #define O_NOCTTY 0400

    // ioctls.h
    #define TIOCGWINSZ 0x5413

    // signal.h
    #define SIGPIPE  13
    #define SIGCONT  18
    #define SIGTSTP  20
    #define SIGWINCH 28

    // tcsetattr arguments
    #define TCSANOW 0

    #define NCCS 32
    struct termios
    {
        tcflag_t c_iflag;  // input mode flags
        tcflag_t c_oflag;  // output mode flags
        tcflag_t c_cflag;  // control mode flags
        tcflag_t c_lflag;  // local mode flags
        cc_t c_line;       // line discipline
        cc_t c_cc[NCCS];   // control characters
        speed_t c_ispeed;  // input speed
        speed_t c_ospeed;  // output speed
    };

    typedef long pid_t;

    #define STDIN_FILENO 0

    #define PATH_MAX MAX_PATH
    #define snprintf _snprintf

    extern int  ioctl( int d, int request, ... );
    extern int  kill ( pid_t pid, int sig      );
    extern int  tcsetattr( int fd, int optional_actions, const struct termios *termios_p );
    extern int  tcgetattr( int fildes, struct termios *termios_p );

    // signal handler function pointer type
    typedef void (*sighandler_t)(int);

    // signal.h
    #define SIGINT 2
    // default handler
    #define SIG_DFL ( (sighandler_t) -1 )
    // ignored
    #define SIG_IGN ( (sighandler_t) -2 )
    extern sighandler_t signal( int sig, sighandler_t );

#else

    #include <inttypes.h>

    #include <getopt.h>
    #include <libgen.h>
    #include <sys/ioctl.h>
    #include <termios.h>
    #include <unistd.h>

    #include <histedit.h>
    #include <pthread.h>
    #include <sys/time.h>

    #if defined(__FreeBSD__)
        #include <readline/readline.h>
    #else
        #include <editline/readline.h>
    #endif

#endif

#endif // lldb_Platform_h_
