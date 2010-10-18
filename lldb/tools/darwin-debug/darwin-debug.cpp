//===-- Launcher.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//----------------------------------------------------------------------
// Darwin launch helper
//
// This program was written to allow programs to be launched in a new
// Terminal.app window and have the application be stopped for debugging
// at the program entry point.
//
// Although it uses posix_spawn(), it uses Darwin specific posix spawn
// attribute flags to accomplish its task. It uses an "exec only" flag
// which avoids forking this process, and it uses a "stop at entry"
// flag to stop the program at the entry point.
// 
// Since it uses darwin specific flags this code should not be compiled
// on other systems.
//----------------------------------------------------------------------
#if defined (__APPLE__)

#include <getopt.h>
#include <mach/machine.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _POSIX_SPAWN_DISABLE_ASLR
#define _POSIX_SPAWN_DISABLE_ASLR       0x0100
#endif

#define streq(a,b) strcmp(a,b) == 0

static struct option g_long_options[] = 
{
	{ "arch",			required_argument,	NULL,           'a'		},
	{ "disable-aslr",   no_argument,		NULL,           'd'		},
	{ "no-env",         no_argument,		NULL,           'e'		},
	{ "help",           no_argument,		NULL,           'h'		},
	{ "setsid",         no_argument,		NULL,           's'		},
	{ NULL,				0,					NULL,            0		}
};

static void
usage()
{
    puts (
"NAME\n"
"    darwin-debug -- posix spawn a process that is stopped at the entry point\n"
"                    for debugging.\n"
"\n"
"SYNOPSIS\n"
"    darwin-debug [--arch=<ARCH>] [--disable-aslr] [--no-env] [--setsid] [--help] -- <PROGRAM> [<PROGRAM-ARG> <PROGRAM-ARG> ....]\n"
"\n"
"DESCRIPTION\n"
"    darwin-debug will exec itself into a child process <PROGRAM> that is\n"
"    halted for debugging. It does this by using posix_spawn() along with\n"
"    darwin specific posix_spawn flags that allows exec only (no fork), and\n"
"    stop at the program entry point. Any program arguments <PROGRAM-ARG> are\n"
"    passed on to the exec as the arguments for the new process. The current\n"
"    environment will be passed to the new process unless the \"--no-env\"\n"
"    option is used.\n"
"\n"
"EXAMPLE\n"
"   darwin-debug --arch=i386 -- /bin/ls -al /tmp\n"
);
    exit (1);
}

static void
exit_with_errno (int err, const char *prefix)
{
    if (err)
    {
        fprintf (stderr, 
                 "%s%s", 
                 prefix ? prefix : "", 
                 strerror(err));
        exit (err);
    }
}

pid_t
posix_spawn_for_debug (char *const *argv, char *const *envp, cpu_type_t cpu_type, int disable_aslr)
{
    pid_t pid = 0;

    const char *path = argv[0];

    posix_spawnattr_t attr;

    exit_with_errno (::posix_spawnattr_init (&attr), "::posix_spawnattr_init (&attr) error: ");

    // Here we are using a darwin specific feature that allows us to exec only
    // since we want this program to turn into the program we want to debug, 
    // and also have the new program start suspended (right at __dyld_start)
    // so we can debug it
    short flags = POSIX_SPAWN_START_SUSPENDED | POSIX_SPAWN_SETEXEC;

    // Disable ASLR if we were asked to
    if (disable_aslr)
        flags |= _POSIX_SPAWN_DISABLE_ASLR;
    
    // Set the flags we just made into our posix spawn attributes
    exit_with_errno (::posix_spawnattr_setflags (&attr, flags), "::posix_spawnattr_setflags (&attr, flags) error: ");
    
    
    // Another darwin specific thing here where we can select the architecture
    // of the binary we want to re-exec as.
    if (cpu_type != 0)
    {
        size_t ocount = 0;
        exit_with_errno (::posix_spawnattr_setbinpref_np (&attr, 1, &cpu_type, &ocount), "posix_spawnattr_setbinpref_np () error: ");
    }

    exit_with_errno (::posix_spawnp (&pid, path, NULL, &attr, (char * const*)argv, (char * const*)envp), "posix_spawn() error: ");
    
    // This code will only be reached if the posix_spawn exec failed...
    ::posix_spawnattr_destroy (&attr);

    return pid;
}


int main (int argc, char *const *argv, char *const *envp, const char **apple)
{
    const char *program_name = strrchr(apple[0], '/');
    
    if (program_name)
        program_name++; // Skip the last slash..
    else
        program_name = apple[0];
    
#if defined (DEBUG_LLDB_LAUNCHER)
    printf("%s called with:\n", program_name);
    for (int i=0; i<argc; ++i)
        printf("argv[%u] = '%s'\n", i, argv[i]);
#endif

    cpu_type_t cpu_type = 0;
    bool show_usage = false;
    char ch;
    int disable_aslr = 0; // By default we disable ASLR
    int pass_env = 1;
	while ((ch = getopt_long(argc, argv, "a:dfh?", g_long_options, NULL)) != -1)
	{
		switch (ch) 
		{
        case 0:
            break;

		case 'a':	// "-a i386" or "--arch=i386"
			if (optarg)
			{
				if (streq (optarg, "i386"))
                    cpu_type = CPU_TYPE_I386;
				else if (streq (optarg, "x86_64"))
                    cpu_type = CPU_TYPE_X86_64;
                else if (strstr (optarg, "arm") == optarg)
                    cpu_type = CPU_TYPE_ARM;
                else
                {
                    ::fprintf (stderr, "error: unsupported cpu type '%s'\n", optarg);
                    ::exit (1);
                }
			} 
			break;

        case 'd':
            disable_aslr = 1;
            break;            

        case 'e':
            pass_env = 0;
            break;
        
        case 's':
            // Create a new session to avoid having control-C presses kill our current
            // terminal session when this program is launched from a .command file
            ::setsid();
            break;

		case 'h':
		case '?':
		default:
			show_usage = true;
			break;
		}
	}
	argc -= optind;
	argv += optind;

    if (show_usage || argc <= 0)
        usage();

#if defined (DEBUG_LLDB_LAUNCHER)
    printf ("\n%s post options:\n", program_name);
    for (int i=0; i<argc; ++i)
        printf ("argv[%u] = '%s'\n", i, argv[i]);
#endif

    posix_spawn_for_debug (argv, 
                           pass_env ? envp : NULL, 
                           cpu_type, 
                           disable_aslr);
    
	return 0;
}

#endif // #if defined (__APPLE__)

