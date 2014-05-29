#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

static const char *const RETVAL_PREFIX = "retval:";
static const char *const SLEEP_PREFIX  = "sleep:";
static const char *const STDERR_PREFIX = "stderr:";

static const char *const THREAD_PREFIX = "thread:";
static const char *const THREAD_COMMAND_NEW = "new"; 
static const char *const THREAD_COMMAND_PRINT_IDS = "print-ids"; 

static bool g_print_thread_ids = false;
static pthread_mutex_t g_print_mutex = PTHREAD_MUTEX_INITIALIZER;

static void
print_thread_id ()
{
	// Put in the right magic here for your platform to spit out the thread id (tid) that debugserver/lldb-gdbserver would see as a TID.
	// Otherwise, let the else clause print out the unsupported text so that the unit test knows to skip verifying thread ids.
#if defined(__APPLE__)
	printf ("%" PRIx64, static_cast<uint64_t> (pthread_mach_thread_np(pthread_self())));
#elif defined (__linux__)
	// This is a call to gettid() via syscall.
	printf ("%" PRIx64, static_cast<uint64_t> (syscall (__NR_gettid)));
#else
	printf("{no-tid-support}");
#endif
}

static void
signal_handler (int signo)
{
	switch (signo)
	{
	case SIGUSR1:
		// Print notice that we received the signal on a given thread.
		pthread_mutex_lock (&g_print_mutex);
		printf ("received SIGUSR1 on thread id: ");
		print_thread_id ();
		printf ("\n");
		pthread_mutex_unlock (&g_print_mutex);
		
		// Reset the signal handler.
		sig_t sig_result = signal (SIGUSR1, signal_handler);
		if (sig_result == SIG_ERR)
		{
			fprintf(stderr, "failed to set signal handler: errno=%d\n", errno);
			exit (1);
		}

		break;
	}
}

static void*
thread_func (void *arg)
{
	static int s_thread_index = 1;
	// For now, just sleep for a few seconds.
	// std::cout << "thread " << pthread_self() << ": created" << std::endl;

	const int this_thread_index = s_thread_index++;

	if (g_print_thread_ids)
	{
		pthread_mutex_lock (&g_print_mutex);
		printf ("thread %d id: ", this_thread_index);
		print_thread_id ();
		printf ("\n");
		pthread_mutex_unlock (&g_print_mutex);
	}

	int sleep_seconds_remaining = 20;
	while (sleep_seconds_remaining > 0)
	{
		sleep_seconds_remaining = sleep (sleep_seconds_remaining);
	}

	// std::cout << "thread " << pthread_self() << ": exiting" << std::endl;
	return NULL;
}

int main (int argc, char **argv)
{
	std::vector<pthread_t> threads;
    int return_value = 0;

	// Set the signal handler.
	sig_t sig_result = signal (SIGUSR1, signal_handler);
	if (sig_result == SIG_ERR)
	{
		fprintf(stderr, "failed to set signal handler: errno=%d\n", errno);
		exit (1);
	}
	
	// Process command line args.
    for (int i = 1; i < argc; ++i)
    {
        if (std::strstr (argv[i], STDERR_PREFIX))
        {
            // Treat remainder as text to go to stderr.
            fprintf (stderr, "%s\n", (argv[i] + strlen (STDERR_PREFIX)));
        }
        else if (std::strstr (argv[i], RETVAL_PREFIX))
        {
            // Treat as the return value for the program.
            return_value = std::atoi (argv[i] + strlen (RETVAL_PREFIX));
        }
        else if (std::strstr (argv[i], SLEEP_PREFIX))
        {
            // Treat as the amount of time to have this process sleep (in seconds).
            int sleep_seconds_remaining = std::atoi (argv[i] + strlen (SLEEP_PREFIX));
			
			// Loop around, sleeping until all sleep time is used up.  Note that
			// signals will cause sleep to end early with the number of seconds remaining.
			for (int i = 0; sleep_seconds_remaining > 0; ++i)
			{
				sleep_seconds_remaining = sleep (sleep_seconds_remaining);
				// std::cout << "sleep result (call " << i << "): " << sleep_seconds_remaining << std::endl;
			}
        }
		else if (std::strstr (argv[i], THREAD_PREFIX))
		{
			// Check if we're creating a new thread.
			if (std::strstr (argv[i] + strlen(THREAD_PREFIX), THREAD_COMMAND_NEW))
			{
				// Create a new thread.
				pthread_t new_thread;
				const int err = ::pthread_create (&new_thread, NULL, thread_func, NULL);
			    if (err)
				{
					fprintf (stderr, "pthread_create() failed with error code %d\n", err);
					exit (err);
				}
				threads.push_back (new_thread);
			}
			else if (std::strstr (argv[i] + strlen(THREAD_PREFIX), THREAD_COMMAND_PRINT_IDS))
			{
				// Turn on thread id announcing.
				g_print_thread_ids = true;
				
				// And announce us.
				pthread_mutex_lock (&g_print_mutex);
				printf ("thread 0 id: ");
				print_thread_id ();
				printf ("\n");
				pthread_mutex_unlock (&g_print_mutex);
			}
			else
			{
				// At this point we don't do anything else with threads.
				// Later use thread index and send command to thread.
			}
		}
        else
        {
            // Treat the argument as text for stdout.
            printf("%s\n", argv[i]);
        }
    }

	// If we launched any threads, join them
	for (std::vector<pthread_t>::iterator it = threads.begin (); it != threads.end (); ++it)
	{
		void *thread_retval = NULL;
		const int err = ::pthread_join (*it, &thread_retval);
	    if (err != 0)
			fprintf (stderr, "pthread_join() failed with error code %d\n", err);
	}

    return return_value;
}
