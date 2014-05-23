#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <vector>

static const char *const RETVAL_PREFIX = "retval:";
static const char *const SLEEP_PREFIX  = "sleep:";
static const char *const STDERR_PREFIX = "stderr:";

static const char *const THREAD_PREFIX = "thread:";
static const char *const THREAD_COMMAND_NEW = "new"; 

static void*
thread_func (void *arg)
{
	// For now, just sleep for a few seconds.
	// std::cout << "thread " << pthread_self() << ": created" << std::endl;

	int sleep_seconds_remaining = 5;
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

    for (int i = 1; i < argc; ++i)
    {
        if (std::strstr (argv[i], STDERR_PREFIX))
        {
            // Treat remainder as text to go to stderr.
            std::cerr << (argv[i] + strlen (STDERR_PREFIX)) << std::endl;
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
					std::cerr << "pthread_create() failed with error code " << err << std::endl;
					exit (err);
				}
				threads.push_back (new_thread);
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
            std::cout << argv[i] << std::endl;
        }
    }

	// If we launched any threads, join them
	for (std::vector<pthread_t>::iterator it = threads.begin (); it != threads.end (); ++it)
	{
		void *thread_retval = NULL;
		const int err = ::pthread_join (*it, &thread_retval);
	    if (err != 0)
		{
			std::cerr << "pthread_join() failed with error code " << err << std::endl;
		}
	}

    return return_value;
}
