#include <pthread.h>
#include <unistd.h>

void *
thread_function (void *thread_marker)
{
  int keep_going = 1; 
  int my_value = *((int *)thread_marker);
  int counter = 0;

  while (counter < 20)
    {
      counter++; // Break here in thread body.
      usleep (10);
    }
  return NULL;
}


int 
main ()
{

  pthread_t threads[10];

  int thread_value = 0;
  int i;

  for (i = 0; i < 10; i++)
    {
      thread_value += 1;
      pthread_create (&threads[i], NULL, &thread_function, &thread_value);
    }

  for (i = 0; i < 10; i++)
    pthread_join (threads[i], NULL);

  return 0;
}
