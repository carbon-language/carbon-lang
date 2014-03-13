#include <stdio.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <pthread.h>

void
doing_the_work_1(void *in)
{
    while (1)
        sleep (1);
}

void
submit_work_1a(void *in)
{
    dispatch_queue_t *work_performer_1 = (dispatch_queue_t*) in;
    dispatch_async_f (*work_performer_1, NULL, doing_the_work_1);
    dispatch_async_f (*work_performer_1, NULL, doing_the_work_1);
}

void
submit_work_1b(void *in)
{
    dispatch_queue_t *work_performer_1 = (dispatch_queue_t*) in;
    dispatch_async_f (*work_performer_1, NULL, doing_the_work_1);
    dispatch_async_f (*work_performer_1, NULL, doing_the_work_1);
    while (1)
      sleep (1);
}

void
doing_the_work_2(void *in)
{
    while (1)
        sleep (1);
}

void
submit_work_2(void *in)
{
    dispatch_queue_t *work_performer_2 = (dispatch_queue_t*) in;
    int i = 0;
    while (i++ <  5000)
    {
        dispatch_async_f (*work_performer_2, NULL, doing_the_work_2);
        dispatch_async_f (*work_performer_2, NULL, doing_the_work_2);
    }
}


void
doing_the_work_3(void *in)
{
    while (1)
        sleep(1);
}

void
submit_work_3(void *in)
{
    dispatch_queue_t *work_performer_3 = (dispatch_queue_t*) in;
    dispatch_async_f (*work_performer_3, NULL, doing_the_work_3);
    dispatch_async_f (*work_performer_3, NULL, doing_the_work_3);
    dispatch_async_f (*work_performer_3, NULL, doing_the_work_3);
    dispatch_async_f (*work_performer_3, NULL, doing_the_work_3);
}


void
stopper ()
{
    while (1)
        sleep (1);
}

int main ()
{
    dispatch_queue_t work_submittor_1 = dispatch_queue_create ("com.apple.work_submittor_1", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_t work_submittor_2 = dispatch_queue_create ("com.apple.work_submittor_and_quit_2", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_t work_submittor_3 = dispatch_queue_create ("com.apple.work_submittor_3", DISPATCH_QUEUE_SERIAL);

    dispatch_queue_t work_performer_1 = dispatch_queue_create ("com.apple.work_performer_1", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_t work_performer_2 = dispatch_queue_create ("com.apple.work_performer_2", DISPATCH_QUEUE_SERIAL);

    dispatch_queue_t work_performer_3 = dispatch_queue_create ("com.apple.work_performer_3", DISPATCH_QUEUE_CONCURRENT);

    dispatch_async_f (work_submittor_1, (void*) &work_performer_1, submit_work_1a);
    dispatch_async_f (work_submittor_1, (void*) &work_performer_1, submit_work_1b);

    dispatch_async_f (work_submittor_2, (void*) &work_performer_2, submit_work_2);

    dispatch_async_f (work_submittor_3, (void*) &work_performer_3, submit_work_3);

    sleep (1);
    stopper ();

}

